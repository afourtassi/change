"""Compute output files using embeddings for one single language

Author: Hang Jiang (hjian42@icloud.com)
"""
import sys
import os
import argparse
import glob
import gensim
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from collections import defaultdict, Counter
from scipy import spatial
from pathlib import Path
from typing import List, Dict, Tuple, Sequence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sparse


def neighbors(query : str,
              embs: np.ndarray,
              vocab: list,
              K : int = 3) -> list:
    """returns k nearest neighbors for a query"""
    sims = np.dot(embs[vocab.index(query),],embs.T)
    output = []
    for sim_idx in sims.argsort()[::-1][1:(1+K)]:
        if sims[sim_idx] > 0:
            output.append(vocab[sim_idx])
    return output


def compute_cos_dist(vec1 : np.array, vec2 : np.array) -> np.array:
    """compute cosine distance between two arrays (np.array)"""
    numerator = (vec1 * vec2).sum(1)
    denominator = np.sqrt((vec1**2).sum(1)) * np.sqrt((vec2**2).sum(1))
    cos_dist = 1 - numerator / (1e-5 + denominator)

    # eliminate words that are zeroed out
    return cos_dist * (vec1.var(1) > 0) * (vec2.var(1) > 0)


def get_neighbor_sims(query : str, neighbor_set : set, vec : np.ndarray, voc : list) -> np.ndarray:
    """get cosine similarity list between query and neighbors"""
    v_self = vec[voc.index(query),]
    v_neighbors = vec[[voc.index(neighbor) for neighbor in neighbor_set],]
    return np.dot(v_neighbors, v_self)


def neighbors_per_year(query: str, K=5) -> dict:
    """returns query's neighbors at each period"""
    nns = dict()
    for year, embs in vecs.items():
        nns[year] = neighbors(query, embs, vocs[year], K=K)
    return nns


def build_alignment(year1 : str, year2 : str, vocs, vecs) -> sparse.csr_matrix:
    """
    given two years (keys into vecs and vocs)
    compute a sparse matrix M representing the permutation of word indices from the year1 vecs to the year2 vecs  
    then the rows of v1 will be aligned with the rows of M v2 
    """
   
    all_var1 = vecs[year1].var(1)
    all_var2 = vecs[year2].var(1)

    rows = []
    cols = []

    ivoc2 = {j:i for i,j in enumerate(vocs[year2])}

    for widx1,word in enumerate(vocs[year1]):
        if all_var1[widx1] != 0 and word in set(vocs[year2]):
            widx2 = ivoc2[word]
            if all_var2[widx2] != 0:
                rows.append(widx1)
                cols.append(widx2)

    align = sparse.csr_matrix(([1]*len(rows),(rows,cols)),shape=[len(vocs[year1]),len(vocs[year2])])
    return align


def procrustes(A, B):
    U, _, Vt = np.linalg.svd(B.dot(A.T))
    return U.dot(Vt)


def load_embeddings(embedding_file_path, args):
    """
    Load and normalize embeddings

    returns
        year2vocab: dict of year to vocabulary (list) 
        years: list of year/epoch names (str)
        year2embedding: dict of year to embeddeings (np.array)
    """
    dicts = pickle.load(open(embedding_file_path,'rb'))

    year2vocab: Dict[str, List] = {k: list(v.keys()) for k, v in dicts.items()}
    years: List[str] = list(year2vocab.keys())

    # set the vocab to be the smallest of the epochs
    # TODO: make this part less heuristic (share common words first then cut off)
    vocab_size = min([len(l) for k, l in year2vocab.items()])

    for key, item in year2vocab.items():
        print(key, "First 5", item[:5], " Size: ", len(item))
        year2vocab[key] = item[:vocab_size]
    year2embedding: Dict[str, List] = dict()

    for year, d in dicts.items():
        year_vectors = np.array([list(d.values())]).reshape(-1, 100)[:vocab_size]
        print(year, year_vectors.shape)
        year2embedding[year] = year_vectors / np.linalg.norm(year_vectors, axis=-1)[:, np.newaxis]

    # sanity check
    print((year2embedding['period1']**2).sum(1))
    return year2vocab, years, year2embedding


def compute_global_cos_dist(year1, year2, vocs, vecs):
    """compute cos dist for global measure"""

    align = build_alignment(year2, year1, vocs, vecs)
    print("align", align.shape)
    vecs_aligned = align.dot(vecs[year1])
    print("vecs_aligned", vecs_aligned.shape)

    Omega = procrustes(vecs_aligned.T, vecs[year2].T)
    vecs_projected = Omega.dot(vecs_aligned.T).T
    print("vecs_projected", vecs_projected.shape)

    length_of_vocab = min(vecs[year1].shape[0], vecs[year2].shape[0])
    print(length_of_vocab)
    print('pre-alignment: ',np.linalg.norm(vecs[year2][:length_of_vocab] - vecs[year1]))
    print('pre-projection: ',np.linalg.norm(vecs[year2] - vecs_aligned))
    print('after projection: ',np.linalg.norm(vecs[year2] - vecs_projected))

    cos_dist = compute_cos_dist(vecs[year2], vecs_projected)
    print("cos_dist: ", cos_dist.shape, cos_dist[:10])
    return cos_dist


def compute_embedding_shifts(year1, year2, vocs, vecs, emb_shift_file_path, num_neighbors=25):
    """compute shift with both local and global metrics for all the words
        on one corpus
        
        year1: first year
        year2; last year
        vecs: year2embedding
        vocs: year2vocab
    
    returns
        df_emb_shift: df
    """
    # this may take 5-10 minutes to execute
    neighbor_shift = dict()

    # compute global_cos_dist
    global_cos_dist = compute_global_cos_dist(year1, year2, vocs, vecs)

    # compute embedding_shifts
    for word in tqdm(vocs[year2]):
        if word in vocs[year1]:
            nn_old = neighbors(word, vecs[year1], vocs[year1], K=num_neighbors)
            nn_new = neighbors(word, vecs[year2], vocs[year2], K=num_neighbors)

            neighbor_set = [word for word in set(nn_old).union(set(nn_new)) 
                            if (word in vocs[year1])
                            and (word in vocs[year2])
                            and (vecs[year1][vocs[year1].index(word)].var() > 0)
                            and (vecs[year2][vocs[year2].index(word)].var() > 0)]
        
            s1 = get_neighbor_sims(word, neighbor_set, vecs[year1], vocs[year1])
            s2 = get_neighbor_sims(word, neighbor_set, vecs[year2], vocs[year2])

            dL = compute_cos_dist(s1.reshape(1,-1),s2.reshape(1,-1))[0] / len(neighbor_set)
    
            neighbor_shift[word] = dL
    df_emb_shift = pd.DataFrame({'local':list(neighbor_shift.values()),
                                 'global':[global_cos_dist[vocs[year2].index(word)] for word in neighbor_shift.keys()]},
                                index=list(neighbor_shift.keys()))
    df_emb_shift.to_csv(emb_shift_file_path)

    print(df_emb_shift.shape)
    print(df_emb_shift.head())

    return df_emb_shift


def compute_semantic_change(df_inventory, df_emb_shift, category_name):
    """depends on df_inventory and df_emb_shift"""
    global_change_per_category = defaultdict(list)
    local_change_per_category = defaultdict(list)
    for group_name, group_df in df_inventory.groupby([category_name]):
    #     print(group_name, group_df.definition.values)
        local_changes = []
        global_changes = []
        for word in group_df.word.values:
            if word in df_emb_shift.index:
                local_changes.append(df_emb_shift.loc[word]['local'])
                global_changes.append(df_emb_shift.loc[word]['global'])
        local_change_per_category[group_name] = sum(local_changes)/len(local_changes)
        global_change_per_category[group_name] = sum(global_changes)/len(global_changes)
    return local_change_per_category, global_change_per_category


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./output/english-uk/", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./output/english-uk/", help="dest dir")
    parser.add_argument("--embedding_filename", type=str, default="embeddings-ep2-f15-d100-w5.pickle", help="specify which embedding file to use")
    parser.add_argument("--word_inventory", type=str, default="./data/english-uk/word_inventory.csv", help="specify which embedding file to use")

    # args for word2vec model
    parser.add_argument('--num_neighbors', type=int, default=25, help='Number of neighbors used for local measure. Default is 25.')
    # parser.add_argument('--min_count', type=int, default=15, help='Min frequency cutoff, only words more than this will be used for training.')
    # parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs. Default is 100.')
    # parser.add_argument('--window', type=int, default=5, help='Window size for Skip-gram. Default is 5.')
    # parser.add_argument('--negative', type=int, default=5, help='Number of negative samples. Default is 5.')
    # parser.add_argument('--sg', type=int, default=1, help='0 for bow, 1 for skip-gram.')
    # parser.add_argument('--sample', type=float, default=1e-5, help='The threshold for configuring which higher-frequency words are randomly downsampled.')
    # parser.add_argument('--ns_exponent', type=float, default=0.75, help='The exponent used to shape the negative sampling distribution.')
    # parser.add_argument('--workers', type=int, default=4, help='# of workers for training (=faster training with multicore machines)')

    # args for 
    args = parser.parse_args()

    # get proc and shuffle folders
    folder_names = []
    for file in os.listdir(args.source_dir):
        folder_path = os.path.join(args.source_dir, file)
        if os.path.isdir(folder_path):
            if file != "raw":
                folder_names.append(file)

    for folder_name in folder_names:
        embedding_file_path = os.path.join(args.source_dir, folder_name, "embeddings-over-time", args.embedding_filename)
        print(embedding_file_path)

        # load and normalize embedding file
        year2vocab, years, year2embedding = load_embeddings(embedding_file_path, args)

        # compute embedding-shifts.csv between the first and last epochs
        years = sorted(years)
        first_year, last_year = years[0], years[-1]
        emb_shift_file_path = os.path.join(args.source_dir, folder_name, "embedding-shifts-K{}.csv".format(args.num_neighbors))
        df_emb_shift = compute_embedding_shifts(first_year, last_year, year2vocab, year2embedding, emb_shift_file_path, num_neighbors=args.num_neighbors)

        # compute semantic changes
        df_inventory =  pd.read_csv(args.word_inventory)
        cat_local_change_per_category, \
            cat_global_change_per_category = compute_semantic_change(df_inventory, df_emb_shift, 'category')
        lex_local_change_per_category, \
            lex_global_change_per_category = compute_semantic_change(df_inventory, df_emb_shift, 'lexical_class')
        print("intersection: ", len(set(df_inventory['word'].values).intersection(df_emb_shift.index)))

        # TODO: compute dist b/w google-word2vec and our embeddings

        break
        # save dictionary in dest directory
        # output_folder = os.path.join(args.dest_dir, folder_name, "embeddings-over-time")
        # embedding_filename = "embeddings-ep{}-f{}-d{}-w{}.pickle".format(args.n_epochs, args.min_count, args.dim, args.window)
        # Path(output_folder).mkdir(parents=True, exist_ok=True)
        # with open(os.path.join(output_folder, embedding_filename), 'wb') as handle:
        #     pickle.dump(year2vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
