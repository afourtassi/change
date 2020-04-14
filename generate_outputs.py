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


def neighbors(query : str, embs: np.ndarray, vocab: list, K : int = 3) -> list:
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


def compute_embedding_shifts(year1, year2, vocs, vecs, emb_shift_file_path, num_neighbors):
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


def get_word_change_over_time(model, query, years, vocs, vecs, num_neighbors):
    """get the query and its neighbors cosine similarity in Google's
        pretrained word2vec vector space
        if not exist, the sim is 0
    """

    word_gold_sim = []
    
    if query in model.vocab:
        for period in years:
            if query not in vocs[period]:
                word_gold_sim.append(0)
                continue
            neighbor_words = neighbors(query, vecs[period], vocs[period], K=num_neighbors)
            sims = [model.similarity(query, w) for w in neighbor_words if w in model.vocab]
            if sims:
                word_gold_sim.append(sum(sims) / len(sims))
            else:
                word_gold_sim.append(0)
    return word_gold_sim


def get_w2v_sim(df_inventory, column_name, model, years, vocs, vecs, num_neighbors):
    """get semantic changes"""
    gold_change_per_category = defaultdict(list)
    for group_name_tmp, group_df in df_inventory.groupby([column_name]):
        gold_changes = []
        for word in group_df.word.values:
            gold_sims = get_word_change_over_time(model, word, years, vocs, vecs, num_neighbors)
            if gold_sims:
                gold_changes.append(gold_sims)
            else:
                gold_changes.append([0]*len(years))
        gold_change_per_category[group_name_tmp] = np.array(gold_changes)
    return gold_change_per_category


def save_changes_to_csv(output_dir, df_inventory, model, years, vocs, vecs, num_neighbors):
    """save each word's W2V-SIM changes in all periods to csv"""
    data = []
    column_name = 'category'  # use `category` to changes per group (using `lexical` is also fine, either works)
    gold_change_per_category = get_w2v_sim(df_inventory, column_name, model, years, vocs, vecs, num_neighbors)
    for group_name, group_df in df_inventory.groupby([column_name]):
        for word, sims in zip(group_df.word.values, gold_change_per_category[group_name]):
            lexical_class = df_inventory[df_inventory.word == word].values[0][-1]
            row = []
            row.append(word)
            row.append(group_name)
            row.append(lexical_class)
            row.extend(sims)
            data.append(row)
    columns = ['word', 'category', 'lexical_class'] + years
    df_change = pd.DataFrame(data, columns=columns)
    output_path = os.path.join(output_dir, "google-w2v-changes-K{}.csv".format(num_neighbors))
    df_change.to_csv(output_path)
    return df_change


def compute_distances_per_period(df_inventory, vocs, vecs, period):
    """compute all sims between words in the word inventory for one period"""
    intersected_words = set(df_inventory['word'].values).intersection(vocs[period])
    voc_cdi = [w for w in vocs[period] if w in intersected_words]
    vecs_cdi = np.array([v for i, v in enumerate(vecs[period]) if vocs[period][i] in intersected_words])
    length = len(intersected_words)
    assert len(voc_cdi) == len(vecs_cdi)
    distances_period = []
    w1_period = []
    w2_period = []
    for i in range(length):
        distances = compute_cos_dist(np.tile(vecs_cdi[i], (length,1)), vecs_cdi)
        distances_period.extend(distances)
        w1_period.extend([voc_cdi[i]]*length)
        w2_period.extend(voc_cdi)
    return distances_period, w1_period, w2_period


def run_word_inventory_sims_across_periods(df_inventory, vocs, vecs, years, output_dir):
    for period in years:
        print(period)
        distances_period, w1_period, w2_period = compute_distances_per_period(df_inventory, vocs, vecs, period)
        df_cdi_product_distance = pd.DataFrame({
            'word1': w1_period,
            'word2': w2_period,
            'cos_dist': distances_period
        })
        output_file_path = os.path.join(output_dir, "{}_word_inventory_distance.csv".format(period))
        df_cdi_product_distance.to_csv(output_file_path)


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="the output generation script")
    parser.add_argument("--source_dir", type=str, default="./output/english-uk/", help="source dir")
    parser.add_argument("--embedding_filename", type=str, default="embeddings-ep2-f15-d100-w5.pickle", help="specify which embedding file to use")
    parser.add_argument("--word_inventory", type=str, default="./data/english-uk/word_inventory.csv", help="specify which embedding file to use")
    parser.add_argument("--google_word2vec", type=str, default="./data/google-word2vec/GoogleNews-vectors-negative300.bin", help="specify where google word2vec file is")

    # args for word2vec model
    parser.add_argument('--num_neighbors', type=int, default=25, help='Number of neighbors used for local measure. Default is 25.')

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
        df_emb_shift = compute_embedding_shifts(first_year, last_year, year2vocab, year2embedding, emb_shift_file_path, args.num_neighbors)

        # compute semantic changes
        df_inventory =  pd.read_csv(args.word_inventory)
        print("intersection: ", len(set(df_inventory['word'].values).intersection(df_emb_shift.index)))
        # cat_local_change_per_category, \
        #     cat_global_change_per_category = compute_semantic_change(df_inventory, df_emb_shift, 'category')
        # lex_local_change_per_category, \
        #     lex_global_change_per_category = compute_semantic_change(df_inventory, df_emb_shift, 'lexical_class')

        output_dir = os.path.join(args.source_dir, folder_name)

        # compute local cosine similarities b/w google-word2vec and our embeddings
        google_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.google_word2vec, binary=True)
        save_changes_to_csv(output_dir, df_inventory, google_w2v_model, years, year2vocab, year2embedding, args.num_neighbors)

        # compute cosine distance between word inventory words
        run_word_inventory_sims_across_periods(df_inventory, year2vocab, year2embedding, years, output_dir)


if __name__ == "__main__":
    main()
