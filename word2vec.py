"""Train dynamic word embeddings with word2vec model 
    for one single language at a time

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
from collections import defaultdict
from scipy import spatial
from collections import Counter
from pathlib import Path


def train_word2vec(documents, args):
    """
    Train one word2vec model on a corpus and returns embeddings (dict)
    """
    model = gensim.models.Word2Vec(
        documents,
        sg=args.sg, 
        size=args.dim,
        window=args.window,
        min_count=args.min_count,
        sample=args.sample,
        iter=args.n_epochs,
        ns_exponent=args.ns_exponent,
        negative=args.negative,
        workers=args.workers)
    model.train(documents, total_examples=len(documents), epochs=model.epochs)
    embedding_dict = {w:v for w, v in zip(model.wv.index2word, model.wv.vectors)}
    return embedding_dict


def train_dynamic_word_embeddings(epoch_files, args):
    """train miltiple vector space models on each epoch of the data
    
    returns a dictionry of embeddings 
    """
    year2vecs = {}
    for i, epoch_file in enumerate(epoch_files):
        with open(epoch_file) as f:
            sentences = f.readlines()
            sentences = [s.strip().split() for s in sentences]

            # train word embeddings on one chunk of data (one epoch)
            embedding_dict = train_word2vec(sentences, args)
            year2vecs['period'+str(i)] = embedding_dict
    return year2vecs


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./data/english-uk/", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./output/english-uk/", help="dest dir")

    # args for word2vec model
    parser.add_argument('--dim', type=int, default=100, help='Number of dimensions. Default is 100.')
    parser.add_argument('--min_count', type=int, default=15, help='Min frequency cutoff, only words more than this will be used for training.')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs. Default is 2.')
    parser.add_argument('--window', type=int, default=5, help='Window size for Skip-gram. Default is 5.')
    parser.add_argument('--negative', type=int, default=5, help='Number of negative samples. Default is 5.')
    parser.add_argument('--sg', type=int, default=1, help='0 for bow, 1 for skip-gram.')
    parser.add_argument('--sample', type=float, default=1e-5, help='The threshold for configuring which higher-frequency words are randomly downsampled.')
    parser.add_argument('--ns_exponent', type=float, default=0.75, help='The exponent used to shape the negative sampling distribution.')
    parser.add_argument('--workers', type=int, default=4, help='# of workers for training (=faster training with multicore machines)')

    # args 
    args = parser.parse_args()

    # get proc and shuffle folders
    folder_names = []
    for file in os.listdir(args.source_dir):
        folder_path = os.path.join(args.source_dir, file)
        if os.path.isdir(folder_path):
            if file != "raw":
                folder_names.append(file)

    for folder_name in folder_names:
        epoch_files = sorted(glob.glob(os.path.join(args.source_dir, folder_name, "period_*.txt")))
        print(epoch_files)

        # store embeddings in dictionary
        year2vecs = train_dynamic_word_embeddings(epoch_files, args)

        # save dictionary in dest directory
        output_folder = os.path.join(args.dest_dir, folder_name, "embeddings-over-time")
        embedding_filename = "embeddings-ep{}-f{}-d{}-w{}.pickle".format(args.n_epochs, args.min_count, args.dim, args.window)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_folder, embedding_filename), 'wb') as handle:
            pickle.dump(year2vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
