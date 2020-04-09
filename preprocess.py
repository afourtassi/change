"""Divide corpus of one language to N epochs (parts) with similar sizes

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

random.seed(123)


def read_docs(csv_file, column='stem'):
    """read stem utterances from childes csv files"""

    df = pd.read_csv(csv_file)
    if len(df) == 0:
        return []
    tags = df['part_of_speech'].values
    stems = df['stem'].values
    ret_list = []
    for t, s in zip(tags, stems):
        tl, sl = str(t).lower().split(), str(s).lower().split()
        
        # replace NAME and interjections with $name$ and (maybe) $co$ respectively
        ntl = []
        for t, s in zip(tl, sl):
            if t == "n:prop":
                ntl.append('$name$')
#             elif t == 'co':
#                 ntl.append('$co$')
            else:
                ntl.append(s)

        ret_list.append(ntl)
    return ret_list


def get_token_num_per_file(childes_files):
    """get a list of token numbers per file"""

    num_tokens = []
    for filename in sorted(childes_files, key=lambda x: int(x.split('_')[-1][:-4])):
        month = int(filename.split('_')[-1][:-4])
        lines = read_docs(filename)
        num_tokens.append(sum([len(l) for l in lines]))
    print(num_tokens)
    return num_tokens


def divide_corpus(num_tokens, threshold_num):
    """divide num_tokens into N parts/epochs using threshold_num"""
    window = []
    months = []
    periods = []

    for i, num in enumerate(num_tokens):
        if num != 0:
            window.append(num)
            months.append(i)
        if sum(window) >= threshold_num:
            periods.append(months)
            print(sum(window), months, sum(months)/len(months))
            window = []
            months = []

    # when break out
    periods.append(months)
    print(sum(window), months, sum(months)/len(months))
    return periods


def aggregate_corpus_by_periods(childes_files, periods, dest_dir):
    """aggregate corpus into big chunk files by periods"""

    for i, period in enumerate(periods):
        # print(np.array(childes_files)[period])
        print('key', 'period'+str(i))
        sents = []
        for filename in np.array(childes_files)[period]:
            sents.extend(read_docs(filename))


        # write docs into one file
        output_filename = "period_{}.txt".format(i)
        with open(os.path.join(dest_dir, output_filename), 'w') as out:
            for sent in sents:
                sent = ' '.join(sent)
                if sent.strip() and sent != "nan":
                    out.write(sent)
                    out.write("\n")


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./data/german/raw", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./data/german/proc", help="dest dir")

    # args for dividing the corpus
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs (parts) we divide the corpus.')
    parser.add_argument('--token_num_per_epoch', type=float, default=None, help='Number of tokens per epoch. Set None by default, when used, we stop using num_epochs.')

    # args for 
    args = parser.parse_args()

    # get all raw files
    childes_files = sorted(glob.glob(os.path.join(args.source_dir, "period*.csv")), 
                            key=lambda x: int(x.split('_')[-1][:-4]))

    # get num_tokens list
    num_tokens = get_token_num_per_file(childes_files)

    # mkdir output dir
    Path(args.dest_dir).mkdir(parents=True, exist_ok=True)

    if args.token_num_per_epoch:
        threshold_num = args.token_num_per_epoch
    else:
        threshold_num = int(sum(num_tokens) / args.num_epochs)

    # get periods for each epoch
    periods = divide_corpus(num_tokens, threshold_num)

    # generate output files
    aggregate_corpus_by_periods(childes_files, periods, args.dest_dir)



if __name__ == "__main__":
    main()

