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
        if ntl != ['nan']: # add as a sentence only if we have meaningful input
            ret_list.append(ntl)
    return ret_list


def write_sentences(sentences, output_path):
    """write sentences to a file"""
    with open(output_path, "w") as out:
        for sent in sentences:
            sent = ' '.join(sent)
            if sent.strip():
                out.write(sent)
                out.write("\n")


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

    controlled_token_nums = [] 

    for i, num in enumerate(num_tokens):
        if num != 0:
            window.append(num)
            months.append(i)
        if sum(window) >= threshold_num:
            periods.append(months)
            controlled_token_nums.append(sum(window))
            print(sum(window), months, sum(months)/len(months))
            window = []
            months = []

    # when break out without recording
    if sum(window) < threshold_num:
        periods.append(months)
        controlled_token_nums.append(sum(window))
        print(sum(window), months, sum(months)/len(months))
        print("controlled_token_nums")
        print(controlled_token_nums)
    return periods, controlled_token_nums


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
        write_sentences(sents, os.path.join(dest_dir, output_filename))


def create_shuffled_divided_corpus(childes_files, num_tokens, controlled_token_nums, dest_dir, shuffle_id):
    """use controlled_token_nums to create shuffled corpus of similar sizes"""
    
    # make directory for the shuffle
    shuffled_folder = os.path.join(dest_dir, "shuffle_{}".format(shuffle_id))
    Path(shuffled_folder).mkdir(parents=True, exist_ok=True)

    # get all sentences
    sentences = []  
    for filename in childes_files:
        sentences.extend(read_docs(filename))

    print("Before shuffling:")
    print(sentences[0])
    random.shuffle(sentences)
    print("After shuffling")
    print(sentences[0])
        
    # start index of sentences
    start = 0
    for i, window_size in enumerate(controlled_token_nums):
        num_tokens = 0
        sents_per_period = []
        
        for j, sentence in enumerate(sentences[start:]):
            sents_per_period.append(sentence)
            num_tokens += len(sentence)
            if num_tokens >= window_size:

                # write down the chunk of corpus
                output_filename = "period_{}.txt".format(i)
                print(output_filename, "corpus size: ", num_tokens)
                write_sentences(sents_per_period, os.path.join(shuffled_folder, output_filename))
                
                # update start
                start += (j+1)
                break

        # when breakout without recording
        if num_tokens < window_size:
            output_filename = "period_{}.txt".format(i)
            print(output_filename, "corpus size: ", num_tokens)
            write_sentences(sents_per_period, os.path.join(shuffled_folder, output_filename))


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./data/english-uk/raw", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./data/english-uk/", help="dest dir")

    # args for dividing the corpus
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs (parts) we divide the corpus.')
    parser.add_argument('--token_num_per_epoch', type=float, default=None, help='Number of tokens per epoch. Set None by default, when used, we stop using num_epochs.')
    parser.add_argument('--num_shuffles', type=int, default=1, help='Number of shuffling.')

    # args
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

    ### generate divided original data
    periods, controlled_token_nums = divide_corpus(num_tokens, threshold_num)
    proc_folder = os.path.join(args.dest_dir, 'proc')
    Path(proc_folder).mkdir(parents=True, exist_ok=True)
    aggregate_corpus_by_periods(childes_files, periods, proc_folder)

    ## generate divided shuffled data
    for shuffle_id in range(args.num_shuffles):
        print("shuffle_{}".format(shuffle_id))
        create_shuffled_divided_corpus(childes_files, num_tokens, \
            controlled_token_nums, args.dest_dir, shuffle_id)


if __name__ == "__main__":
    main()

