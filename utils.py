"""Utils.py: for now, generate log_freq.csv and poly.csv

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
import nltk
from nltk.corpus import wordnet as wn


def read_docs(csv_file, use_stem=False):
    """read stem utterances from childes csv files"""

    df = pd.read_csv(csv_file)
    if len(df) == 0:
        return []

    if use_stem:
        stems = df['stem'].values
        tags = df['part_of_speech'].values
        ret_list = []
        for t, s in zip(tags, stems):
            tl, sl = str(t).lower().split(), str(s).lower().split()
            
            # replace NAME and interjections with $name$ and (maybe) $co$ respectively
            ntl = []
            for t, s in zip(tl, sl):
                if t == "n:prop":
                    ntl.append('$name$')
                else:
                    ntl.append(s)
            if ntl != ['nan']: # add as a sentence only if we have meaningful input
                ret_list.append(ntl)
    else:
        # use only gloss
        sents = df['gloss'].values
        ret_list = []
        for s in sents:
            ntl = str(s).lower().split()
            ret_list.append(ntl)

    return ret_list


def create_freq_file(childes_files, use_stem, dest_dir):
    """create log freq file for the whole corpus"""
    wordCounter = Counter()
    for filename in sorted(childes_files, key=lambda x: int(x.split('_')[-1][:-4])):
        lines = read_docs(filename, use_stem)
        for line in lines:
            wordCounter.update(line)

    rows = []
    for word, freq in wordCounter.most_common():
        if freq >= 10:
            rows.append([word, np.log10(freq)])

    df = pd.DataFrame(rows, columns=['uni_lemma', 'freq'])
    df.to_csv(os.path.join(dest_dir, 'log_freq.csv'))


def polysemy(word, language):
    return len(wn.synsets(word, lang=language))


def calculate_polysemy(csv_file_path, dest_dir, language):
    """add polysemy column to the original inventory csv file"""

    language2acronym = {
        "Chinese": "cmn",
        "French": "fra",
        "Japanese": "jpn",
        "Spanish": "spa",
        "English": "eng"
    }

    df = pd.read_csv(csv_file_path)
    poly_levels = []
    wn_lemmas = wn.all_lemma_names(lang=language2acronym[language])
    wn_lemmas = set([w.lower() for w in wn_lemmas])
    for word in df.word.values:
        if word in wn_lemmas:
            poly_levels.append(polysemy(word, language2acronym[language]))
        else:
            poly_levels.append(0)
    df['polysemy'] = poly_levels
    df.to_csv(os.path.join(dest_dir, "merged_cdi_polysemy.csv"))


def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./data/english-na/raw", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./output/english-na/", help="dest dir")
    parser.add_argument("--word_inventory", type=str, default="./data/english-na/word_inventory.csv", help="specify which embedding file to use")
    parser.add_argument("--use_stem", action="store_true", help="default is to use `gloss`, otherwise use stem")
    parser.add_argument("--language", type=str, default="English", help="English, French, Spanish, Chinese")

    # args
    args = parser.parse_args()

    # get all raw files
    childes_files = sorted(glob.glob(os.path.join(args.source_dir, "period*.csv")), 
                            key=lambda x: int(x.split('_')[-1][:-4]))

    # create log frequency file for the language
    _ = create_freq_file(childes_files, args.use_stem, args.dest_dir)

    # write (adding polysemy) a new inventory csv file
    _ = calculate_polysemy(args.word_inventory, args.dest_dir, args.language)


if __name__ == "__main__":
    main()
