# Exploring patterns of stability and change in caregivers' word usage across early childhood

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements dynamic word embeddings for exploring the language variations of caregivers across multiple languages in CHILDES dataset. 

### Setup Environment

#### Install via conda:
1. install requirements </br>
   ```pip install -r requirements.txt```

### Download data

Run the commands to download the data from Google Drive and unzip it to be `data`
   ```shell
   python download_gdrive.py 1NXMPs_f1crasC1JM50RSIJweLuPAG9Hk data.zip
   unzip data.zip data
   ```

### Train dynamic word embeddings

We take the `english-uk` language for example, whose raw csv files sit in `./data/german/raw/`. We also put the pre-computed `word_inventory.csv` for each language inside `./data/{language}/`. German and Japanese do not have `stem` column.

Note that we only create one shuffled corpus and train 2 iterations below as a toy example.

1. Preprocess data </br>
   ```shell
   python preprocess.py \
       --source_dir ./data/English-uk/raw \
       --dest_dir ./data/English-uk/ \
       --num_epochs 2 \
       --num_shuffles 1 \
       --use_stem        
   ```

2. Train models </br>
   ```shell
   python word2vec.py \
       --source_dir ./data/English-uk/ \
       --dest_dir ./output/English-uk/ \
       --dim 100 \
       --min_count 15 \
       --n_epochs 2 \
       --window 5 \
       --negative 5 \
       --sg 1 \
       --sample 1e-5 \
       --ns_exponent 0.75 \
       --workers 4
   ```

3. Generate outputs on language variations  </br>
   ```shell
   python generate_outputs.py \
       --source_dir ./output/English-uk/ \
       --embedding_filename embeddings-ep2-f15-d100-w5.pickle \
       --word_inventory ./data/English-uk/word_inventory.csv \
       --google_word2vec ./data/google-word2vec/GoogleNews-vectors-negative300.bin \
       --num_neighbors 25
   ```

### Analysis

The analysis code is in R. Please refer to `analysis` folder in the repository. 


## Citation

We now have the paper you can cite:
```bibtex
@misc{jiang_frank_kulkarni_fourtassi_2020,
 title={Exploring patterns of stability and change in caregivers’ word usage across early childhood},
 url={psyarxiv.com/fym86},
 DOI={10.31234/osf.io/fym86},
 publisher={PsyArXiv},
 author={Jiang, Hang and Frank, Michael C and Kulkarni, Vivek and Fourtassi, Abdellah},
 year={2020},
 month={Feb}
}
```
