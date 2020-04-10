# Exploring patterns of stability and change in caregivers' word usage across early childhood

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements dynamic word embeddings for exploring the language variations of caregivers across multiple languages in CHILDES dataset. 

### Setup Environment

#### Install via pip:
1. install requirements </br>
   ```conda env create -f environment.yml```

### Train dynamic word embeddings

We take the `german` language for example, whose raw csv files sit in `./data/german/raw/`.

1. Preprocess data </br>

   ```shell
   python preprocess.py \
       --source_dir ./data/german/raw \
       --dest_dir ./data/german/ \
       --num_epochs 2 \
	   --num_shuffles 5       
   ```

2. Train models </br>
   ```shell
   python word2vec.py
       --source_dir ./data/german/ \
       --dest_dir ./output/german/ \
       --dim 100 \
	    --min_count 15 \
       --n_epochs 100 \
       --window 5 \
       --negative 5 \
       --sg 1 \
       --sample 1e-5 \
       --ns_exponent 0.75 \
       --workers 4
   ```

3. Generate outputs on language variations  </br>
   ```> python generate_outputs.py```

### Analysis

The analysis code is in R. Please refer to `analysis` folder in the repository. 
