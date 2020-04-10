# Exploring patterns of stability and change in caregivers' word usage across early childhood

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements dynamic word embeddings for exploring the language variations of caregivers across multiple languages in CHILDES dataset. 

### Setup Environment

#### Install via pip:
1. install requirements </br>
   ```> conda env create -f environment.yml```

### Train dynamic word embeddings
1. Preprocess data </br>
   ```> python preprocess.py``` </br>

2. Train models </br>
   ```> python word2vec.py```

3. Generate outputs on language variations  </br>
   ```> python generate_outputs.py```

### Analysis

The analysis code is in R. Please refer to `analysis` folder in the repository. 
