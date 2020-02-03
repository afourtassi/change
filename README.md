# Exploring patterns of stability and change in caregivers' word usage across early childhood

## Training Word2Vec Models

There are four ipython notebooks for training different word2vec models on different data: English, North American (NA) English, Shuffling (once) on English, five times shuffling on English. 

For instance, run `1-train_word_embeddings-English.ipynb` to 
- preprocessing childes data
	- replace proper names with `$name$` and interjections with `$co$`
- split data into 6 periods, 2M tokens per epoch
- train word embeddings on each epoch
The output for this ipython notebook will be saved in `embeddings-over-time/embeddings-English-NA-2M-ep25-f15.pickle`

## Saving Semantic Changes to csv files

There are four ipython notebooks for visualizing semantic changes and save these changes in csv files. Later, we use R to do analysis on these changes. 
For instance, run `2-plot_change_by_category-English.ipynb`. 

The output of this file will be saved in `data/cdi_output_en`

## Analysis

The analysis code is in R. Please refer to the R code in the repository. 
