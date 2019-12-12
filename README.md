# childes_language
Exploring language changes with CHILDES

## Step 1

Run `1-train_word_embeddings.ipynb` to 
- preprocessing childes data
	- replace proper names with `$name$` and interjections with `$co$`
- split data into 12 periods, 1M tokens per period
- train word embeddings on each period

## Step 2

Run `2-plot_change_per_category.ipynb` to
- save local and global semantic changes to `childes_data/embedding_shift.csv`
- Finding 1:
	- semantic category
		- global metric
		- local metric
	- lexical category
		- global metric
		- local metric
- Finding 2:
	- Google-W2V-SIM: similarity w/ Goolge's Word2Vec embeddings
		- semantic category
		- lexical category
