# python generate_outputs.py \
#        --source_dir /deep/group/sharonz/change/output/Japanese/ \
#        --embedding_filename embeddings-ep50-f15-d100-w5.pickle \

# python generate_outputs.py \
#        --source_dir /deep/group/sharonz/change/output/German/ \
#        --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
#        --word_inventory ./translate/en.csv \

python generate_outputs.py \
       --source_dir ./output/English-uk/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/en.csv \
       --google_word2vec ./google-word2vec/English/model.bin \
       --num_neighbors 25

python generate_outputs.py \
       --source_dir ./output/English-na/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/en.csv \
       --google_word2vec ./google-word2vec/English/model.bin \
       --num_neighbors 25

python generate_outputs.py \
       --source_dir ./output/Chinese/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/zh-cn.csv \
       --google_word2vec ./google-word2vec/Chinese/model.bin \
       --num_neighbors 25

python generate_outputs.py \
       --source_dir ./output/French/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/fr.csv \
       --google_word2vec ./google-word2vec/French/model.bin \
       --num_neighbors 25

python generate_outputs.py \
       --source_dir ./output/Spanish/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/es.csv \
       --google_word2vec ./google-word2vec/Spanish/model.bin \
       --num_neighbors 25

python generate_outputs.py \
       --source_dir ./output/German/ \
       --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
       --word_inventory ./translate/de.csv \
       --google_word2vec ./google-word2vec/German/model.bin \
       --num_neighbors 25
 

# python generate_outputs.py \
#        --source_dir ./output/English-na/ \
#        --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
#        --word_inventory ./data/English-na/word_inventory.csv \
#        --google_word2vec ./google-word2vec/English-na/GoogleNews-vectors-negative300.bin \
#        --num_neighbors 25

# python generate_outputs.py \
#        --source_dir output/Spanish/ \
#        --word_inventory translate/spanish_inventory.csv \
#        --google_word2vec google-word2vec/Spanish/model.bin


# python generate_outputs.py \
#        --source_dir ./output/English-uk/ \
#        --embedding_filename embeddings-ep50-f15-d100-w5.pickle \
#        --word_inventory ./data/English-uk/word_inventory.csv \
#        --google_word2vec ./google-word2vec/English/GoogleNews-vectors-negative300.bin \
#        --num_neighbors 25


# scp hjian42@sc.stanford.edu:/sailhome/hjian42/change/output/English-uk.zip output/









