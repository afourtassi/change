python utils.py \
--source_dir ./data/English-na/raw/ \
--dest_dir ./output/English-na/ \
--word_inventory ./translate/en.csv \
--use_stem \
--language English

python utils.py \
--source_dir ./data/English-uk/raw/ \
--dest_dir ./output/English-uk/ \
--word_inventory ./translate/en.csv \
--use_stem \
--language English

python utils.py \
--source_dir ./data/Spanish/raw/ \
--dest_dir ./output/Spanish/ \
--word_inventory ./translate/es.csv \
--use_stem \
--language Spanish

python utils.py \
--source_dir ./data/French/raw/ \
--dest_dir ./output/French/ \
--word_inventory ./translate/fr.csv \
--use_stem \
--language French

python utils.py \
--source_dir ./data/Chinese/raw/ \
--dest_dir ./output/Chinese/ \
--word_inventory ./translate/zh-cn.csv \
--use_stem \
--language Chinese