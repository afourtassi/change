# python preprocess.py \
#        --source_dir /deep/group/sharonz/change/data/Japanese/raw \
#        --dest_dir /deep/group/sharonz/change/data/Japanese/ \
#        --num_epochs 2 \
#        --num_shuffles 5

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/German/raw \
       --dest_dir /deep/group/sharonz/change/data/German/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem 

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/English-uk/raw \
       --dest_dir /deep/group/sharonz/change/data/English-uk/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem 

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/English-na/raw \
       --dest_dir /deep/group/sharonz/change/data/English-na/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/Chinese/raw \
       --dest_dir /deep/group/sharonz/change/data/Chinese/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem      

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/French/raw \
       --dest_dir /deep/group/sharonz/change/data/French/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem 

python preprocess.py \
       --source_dir /deep/group/sharonz/change/data/Spanish/raw \
       --dest_dir /deep/group/sharonz/change/data/Spanish/ \
       --num_epochs 2 \
       --num_shuffles 5 \
       --use_stem 










