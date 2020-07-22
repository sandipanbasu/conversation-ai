export FAQ_TRAINING_DATA=data/faq.csv
export USE_MODEL=/home/sandipan/projects/model/
export FAQ_INDEX_PATH=/home/sandipan/projects/index/faq.ann
export FAQ_TRAIN_BATCH_SIZE=10
export MODEL_ARTIFACTS=/home/sandipan/projects/model-artifacts



export TEST_DATA=input/test_cat.csv
# export MODEL=$1

#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
python -m src.train_qna_kb