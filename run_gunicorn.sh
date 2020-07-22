export FAQ_TRAINING_DATA=/home/sandipan/projects/data/faq.csv
export USE_MODEL=/home/sandipan/projects/model/
export FAQ_INDEX_PATH=/home/sandipan/projects/index/faq.ann
export FAQ_TRAIN_BATCH_SIZE="10"
export MODEL_ARTIFACTS=/home/sandipan/projects/model-artifacts
export FAQ_TRAIN_FOLDER=/home/sandipan/projects/uploads
export PYTHONPATH=$PYTHONPATH:/home/sandipan/projects/conversation-ai/src

cd flask_app

nohup gunicorn -w 1 -b 0.0.0.0:8000 --log-file=/home/sandipan/projects/logs/gunicorn.log wsgi:server &
