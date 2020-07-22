#!/bin/bash

# create build folder
echo "delete build dir"
rm -rf build
mkdir build
cp docker-compose.yml build/
cp -r flask_app build/
cp -r nginx build/
cp -r data build/
cp -r data build/
cp -r src build/flask_app/src

cd build

export FAQ_TRAINING_DATA=data/faq.csv
export USE_MODEL=/home/sandipan/projects/model/
export FAQ_INDEX_PATH=/home/sandipan/projects/index/faq.ann
export FAQ_TRAIN_BATCH_SIZE=10
export MODEL_ARTIFACTS=/home/sandipan/projects/model-artifacts
export FAQ_TRAIN_FOLDER=/home/sandipan/projects/uploads

export PYTHONPATH=build/flask_app/src

echo "killing old docker processes"
docker-compose rm -fs

echo "building docker containers"
docker-compose up --build -d