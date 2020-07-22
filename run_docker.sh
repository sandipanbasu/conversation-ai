#!/bin/bash

# create build folder
echo "delete build dir"
rm -rf build
mkdir build
cp docker-compose.yml build/
echo "copy flask app"
cp -r flask_app build/
echo "copy nginx app"
cp -r nginx build/
echo "copy a sample data"
cp -r data build/flask_app/
echo "copy embedding model"
cp -r ../model build/flask_app/
echo "copy model artifacts"
cp -r ../model-artifacts build/flask_app/
echo "copy the base ANN index"
cp -r ../index build/flask_app/
echo "copy libs"
cp -r src build/flask_app/
echo "copy all reqs"
cp src/requirements.txt build/flask_app/requirements.txt


cd build

# export FAQ_TRAINING_DATA=data/faq.csv
# export USE_MODEL=/home/sandipan/projects/model/
# export FAQ_INDEX_PATH=/home/sandipan/projects/index/faq.ann
# export FAQ_TRAIN_BATCH_SIZE=10
# export MODEL_ARTIFACTS=/home/sandipan/projects/model-artifacts
# export FAQ_TRAIN_FOLDER=/home/sandipan/projects/uploads
# export PYTHONPATH=build/flask_app/src

echo "killing old docker processes"
docker-compose rm -fs

echo "building docker containers"
docker-compose up --build -d