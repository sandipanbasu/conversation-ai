#!/bin/bash

# create build folder
echo "delete build dir"
rm -rf build
mkdir build
cp docker-compose.yml build/
cp -r flask_app build/
cp -r nginx build/
cp -r src build/flask_app/src

cd build


echo "killing old docker processes"
docker-compose rm -fs

echo "building docker containers"
docker-compose up --build -d