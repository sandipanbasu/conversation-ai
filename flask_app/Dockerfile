FROM python:3.7.6

WORKDIR /opt/convai
RUN ls
COPY requirements.txt .
# COPY src src
# COPY index .
# COPY model .
# COPY data .
# COPY model-artifacts .
# COPY app.py .
# COPY wsgi.py .
COPY . .

RUN mkdir /opt/convai/uploads
RUN mkdir /opt/convai/logs

RUN export FAQ_TRAINING_DATA=/opt/convai/data/faq.csv
RUN export USE_MODEL=/opt/convai/model/
RUN export FAQ_INDEX_PATH=/opt/convai/index/faq.ann
RUN export FAQ_TRAIN_BATCH_SIZE="10"
RUN export MODEL_ARTIFACTS=/opt/convai/model-artifacts
RUN export FAQ_TRAIN_FOLDER=/opt/convai/uploads
RUN export PYTHONPATH=/opt/convai/src;$PYTHONPATH

RUN pip install --no-cache-dir -r requirements.txt