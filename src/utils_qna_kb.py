import os
import json
import nltk
import os
import pprint
import simpleneighbors
import urllib
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import joblib


nltk.download('punkt')
questions = []
# Create the global index object
index = simpleneighbors.SimpleNeighbors(512, metric='angular')

MODEL_ARTIFACTS = os.environ.get("MODEL_ARTIFACTS")
FAQ_TRAINING_DATA = os.environ.get("FAQ_TRAINING_DATA")
USE_MODEL = os.environ.get("USE_MODEL")
FAQ_INDEX_PATH = os.environ.get("FAQ_INDEX_PATH")
FAQ_TRAIN_BATCH_SIZE = int(os.environ.get("FAQ_TRAIN_BATCH_SIZE"))


def load_data(path='../data/faq.csv', sep="##,##"):
    print('Loading data....')
    data = pd.read_csv(path, sep='##,##', engine='python')
    print('...... done')
    return data

def extract_sentences_from_answer(df):
    all_sentences = []
    for index, row in df.iterrows():
        # nltk.tokenize.sent_tokenize is a unsupervised algo
        sentences = nltk.tokenize.sent_tokenize(row['answer'])
        sentences = sentences + nltk.tokenize.sent_tokenize(row['question'])
        all_sentences.extend(zip(sentences, [row['answer']] * len(sentences)))
    return list(set(all_sentences))  # remove duplicates


def extract_questions(df):
    for index, row in df.iterrows():
        questions.append((row['question'], row['answer']))
    return list(set(questions))


def get_nearest(query_text, model, num_results=10):
    query_embedding = model.signatures['question_encoder'](
        tf.constant([query_text]))['outputs'][0]
    search_results = index.nearest(query_embedding, n=num_results)
    return search_results


def load_USE_model(path='/home/sandipan/projects/model/'):
    print('Loading Universal Sentence Encoder from', path)
    m = hub.load(path)
    print('Universal Sentence Encoder model from in mem', m)
    return m


def build_search_index(data, model, batch_size=10, index_path='/home/sandipan/projects/index/faq.ann'):
    sentences = extract_sentences_from_answer(data)
    encodings = model.signatures['response_encoder'](
        input=tf.constant([sentences[0][0]]),
        context=tf.constant([sentences[0][1]]))
    print('Computing embeddings for %s sentences' % len(sentences))
    slices = zip(*(iter(sentences),) * batch_size)
    num_batches = int(len(sentences) / batch_size)
    print('Batch wise index add...')
    for s in tqdm(slices, total=num_batches):
        response_batch = list([r for r, c in s])
        context_batch = list([c for r, c in s])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(response_batch),
            context=tf.constant(context_batch)
        )
        for batch_index, batch in enumerate(response_batch):
            index.add_one(batch, encodings['outputs'][batch_index])

    print('Building Index...')
    index.build()
    print('Saving Index...')
    index.save(index_path)
    return dict(sentences)


def load_index(index_path='/home/sandipan/projects/index/faq.ann'):
    index.load(index_path)

def train_faq_kb(data_csv_path, USE_path, faq_train_batch_size):
    print('FAQ KB Training started...')
    data = load_data(data_csv_path)
    print('...')
    print('Data @head 2')
    print(data.head(2))
    model = load_USE_model(USE_path)
    # extract_questions(data)
    sentence_dict = build_search_index(
        data=data, model=model, batch_size=faq_train_batch_size)
    print('Dumping sentence dict ...')
    joblib.dump(sentence_dict, f"{MODEL_ARTIFACTS}/sentence_dict.pkl")
    print('FAQ KB Training completed ...')
    return sentence_dict

def askfaq(question, num_results=5):
    try:
        model
    except NameError:
        print("Load USE Model")
        model = load_USE_model(USE_MODEL)
    
    # if(index.__len__() == 0):
    print("Load index")
    load_index(FAQ_INDEX_PATH)
    
    try:
        sentence_dict 
    except NameError:        
        sentence_dict = joblib.load(os.path.join(MODEL_ARTIFACTS, "sentence_dict.pkl"))

    results = get_nearest(question, model, num_results=num_results)
    ans = pd.DataFrame(columns=['original_q','answer','confidence'])        
    for result in results:
        print(result, ans[ans['answer'] == sentence_dict[result]].size)
        if(ans[ans['answer'] == sentence_dict[result]].size == 0):
            values = ['',sentence_dict[result],100]
            zipped = zip(ans.columns, values)    
            a_dictionary = dict(zipped)
            ans = ans.append(a_dictionary,ignore_index=True)  
    return ans




if __name__ == "__main__":
    print('load util')
