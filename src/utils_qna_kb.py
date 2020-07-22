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
import pickle


nltk.download('punkt')

class UtilsQnAFAQ:

    def __init__(self):
        self.questions = []
        # Create the global index object
        self.index = simpleneighbors.SimpleNeighbors(512, metric='angular')
        self.data = pd.DataFrame()
        self.model = None
        self.MODEL_ARTIFACTS = os.environ.get("MODEL_ARTIFACTS")
        self.FAQ_TRAINING_DATA = os.environ.get("FAQ_TRAINING_DATA")
        self.USE_MODEL = os.environ.get("USE_MODEL")
        self.FAQ_INDEX_PATH = os.environ.get("FAQ_INDEX_PATH")
        self.FAQ_TRAIN_BATCH_SIZE = int(os.environ.get("FAQ_TRAIN_BATCH_SIZE"))
        self.sentence_dict = {}


    def load_data(self, path='../data/faq.csv', sep="##,##"):
        print('Loading data....')
        self.data = pd.read_csv(path, sep=sep, engine='python')
        print('...... done')
        print('Data @head 2')
        print(self.data.head(2))        
        # return data

    def extract_sentences_from_answer(self):
        all_sentences = []
        for index, row in self.data.iterrows():
            # nltk.tokenize.sent_tokenize is a unsupervised algo
            sentences = nltk.tokenize.sent_tokenize(row['answer'])
            sentences = sentences + nltk.tokenize.sent_tokenize(row['question'])
            all_sentences.extend(zip(sentences, [row['answer']] * len(sentences)))
        return list(set(all_sentences))  # remove duplicates


    def extract_questions(self):
        for item, row in self.data.iterrows():
            questions.append((row['question'], row['answer']))
        return list(set(questions))


    def get_nearest(self, query_text, num_results=10):
        query_embedding = self.model.signatures['question_encoder'](
            tf.constant([query_text]))['outputs'][0]            
    
        if(self.index.__len__() == 0):            
            self.load_index(self.FAQ_INDEX_PATH)
                
        print('Searching in ',self.index.__len__(),'sentences in Annoy Index')        
        search_results = self.index.nearest(query_embedding, n=num_results)
        return search_results


    def load_USE_model(self, path='/home/sandipan/projects/model/'):
        print('Loading Universal Sentence Encoder from', path)
        self.model = hub.load(path)
        print('Universal Sentence Encoder model from in mem', self.model)


    def build_search_index(self, batch_size=10, index_path='/home/sandipan/projects/index/faq.ann'):
        sentences = self.extract_sentences_from_answer()
        encodings = self.model.signatures['response_encoder'](
            input=tf.constant([sentences[0][0]]),
            context=tf.constant([sentences[0][1]]))
        print('Computing embeddings for %s sentences' % len(sentences))
        slices = zip(*(iter(sentences),) * batch_size)
        num_batches = int(len(sentences) / batch_size)
        print('Batch wise index add...')
        for s in tqdm(slices, total=num_batches):
            response_batch = list([r for r, c in s])
            context_batch = list([c for r, c in s])
            encodings = self.model.signatures['response_encoder'](
                                                input=tf.constant(response_batch),
                                                context=tf.constant(context_batch)
                                            )
            for batch_index, batch in enumerate(response_batch):
                self.index.add_one(batch, encodings['outputs'][batch_index])

        print('Building Index...')
        self.index.build()
        print('Saving Index...')
        self.index.save(index_path)
        self.sentence_dict = dict(sentences)


    def load_index(self, index_path='/home/sandipan/projects/index/faq.ann'):
        print('Start to load index from',index_path)
        self.index = simpleneighbors.SimpleNeighbors.load(index_path)
        print('Loaded',self.index.__len__(),'sentences in Annoy Index')

    def train_faq_kb(self, data_csv_path, USE_path, faq_train_batch_size):
        print('FAQ KB Training started...')
        self.load_data(data_csv_path)
        self.load_USE_model(USE_path)
        # extract_questions(data)
        self.build_search_index(batch_size=faq_train_batch_size)
        print('Dumping sentence dict ...')
        joblib.dump(self.sentence_dict, f"{self.MODEL_ARTIFACTS}/sentence_dict.pkl")
        print('FAQ KB Training completed ...')
        # return sentence_dict

    def askfaq(self, question, num_results=5):
        if (self.model == None):
            self.load_USE_model(self.USE_MODEL)

        if(len(self.sentence_dict) == 0):     
            self.sentence_dict = joblib.load(os.path.join(self.MODEL_ARTIFACTS, "sentence_dict.pkl"))

        results = self.get_nearest(question, num_results=num_results)
        print('results = ',results)
        ans = pd.DataFrame(columns=['original_q','answer','matched_line','confidence'])        
        for result in results:
            print(result, ans[ans['answer'] == self.sentence_dict[result]].size)
            if(ans[ans['answer'] == self.sentence_dict[result]].size == 0):
                values = ['',self.sentence_dict[result],result,100]
                zipped = zip(ans.columns, values)    
                a_dictionary = dict(zipped)
                ans = ans.append(a_dictionary,ignore_index=True)  
        return ans

    def save(self):       
        with open(self.MODEL_ARTIFACTS + "util-qna-faq-data.pkl", "wb") as fh:
            pickle.dump(self, fh)        


if __name__ == "__main__":
    print('load util')
