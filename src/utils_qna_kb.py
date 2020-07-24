import os
import json
import nltk
import os
import pprint
# import simpleneighbors
from annoy import AnnoyIndex
import urllib
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import joblib
import pickle


nltk.download('punkt')

class UtilsQnAFAQ:

    def __init__(self, home_path):
        self.questions = []
        # Create the global index object
        self.dims = 512
        self.metric = 'angular'        
        self.index = AnnoyIndex(self.dims, metric=self.metric)
        self.data = pd.DataFrame()
        self.model = None        
        self.corpus = []
        self.id_map = {}
        self.i = 0
        self.built = False
        self.MODEL_ARTIFACTS = os.path.join(home_path, "faq")
        self.USE_MODEL = os.path.join(home_path, "USE", "model")
        self.FAQ_INDEX_PATH = os.path.join(home_path, "faq", "ann_index","faq.ann")
        self.FAQ_TRAIN_BATCH_SIZE = 10
        self.sentence_dict = {} 
        self.questions = []       

    def add_one(self, item, vector):
        """Adds an item to the index.

        You need to provide the item to add and a vector that corresponds to
        that item. (For example, if the item is the name of a color, the vector
        might be a (R, G, B) triplet corresponding to that color. If the item
        is a word, the vector might be a word2vec or GloVe vector corresponding
        to that word.

        Items can be any `hashable
        <https://docs.python.org/3.7/glossary.html#term-hashable>`_ Python
        object. Vectors must be sequences of numbers. (Lists, tuples, and Numpy
        arrays should all be fine, for example.)

        Note: If the index has already been built, you won't be able to add new
        items.

        :param item: the item to add
        :param vector: the vector corresponding to that item
        :returns: None
        """

        assert self.built is False, "Index already built; can't add new items."
        self.index.add_item(self.i, vector)
        self.id_map[item] = self.i
        self.corpus.append(item)
        self.i += 1

    def __index_len__(self):
        return len(self.corpus)


    def load_data(self, path='../data/faq.csv', sep="##,##"):
        print('Loading data from',path)
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
        questions = []
        for item, row in self.data.iterrows():
            questions.append((row['question'], row['answer']))
        return list(set(questions))


    def get_nearest(self, query_text, num_results=10):
        query_embedding = self.model.signatures['question_encoder'](
            tf.constant([query_text]))['outputs'][0]            
    
        if(self.__index_len__() == 0):            
            self.loadindex(self.FAQ_INDEX_PATH)
                
        print('Searching in ',self.__index_len__(),'sentences in Annoy Index')

        search_results = []
        nns = self.index.get_nns_by_vector(query_embedding, num_results, include_distances=True)        
        for j in range(len(nns[0])):
            search_results.append([self.corpus[nns[0][j]],nns[1][j]])       
        return search_results


    def load_USE_model(self, path='/home/sandipan/projects/model/'):
        print('Loading Universal Sentence Encoder from', path)
        self.model = hub.load(path)
        print('Universal Sentence Encoder model from in mem', self.model)


    def build_search_index(self, index_path, batch_size=10):
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
                self.add_one(batch, encodings['outputs'][batch_index])

        print('Building Index...')
        self.index.build(30) # 30 Trees
        self.built = True
        self.sentence_dict = dict(sentences)
        print('Saving Index...')
        self.saveindex(index_path)        
        

    def saveindex(self, prefix):
        with open(prefix + "-data.pkl", "wb") as fh:
            pickle.dump({
                'id_map': self.id_map,
                'corpus': self.corpus,
                'i': self.i,
                'built': self.built,
                'metric': self.metric,
                'dims': self.dims,
                'sentence_dict':self.sentence_dict
            }, fh)
        self.index.save(prefix + ".idx")        

    def loadindex(self, prefix):
        print('Start to load index from',prefix)
        with open(prefix + "-data.pkl", "rb") as fh:
            data = pickle.load(fh)     

        self.dims=data['dims'],
        self.metric=data['metric'],                    
        self.id_map = data['id_map']
        self.corpus = data['corpus']
        self.i = data['i']
        self.built = data['built']
        self.sentence_dict = data['sentence_dict']
        self.index.load(prefix + ".idx")
        print('Loaded',self.__index_len__(),'sentences in Annoy Index')        

    def train_faq_kb(self, data_csv_path, USE_path='', faq_train_batch_size=''):
        print('FAQ KB Training started...')
        if(USE_path == ''):
            USE_path = self.USE_MODEL
        if(faq_train_batch_size == ''):
            faq_train_batch_size = self.FAQ_TRAIN_BATCH_SIZE
        self.load_data(data_csv_path)
        self.load_USE_model(USE_path)
        # extract_questions(data)
        self.build_search_index(index_path=self.FAQ_INDEX_PATH, batch_size=faq_train_batch_size)
        print('FAQ KB Training completed ...')
        return len(self.sentence_dict)

    def askfaq(self, question, num_results=5):
        if (self.model == None):
            self.load_USE_model(self.USE_MODEL)

        results = self.get_nearest(question, num_results=num_results)
        print('results = ',results)
        ans = pd.DataFrame(columns=['original_q','answer','matched_line','confidence'])             
        for result in results:
            print('each res', result,result[0],result[1])
            matched_line = result[0]
            confi = result[1]
            print(result, ans[ans['answer'] == self.sentence_dict[matched_line]].size)
            if(ans[ans['answer'] == self.sentence_dict[matched_line]].size == 0):
                values = ['',self.sentence_dict[matched_line],matched_line,confi]
                zipped = zip(ans.columns, values)    
                a_dictionary = dict(zipped)
                ans = ans.append(a_dictionary,ignore_index=True)
            else:
                print('answer already added, found a new higher confidence of', confi, '>',
                      ans[ans['answer'] == self.sentence_dict[matched_line]]['confidence'].iloc[0])
                if (confi > ans[ans['answer'] == self.sentence_dict[matched_line]]['confidence'].iloc[0]):
                    ans[ans['answer'] == self.sentence_dict[matched_line]]['confidence'].iloc[0] = confi
        return ans.to_dict(orient ='records')


CONVAI_HOME = os.environ.get("CONVAI_HOME")
_inst = UtilsQnAFAQ(CONVAI_HOME)
train = _inst.train_faq_kb
ask = _inst.askfaq

if __name__ == "__main__":
    print('load util')
