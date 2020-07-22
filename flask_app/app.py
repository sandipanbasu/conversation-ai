from flask import Flask
from flask import request, jsonify, Response
from random import sample
import sys
import os
from utils_qna_kb import train, ask

server = Flask(__name__)
FAQ_TRAIN_FOLDER = os.environ.get("FAQ_TRAIN_FOLDER")
print(FAQ_TRAIN_FOLDER)

@server.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return 'The epitome of non-sense '
    else:
        return 'The epitome of non-sense'

@server.route('/train_faq', methods=['GET', 'POST'])
def train_faq():
    response = {}
    if request.method == 'POST':        
        # return 'Send a POST request'
        f = request.files['file']
        f.save(os.path.join(FAQ_TRAIN_FOLDER, f.filename))
        print(f.filename,"is written to disk.")
        sentence_len = train(data_csv_path = os.path.join(FAQ_TRAIN_FOLDER, f.filename))     
        response['status'] = 'sucess'
        response['sent_count'] = sentence_len
        return jsonify(response)
    else:
        return jsonify('Send a POST request')

@server.route('/ask', methods=['GET', 'POST'])
def infer_faq():
    if request.method == 'GET':
        query = request.args.get('q')
        # size = request.args.get('s'))
        # if(size == 0):
        #     size = 1        
        ans = ask(query,3)
        return Response(ans, mimetype='application/json')         