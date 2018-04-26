import requests
import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model

import flask
from flask import Flask, render_template, jsonify
from random import *

app = Flask(__name__,
            static_folder = "../dist/static",
            template_folder = "../dist")
model = None
word2vec_model = None

@app.route('/api/random')
def random_number():
    response = {
        'randomNumber' : randint(1, 200)
    }
    return jsonify(response)

@app.route('/api/chat', methods=['POST'])
def reply():
    response = {"success": False}
    if flask.request.method == "POST":
        message = flask.request.form['message']
        message_vec = prepare_data(message)
        predictions = model.predict(message_vec)
        outputlist=[word2vec_model.most_similar([predictions[0][i]])[0][0] for i in range(15)]
        n_list = []
        for i in outputlist:
            if i is not 'karluah':
                n_list.append(i)
            # if i is 'karluah':
            #     outputlist.pop(i)
        output =' '.join(n_list)
        output = output.strip('karluah')

        response = {
            'message' : output
        }

        response['success'] = True
    return jsonify(response)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get("http://localhost:8080/{}".format(path)).text
    return render_template("index.html")


def load_models():
    global model
    global word2vec_model
    model = load_model('LSTM50000.h5')
    word2vec_model = gensim.models.Word2Vec.load('apnews_sg/word2vec.bin')


def prepare_data(message):
    sent_end=np.ones((300,),dtype=np.float32)

    sent=nltk.word_tokenize(message.lower())
    sent_vec = [word2vec_model[w] for w in sent if w in word2vec_model.wv.vocab]

    sent_vec[14:] = []
    sent_vec.append(sent_end)
    if len(sent_vec)<15:
        for i in range(15-len(sent_vec)):
            sent_vec.append(sent_end)
    sent_vec = np.array([sent_vec])

    return sent_vec

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_models()
    app.run()
