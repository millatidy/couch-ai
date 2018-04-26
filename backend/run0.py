import os
import flask
import requests
from chatbot import Chatbot


from flask import Flask, render_template, jsonify
from flask_cors import CORS
from random import *

app = Flask(__name__,
            static_folder = "../dist/static",
            template_folder = "../dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/random')
def random_number():
    response = {
        'randomNumber' : randint(1, 200)
    }
    return jsonify(response)

@app.route('/api/chat', methods=['POST'])
def reply():
    response = {
        "success": False,
        "message": None
        }
    if flask.request.method == "POST":
        print('here')
        data = flask.request.get_json(silent=True)
        input_text = data.get('message')
        print(input_text)
        decoded_sentence = bot.chat(input_text.lower())
        response['message'] = decoded_sentence
        response['success'] = True
    return jsonify(response)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get("http://localhost:8080/{}".format(path)).text
    return render_template("index.html")


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    global bot
    bot = Chatbot()
    bot.initialize()
    app.run(debug=False)
