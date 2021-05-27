from flask import Flask
from flask import request
import torch


app = Flask(__name__)

from nnet import NeuralNet
model = torch.load('captcha-recognizer.pth')

@app.route('/ping')
def ping():
    print('Hello world!')
    return "Hello"


@app.route('/solve', methods=['POST'])
def solve_captcha():
    file = request.files['captcha']
    print('Jelou')
    return("Image recived.")
