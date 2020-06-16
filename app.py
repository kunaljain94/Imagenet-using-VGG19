# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 02:29:33 2020

@author: kunal.jain
"""

import pandas as pd
import numpy as np
# import streamlit as st

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flasgger import Swagger
from flask import Flask,request



app=Flask(__name__)#Start app, which point you start the app
Swagger(app)

MODEL_PATH = 'vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()#for imagenet



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/')
def index():
    return "Hey Everyone"

@app.route('/predict',methods=['POST','GET'])
def upload():
    if request.method=='POST'
    