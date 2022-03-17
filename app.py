#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:59:54 2022

@author: lenovo
"""

## import libraries
import pickle
from keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
#from wordcloud import STOPWORDS
#nltk.download('punkt')
#nltk.download('wordnet')
#from sklearn.preprocessing import LabelEncoder 
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')
import os
from flask import Flask, request, jsonify, json
from flask import render_template

## load model

filename = "lstm_model.h5"
lstm_model = load_model(filename)
lstm_model.load_weights('lstm_Weights.h5')
with open ("text_tokenizer.pkl",'rb') as file:
    tokenizer = pickle.load(file)
MAX_SEQUENCE_LENGTH = 500

def preprocess_description(text):  
        np.random.seed(1)
        text = ''.join(re.findall(r'[a-zA-Z0-9 ]',text))        
        text = text.lower()
        text = text.strip()        
        words = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if len(word)>2]
        text = ' '.join(words)  
        #desc_vectors = vectorizer.transform([text])
        #text = str(np.array(text))
        #print("NP:    ",text)
        sequences = tokenizer.texts_to_sequences([text])
        print("seq:   ",sequences)
        text = pad_sequences(sequences, maxlen=500)        
        return text    

def get_response(pred):
    if pred ==0:
        return str('Level 1 low risk')
    elif pred == 1:
        return str('Level 2 mild risk')
    elif pred == 2:
        return str('Level3 medium risk')
    elif pred == 3:
        return str('Level 4 there is a risk')
    else:
        return str('Level 5 high risk')

def predict_desc(desc):   
    prepared_desc = preprocess_description(desc)
    print("prepared description : ",prepared_desc)
    prediction = lstm_model.predict([[prepared_desc]])
    print(prediction)
    pred=np.argmax(prediction,axis=1)    
    response = get_response(pred)   #{"prediction":str(pred)}
    return response
   # return pred    

app = Flask(__name__)
#app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/index',methods=['POST'])
def getPrediction():
    # Get the data from the POST request.
    if request.method=="POST":
          data = request.form["description"]
          output = predict_desc(data)
          response =str(output)
          return render_template("index.html",descrip = response)



if __name__ == "__main__":
    app.run(debug=True)
