#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:54:53 2022

@author: hemlata
"""
import pickle
from keras.models import load_model
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
#from wordcloud import STOPWORDS
#nltk.download('punkt')
#nltk.download('wordnet')
from sklearn.preprocessing import LabelEncoder 
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')
filename = "lstm_model.h5"
lstm_model = load_model(filename)
lstm_model.load_weights('lstm_Weights.h5')
with open ("text_tokenizer.pkl",'rb') as file:
    tokenizer = pickle.load(file)
MAX_SEQUENCE_LENGTH = 500
def preprocess_description(text):  
        np.random.seed(1)
        #text = ''.join(re.findall(r'[a-zA-Z0-9 ]',text))        
        #text = text.lower()
        #text = text.strip()        
       # words = word_tokenize(text)
       # lemmatizer = WordNetLemmatizer()
        #words = [lemmatizer.lemmatize(word) for word in words if len(word)>2]
        #text = ' '.join(words)  
        #print("text:   ",text)
        #desc_vectors = vectorizer.transform([text])
        #text = str(np.array(text))
        #print("NP:    ",text)
        sequences = tokenizer.texts_to_sequences([text])
        print("seq:   ",sequences)
        text = pad_sequences(sequences, maxlen=500)
        #text = nltk.word_tokenize(text)
        return text    
def predict_desc(lstm_model,desc):   
    prepared_desc = preprocess_description(desc)
    print("prepared description : ",prepared_desc)
    prediction = lstm_model.predict([[prepared_desc]])
    print(prediction)
    pred=np.argmax(prediction,axis=1)
    print(pred)
    return pred    
desc = input('Enter desc')
print(desc)
predicted_level = predict_desc(lstm_model,desc)


