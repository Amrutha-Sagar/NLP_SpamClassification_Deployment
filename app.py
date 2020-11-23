# -*- coding: latin-1 -*-
"""
Created on Wed Jul  8 20:34:17 2020

@author: DELL
"""
from flask import Flask, render_template, url_for, request 
import nltk
import pandas as pd
import numpy as np
import pickle

#Load file from disk
filename='nlp_model.pkl'
mnb=pickle.load(open(filename,'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction= mnb.predict(vect)
        
    return render_template('result.html', prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)


