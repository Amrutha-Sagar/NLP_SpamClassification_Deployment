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
bnb=pickle.load(open(filename,'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
#    data=pd.read_csv('spam.csv', sep=',', encoding='latin-1')
#    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
#    
#    #Data cleaning and preprocessing
#    import re
#    import nltk
#    from nltk.corpus import stopwords
#    from nltk import PorterStemmer, WordNetLemmatizer
#    
#    
#    stem=PorterStemmer()
#    corpus=[]
#    
#    for i in range(len(data)):
#        words=re.sub('[^a-zA-Z]', ' ', data['Message'][i])
#        words=words.lower()
#        words=words.split()
#        words=[stem.stem(word) for word in words if word not in set(stopwords.words('english'))]
#        words=' '.join(words)
#        corpus.append(words)
#    
#    y=pd.get_dummies(data['Type'])
#    y=y.iloc[:,1].values
#    
#    #creating BagOfWords
#    from sklearn.feature_extraction.text import CountVectorizer
#    cv=CountVectorizer()
#    x=cv.fit_transform(corpus).toarray()
#    
#    #Model
#    from sklearn.model_selection import train_test_split
#    X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=.2, random_state=1)
#    
#    
#    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#    mnb=MultinomialNB().fit(X_train,y_train)
#    bnb=BernoulliNB().fit(X_train,y_train)
#    
#    mnb_prediction=mnb.predict(X_test)
#    bnb_prediction=bnb.predict(X_test)
#    
#    from sklearn.metrics import confusion_matrix, accuracy_score
#    mnb_confution=confusion_matrix(y_test,mnb_prediction)
#    bnb_confution=confusion_matrix(y_test,bnb_prediction)
#    mnb_accuracy=accuracy_score(y_test,mnb_prediction)
#    bnb_accuracy=accuracy_score(y_test,bnb_prediction)
#print(bnb_accuracy)
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction= bnb.predict(vect)
        
    return render_template('result.html', prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)


