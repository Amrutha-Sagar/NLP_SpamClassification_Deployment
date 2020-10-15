# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:17:26 2020

@author: DELL
"""

import nltk
import pandas as pd
import numpy as np
import pickle

data=pd.read_csv('spam.csv', sep=',', encoding='latin-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)

data['Type']=data['Type'].map({'ham':0,'spam':1})
X=data['Message']
y=data['Type']

import re
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer
    
    
#stem=PorterStemmer()
#corpus=[]
#    
#for i in range(len(data)):
#    words=re.sub('[^a-zA-Z]', ' ', data['Message'][i])
#    words=words.lower()
#    words=words.split()
#    words=[stem.stem(word) for word in words if word not in set(stopwords.words('english'))]
#    words=' '.join(words)
#    corpus.append(words)
    
    #creating BagOfWords
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(X)
    
pickle.dump(cv,open('transform.pkl','wb'))

    #Model
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=.33, random_state=42)
    
    
from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB().fit(X_train,y_train)
bnb_prediction=bnb.predict(X_test)
    
from sklearn.metrics import confusion_matrix, accuracy_score
bnb_confution=confusion_matrix(y_test,bnb_prediction)
bnb_accuracy=accuracy_score(y_test,bnb_prediction)
filename = 'nlp_model.pkl'
pickle.dump(bnb, open(filename,'wb'))
