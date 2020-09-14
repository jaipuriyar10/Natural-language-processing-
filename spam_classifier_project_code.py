# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:42:37 2020

@author: Divyansh
"""

import pandas as pd

mess = pd.read_csv("SMSSpamCollection", sep="\t",names=["label","message"])

#data cleaning and preprocessing

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0,len(mess)):
    rev = re.sub('^[a-zA-Z]',' ',mess['message'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in stopwords.words('english')]
    rev = ' '.join(rev)
    corpus.append(rev)
    
#creating bag of words 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=(2500))

x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(mess['label'])
y = y.iloc[:,1].values



# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

