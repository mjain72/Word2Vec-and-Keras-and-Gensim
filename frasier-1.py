#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:45:54 2018

@author: mohit jain

We will use Word2Vec and RNN (LSTM) to predict who said a particular dialogue  in Fraiser. The dataset used 
is from the following website: https://www.kaggle.com/sulabhbista/frasier-dialogs

"""

from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import logging


logging.basicConfig(format='%(asctime)s : %(levelname) s : %(message)s', level=logging.INFO)

#Set random seed
np.random.seed(24)

'''
Get Dataset

'''
#read CSV file containing dialog and cast, using Pandas , to get a dataframe
frasier = pd.read_csv('script.csv')
#only select Frasier, Niles, Martin, Daphne and Roz  from the database
selected_frasier = frasier.loc[(frasier['cast'] == 'Frasier') | (frasier['cast'] == 'Roz') | (frasier['cast'] == 'Niles')
                      | (frasier['cast'] == 'Martin') | (frasier['cast'] == 'Daphne')]


#Divide the dataframe into Cast which will be labels and Dialog that will be features
frasier_cast = selected_frasier['cast']

#Number of dialog assigned to each character
cast_count = frasier_cast.value_counts()
#Plot and print occurence of each cast
cast_count.plot(kind="bar")
print(frasier_cast.value_counts())

#binarize the labels
labels = label_binarize(frasier_cast, classes=['Frasier', 'Roz', 'Niles', 'Martin', 'Daphne'] )

#Extract the dialoges
frasier_dialog = selected_frasier['dialog']

#Lower and split the dialog
#for regular expressions to keep only letters we will use nltk Regular expression package
tkr = RegexpTokenizer('[a-zA-Z]+')

frasier_dia_split = []

for i, line in enumerate(frasier_dialog):
    #print(line)
    dialog = str(line).lower().split()
    dialog = tkr.tokenize(str(dialog))
    frasier_dia_split.append(dialog)
    
#Build a word2vec model to group words in how they are close together, using Gensim


vector_size = 100
window_size = 5

w2v = Word2Vec(sentences=frasier_dia_split, size=vector_size, window = window_size, min_count=5, negative=10, iter=10)


w2vModel = w2v.wv

w2v.save('Word2vecModel')
w2v.wv.save_word2vec_format('Word2vecModel.txt', binary=False)
w2v.wv.save_word2vec_format('Word2vecModel.bin', binary=True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(frasier_dia_split)
X = tokenizer.texts_to_sequences(frasier_dia_split)

maxLenDia = 10 #maximum dialog length
X = pad_sequences(X, maxlen=maxLenDia)
print(X[1:3, 1:50])
print(X.shape)


#Gensim word2Vec model to create an embedding layer
embedding_layer = Embedding(input_dim=w2vModel.syn0.shape[0], output_dim=w2vModel.syn0.shape[1], weights=[w2vModel.syn0], 
                            input_length=X.shape[1])


#Model creation using Keras
lstm_out = 150

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(units=lstm_out))
model.add(Dropout(0.8))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Split the data

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size= 0.1, random_state = 24)

batch_size = 64
#Train model
model.fit(X_train, y_train, epochs=15, verbose=1, batch_size=batch_size)


#Calculate accuracy
score, acc = model.evaluate(X_test, y_test, verbose = 2, batch_size=batch_size)
y_pred = model.predict(X_test)

#Plot ROC Curve for each cast

falsePositiveRate = dict()
truePositiveRate = dict()
rocAucScore = dict()
castDict = {0:'Fraiser', 1:'Roz', 2:'Niles', 3:'Martin', 4:'Daphne'}

for i in range(5):
   falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
   rocAucScore[i] = auc(falsePositiveRate[i], truePositiveRate[i])
    
    
#ROC curve for each cast member
    
for i in range(5):
    plt.figure(i)
    plt.plot(falsePositiveRate[i], truePositiveRate[i], color='green',
             lw=1, linestyle='-.', label='ROC curve (area = %0.2f) for %s' % (rocAucScore[i], castDict[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for the cast')
    plt.legend(loc="top right")
    plt.show()


#confusion matrix

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))





