# -*- coding: utf-8 -*-
import os
import re
import csv
import spacy
import nltk
import glob
import json
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

nltk.download('punkt')
nltk.download('stopwords')

def filter_query(query):
    query = query.lower()
    query = re.sub(r'[@][^\s]+', '', query)
    query = re.sub(r'pav.{0,3}bhaji', ' pavbhaji ', query)
    query = re.sub(r'[\!-\/\:-\@]+', ' ', query)
    query = re.sub('[^A-Za-z0-9\s]+', ' ', query)
    query = re.sub(r'[\t\n\r\f ]+', ' ', re.sub(r'\.', '. ', query))
    query = ' '.join([w for w in query.split() if w not in stopwords.words('english')])
    
    # print (query)
    # doc = nlp(query)
    # tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # filt_q = ' '.join(tokens)
    filt_q = re.sub(r'\b(n\'t|nt)\b', 'not', query)
    filt_q = re.sub(r'\'ll\b', 'will', filt_q)
    return filt_q

# helper function for transform_position_vector
def create_ngram_with_position(text, n):
    # tokens = tokenize_hing(text)
    tokens = word_tokenize(text)
    position_dict = {} # vocab_ind: position
    for i in range(n):
        for j in range(len(tokens) - i):
            vocab_ind = bow_vectorizer.vocabulary_[' '.join(tokens[j : j + 1 + i])]
            if vocab_ind not in position_dict:
                position_dict.update({vocab_ind: j})
    return position_dict

# function to vectorize text with the position of the element in the vocabulary index
# this will help to capture relative positional information to try and classify the data points
def transform_position_vector(text_series, ngram_size):
    row = []
    col = []
    mat_data = []
    for i, t in enumerate(text_series):
        position_dict = create_ngram_with_position(t, ngram_size)
        for ind, pos in position_dict.items():
            row.append(i)
            col.append(ind)
            mat_data.append(pos)
    return csr_matrix((mat_data, (row, col)), shape=(len(text_series), len(bow_vectorizer.vocabulary_)))


with open('dataset_mod/pavbhaji.json', 'r') as f:
    data = json.load(f)

# indexing json data with filename
indexed_data = {d['display_url'].split('/')[-1]: d['edge_media_to_caption']['edges'][0]['node']['text'] for d in data if d['edge_media_to_caption']['edges']}

file_names0_set = set([f.split('/')[-1] for f in glob.glob('dataset_mod/images/0/*.jpg')])
file_names1_set = set([f.split('/')[-1] for f in glob.glob('dataset_mod/images/1/*.jpg')])

# dataframe with columns (filename, text, label)
data_with_labels = pd.DataFrame([{'name': name, 'text': indexed_data[name], 'label': 1 if name in file_names1_set else 0} for name in file_names0_set | file_names1_set])

# applying the preprocessing function
processed_text = data_with_labels['text'].map(filter_query)
df = pd.DataFrame({'name':data_with_labels['name'], 'text':processed_text, 'label': data_with_labels['label']})

# splitting train and test sets with 20% data in test set
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# BOW vectorizer with ngrams ranging from 1 to 5
bow_vectorizer = CountVectorizer(tokenizer=word_tokenize, ngram_range=(1,7))
bow_vectorizer.fit(df['text'])

# getting embeddings for tokens in train and test sets
X_train_mat = transform_position_vector(X_train, 7)
X_test_mat = transform_position_vector(X_test, 7)

# implementing a dense layer for positional features using keras
input_vect = Input(shape=(len(bow_vectorizer.vocabulary_),), dtype=tf.int64)
dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_vect)
dense2 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense1)
out = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(dense2)
model = Model(inputs=[input_vect], outputs=out)

LEARNING_RATE = 0.001

optimizer = Adam(lr=LEARNING_RATE)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(" - lr: {}".format(K.eval(self.model.optimizer.lr))) 

LR_PATIENCE = 10
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-8, verbose=1, mode="min")
es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
lr_tracker = LearningRateTracker()
checkpoint = ModelCheckpoint(
    'pos_vector_classifier.h5',
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

history = model.fit(X_train_mat.toarray(), 
          y_train,
          validation_data=(X_test_mat.toarray(), y_test),
          epochs=1000,
          batch_size=256,
          callbacks=[es_callback, lr_tracker, reduce_lr, checkpoint])

# achieved accuracy of 75.82% on validation set

model.load_weights('pos_vector_classifier.h5')
print ('\n\n')

# Evaluating on test set
print ('test_acc: ', model.evaluate(X_test_mat.toarray(), y_test)[1])

# getting results on 10 random samples from train set and test set
ind_random = np.random.choice(X_train.index, 10, replace=False)
X_random_train = X_train[ind_random]
y_random_train = y_train[ind_random]

ind_random = np.random.choice(X_test.index, 10, replace=False)
X_random_test = X_test[ind_random]
y_random_test = y_test[ind_random]

X_random_train_mat = transform_position_vector(X_random_train, ngram_size=7)
X_random_test_mat = transform_position_vector(X_random_test, ngram_size=7)

print ('\n\n')
print ('accuracy on randomly picked 10 points from train data:', model.evaluate(X_random_train_mat.toarray(), y_random_train)[1])
print ('\n\n')
print ('accuracy on randomly picked 10 points from test data:', model.evaluate(X_random_test_mat.toarray(), y_random_test)[1])

inference_on_traindata_df = pd.DataFrame({'filename': df['name'][X_random_train.index], 
                                          'text': df['text'][X_random_train.index], 
                                          'predicted_probability': model.predict(X_random_train_mat.toarray()).reshape(-1), 
                                          'predicted_class': np.array(model.predict(X_random_train_mat.toarray()).reshape(-1) > 0.5, dtype=np.int32), 
                                          'true_class': y_random_train})

inference_on_testdata_df = pd.DataFrame({'filename': df['name'][X_random_test.index], 
                                          'text': df['text'][X_random_test.index], 
                                          'predicted_probability': model.predict(X_random_test_mat.toarray()).reshape(-1), 
                                          'predicted_class': np.array(model.predict(X_random_test_mat.toarray()).reshape(-1) > 0.5, dtype=np.int32), 
                                          'true_class': y_random_test})

inference_on_traindata_df.to_csv('inference_on_traindata_df.csv', index=False)
inference_on_testdata_df.to_csv('inference_on_testdata_df.csv', index=False)

