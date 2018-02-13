import numpy as np
import timeit
import pandas
import datetime

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm

def main():

    data_dir = '/home/jmcgiff/Documents/research/DeepLearningResearch/Data/badging_small/'
    ben_data = data_dir + 'benign/badging_final.txt'
    mal_data = data_dir + 'mal/badging_final.txt'

    inputs = vectorize(ben_data, mal_data)
    perm_inputs = inputs[0]
    feat_inputs = inputs[1]

    print perm_inputs.shape
    print feat_inputs.shape

    labels = inputs[2]

    perm_width = len(perm_inputs[0])
    feat_width = len(feat_inputs[0])

    #model = perm_model()
    #model.fit(perm_inputs, labels)


    model = Multi_model(perm_width, feat_width)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([perm_inputs, feat_inputs], labels, epochs=10, batch_size=10)

def vectorize(good_path, mal_path):

    with open(good_path) as f:
        ben_samples = f.readlines()
    with open(mal_path) as f:
        mal_samples = f.readlines()

    samples = ben_samples + mal_samples

    labels = np.array([])
    for x in ben_samples:
        labels = np.append(labels, 0)
    for x in mal_samples:
        labels = np.append(labels, 1)

    perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"

    perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))


    time0 = timeit.default_timer()
    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

    inputs = [perm_inputs, feat_inputs, labels]
    return inputs



def perm_model(neurons=25, optimizer='adam'):
    model = Sequential()

    model.add(Dense(neurons, input_dim=230, kernel_initializer='normal' \
    , activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model

#def feat_model(ner)

def Multi_model(perm_width, feat_width):
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    x = Dense(32, activation='relu')(perm_input)
    x = Dense(32, activation='relu')(perm_input)

    feat_input = Input(shape=(feat_width,), name='features_input')
    x = concatenate([x, feat_input])
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=[perm_input, feat_input], outputs=output)
    print model.summary()
    return model

if __name__=="__main__":
    main()
