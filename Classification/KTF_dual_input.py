import numpy as np
import timeit
import pandas
import datetime

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Merge, Dense, Dropout
from keras.models import Sequential
from keras.constraints import maxnorm

def main():

def vectorize(good_path, mal_path, adverse):
    good_path = good_path
    mal_path = mal_path

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

    perm_vect = Countvectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = Countvectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))


    time0 = timeit.default_timer()
    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

def perm_model(neurons=25, optimizer='adam'):
    model = Sequential()

    model.add(Dense(neruons, input_dim=17121, kernel_initializer='normal' \
    , activation='relu'))
    model.add(dropout(0.1))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))




if __name__=="__main__":
    main()
