import numpy as np
import timeit
import pandas
import datetime

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Nadam

def main():

    data_dir = '/home/jmcgiff/Documents/research/DeepLearningResearch/Data/badging_med/'
    ben_data = data_dir + 'ben_badging_med.txt'
    mal_data = data_dir + 'mal_badging_med.txt'

    inputs = vectorize(ben_data, mal_data)
    perm_inputs = inputs[0]
    feat_inputs = inputs[1]
    comb_inputs = inputs[2]

    print perm_inputs.shape
    print feat_inputs.shape

    labels = inputs[3]

    perm_width = len(perm_inputs[0])
    feat_width = len(feat_inputs[0])
    comb_width = len(comb_inputs[0])

    print perm_inputs.shape
    print feat_inputs.shape
    print labels.shape

    #kfolds cross validation
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    sss = StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=.2)
    feat_split = sss.split(feat_inputs, labels)
    for train_index, test_index in feat_split:
        model = Multi_model(perm_width, feat_width)
        #model2 = create_one_layer(comb_width)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
        perm_train, perm_test = perm_inputs[train_index], perm_inputs[test_index]
        feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
        comb_train, comb_test, = comb_inputs[train_index], comb_inputs[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        model.fit([perm_train, feat_train], labels_train, epochs=16, batch_size=10)
        #model2.fit(comb_train, labels_train, epochs=10, batch_size=16)
        print ''
        print model.evaluate([perm_test, feat_test], labels_test)
        #print model2.evaluate(comb_test, labels_test)
        print ''

    return

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
    comb_pattern = "(?:\w|\.)+(?:hardware|permission).(?:\w|\.)+"

    perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))
    comb_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=comb_pattern))

    time0 = timeit.default_timer()
    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

    comb_inputs_sparse = comb_vect.fit_transform(samples)
    comb_inputs_dense = comb_inputs_sparse.todense()
    comb_inputs = np.array(comb_inputs_dense)

    print perm_vect.get_feature_names()
    inputs = [perm_inputs, feat_inputs, comb_inputs, labels]
    return inputs




def Multi_model(perm_width, feat_width):
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    x = Dense(32, activation='relu')(perm_input)
    x = Dropout(.2)(x)
    x = Dense(32, activation='relu')(perm_input)

    feat_input = Input(shape=(feat_width,), name='features_input')
    #y = Dense(16, activation='relu')(feat_input)
    x = concatenate([x, feat_input])
    x = Dense(16, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=[perm_input, feat_input], outputs=output)
    print model.summary()
    return model

def create_one_layer(perm_width, dropout_rate=.1, neurons=32, optimizer='nadam'):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(25, input_dim=perm_width, kernel_initializer='normal', activation='relu',))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__=="__main__":
    main()
