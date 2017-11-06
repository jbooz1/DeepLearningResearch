import numpy as np
import pandas
import timeit

from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

def main():

    good_path = './Data/goodPermissionsFinal.txt'
    mal_path = './Data/malwarePermissionsFinal.txt'

    with open(good_path) as f:
        gdprm = f.readlines()
    with open(mal_path) as f:
        mlprm = f.readlines()

    perms = gdprm + mlprm

    labels = np.array([])
    for x in gdprm:
        labels = np.append(labels, 0)
    for x in mlprm:
        labels = np.append(labels, 1)

    count_vect = CountVectorizer(input=u'content', analyzer=u'word', token_pattern='(\\b(:?uses-|optional-)?permission:\s[^\s]*)')
    time0 = timeit.default_timer()
    features = count_vect.fit_transform(perms)
    features = features.todense()
    features = np.array(features)
    inputSize = len(features)

    print("Done Vectorizing Data")
    #print 'ouput: train_ratio, epochs, batch_size, avg_acc, avg_true_pos, avg_true_neg, avg_fpos_rate, avg_fneg_rate, avg_train_time, avg_test_time'

    print ("Grid Search for One Layer")
    grid_search_EpochBatch("oneLayer", features, labels)
    '''
    print "Grid Search for Binary Decreasing Layers"
    grid_search_EpochBatch("binaryDecrease", features, labels)
    '''
    print("Grid Search for Four equal Layers")
    grid_search_EpochBatch("fourSame", features, labels)
    print("Grid Search for Four Decr Layers")
    grid_search_EpochBatch("fourDecr", features, labels)

    return 0


def grid_search_EpochBatch(modelName, features, labels):
    epochs = [1]
    batch_size = [10]
    test_ratio= .8
    neurons = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000]
    optimizer = ['Nadam']
    #weight_constraint = [1, 2, 3, 4, 5]
    #dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #lr = [.0005, .001, .002, .003, .004, .005, .006, .007, .008, .009]

    paramGrid = dict(epochs=epochs, batch_size=batch_size, optimizer=optimizer, neurons=neurons)

    if modelName == "oneLayer" :
        model = KerasClassifier(build_fn=create_one_layer, verbose=0)
    elif modelName == "binaryDecrease":
        model = KerasClassifier(build_fn=create_binaryDecrease, verbose=0)
    elif modelName == "fourSame":
        model = KerasClassifier(build_fn=create_fourSameLayer, verbose=0)
    elif modelName == "fourDecr":
        model = KerasClassifier(build_fn=create_fourDecrLayer, verbose=0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=paramGrid, n_jobs=1, cv=sss, refit=True, verbose=2)
    grid_fit = grid.fit(features, labels)

    means = grid_fit.cv_results_['mean_test_score']
    stds = grid_fit.cv_results_['std_test_score']
    params = grid_fit.cv_results_['params']

    print("%s Best: %f using %s" % (modelName, grid_fit.best_score_, grid_fit.best_params_))

    df = pandas.DataFrame(grid_fit.cv_results_)
    try:
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/epochBatch' + modelName + '.csv'
        file1 = open(path1, "w+")
    except IOError:
        path1 = "gridSearch" + modelName + ".csv"
        file1=open(path1, "w+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0


def create_one_layer(neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(weight_constraint) ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_fourSameLayer(neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_fourDecrLayer(neurons=25, optimizer='adam', dropout_rate=0.0, weight_constraint=0):
    #baseline Model
    model = Sequential()
    n2 = neurons//2 if neurons//2 > 0 else 1
    n3 = neurons // 3 if neurons // 3 > 0 else 1
    n4 = neurons // 4 if neurons // 4 > 0 else 1
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n2, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n2 > 1 : model.add(Dropout(dropout_rate))
    model.add(Dense(n3, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n3 > 1: model.add(Dropout(dropout_rate))
    model.add(Dense(n4, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    if n4 > 1: model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_binaryDecrease(neurons=25, optimizer='adam'):
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu'))
    while (neurons/2 >=1):
        model.add(Dense(neurons/2, kernel_initializer='normal', activation='relu'))
        neurons/=2
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()

