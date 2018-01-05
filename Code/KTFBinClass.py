import numpy as np
import pandas
import timeit
import sys

from .keras_models import create_binaryDecrease, create_fourDecrLayer, create_fourSameLayer, create_one_layer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score



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

    if sys.argv[1] == 'adverse':
        print("Adversarial Learning")
        count1=0
        count2=0
        gdprmsize = np.size(gdprm, 0)
        mlprmszie = np.size(mlprm, 0)
        for i in range(0, gdprmsize//10):
            if labels[i] == 0 :
                count1 += 1
                labels[i] = 1
        print("Good Permissions Changed: %d" % count1)
        for i in range(gdprmsize, gdprmsize+mlprmszie//10):
            if labels[i] == 1 :
                count2 += 1
                labels[i] = 0
        print("Malware Permissions Changed: %d" % count2)
        print("Total Permissions Changed: %d" % count1+count2)


    # if sys.argv[1] == 'adverse':
    #     print("Adversarial Learning")
    #     full = np.concatenate((perms, labels))
    #     np.random.seed(0)
    #     np.random.shuffle(full)
    #     copyLabels = full[:, -1]
    #     size = np.size(copyLabels)
    #     adverse = np.array([])
    #
    #     for x in range(0, size // 10):
    #         if copyLabels[x] == 0:
    #             adverse = np.append(adverse, 1)
    #         else:
    #             adverse = np.append(adverse, 1)
    #
    #     for x in range(size // 10 + 1, size):
    #         adverse = np.append(adverse, copyLabels[x])
    #
    #     for i in [.2]:
    #         full_run("oneLayer", features, adverse, i)
    #         full_run("fourSame", features, adverse, i)
    #         full_run("fourDecr", features, adverse, i)


    # Used for grid searching
    '''
    print ("Grid Search for One Layer")
    grid_search_EpochBatch("oneLayer", features, labels)
    print "Grid Search for Binary Decreasing Layers"
    grid_search_EpochBatch("binaryDecrease", features, labels)
    print("Grid Search for Four equal Layers")
    grid_search_EpochBatch("fourSame", features, labels)
    print("Grid Search for Four Decr Layers")
    grid_search_EpochBatch("fourDecr", features, labels)
    '''
    #for i in [.8, .2]:
    #for i in [.6, .4]:
    for i in [.8]:
        full_run("oneLayer", features, labels, i)
        full_run("fourSame", features, labels, i)
        full_run("fourDecr", features, labels, i)

    return 0


def full_run(modelName, features, labels, test_ratio):
    epochs=16
    batch_size=10
    neurons=45
    optimizer='Nadam'
    weight_constraint=5
    dropout_rate=0.1
    percent = 1 - test_ratio

    model_params=dict(batch_size=batch_size, epochs=epochs, neurons=neurons, optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate)
    fit_params=dict(batch_size=batch_size, epochs=epochs)
    scoring=['accuracy', 'precision', 'recall', 'f1']

    if modelName == "oneLayer":
        model = KerasClassifier(build_fn=create_one_layer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate, verbose=2)
    elif modelName == "binaryDecrease":
        model = KerasClassifier(build_fn=create_binaryDecrease, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate, verbose=2)
    elif modelName == "fourSame":
        model = KerasClassifier(build_fn=create_fourSameLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate, verbose=2)
    elif modelName == "fourDecr":
        model = KerasClassifier(build_fn=create_fourDecrLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate, verbose=2)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=0)
    cv_result = cross_validate(model, features, labels, cv=sss, fit_params=fit_params, return_train_score=True, scoring=scoring, verbose=2)

    df = pandas.DataFrame(cv_result)
    try:
        #path1 = '/home/lab309/pythonScripts/testResults/deep_results/finalCV' + str(percent) + modelName + '.csv'
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/adverse/' + str(percent) + modelName + '.csv'
        file1 = open(path1, "a+")
    except:
        #path1 = "gridSearch" + modelName + ".csv"
        path1 = "adverse" + modelName + ".csv"
        file1 = open(path1, "a+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0


def grid_search_EpochBatch(modelName, features, labels):

    # This is for computing class weight
    classes = [0, 1]
    class_weight = compute_class_weight("balanced", classes, labels)
    print(class_weight)
    test_ratio = .8
    percent = (1-test_ratio)*100

    epochs = [16]
    batch_size = [10]
    neurons = [45]
    optimizer = ['Nadam']
    weight_constraint = [3, 4, 5]
    dropout_rate = [0, 0.1, 0.2, 0.3]

    paramGrid = dict(epochs=epochs, batch_size=batch_size, optimizer=optimizer,
                     dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                     neurons=neurons)

    if modelName == "oneLayer":
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
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/noClassWeightDropout16epoch' + modelName + '.csv'
        file1 = open(path1, "w+")
    except:
        path1 = "gridSearch" + modelName + ".csv"
        file1=open(path1, "w+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0



if __name__ == "__main__":
    main()

