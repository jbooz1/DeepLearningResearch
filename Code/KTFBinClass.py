import numpy as np
import pandas
import timeit
import sys
import argparse

from .keras_models import create_binaryDecrease, create_fourDecrLayer, create_fourSameLayer, create_one_layer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gp", "--good_path", help="Good File Path")
    parser.add_argument("-mp", "--mal_path", help="Malware File Path")
    parser.add_argument("-ad", "--adverse", help="Turns on Adversarial Learning")
    parser.add_argument("-m", "--mode", help="Choose mode: full, grid")
    parser.add_argument("-e", "--epochs", help="Number of Epochs")
    parser.add_argument("-tr", "--test_ratio", nargs="+", type=int,
                        help="Set Test Ratios. Enter as a percent (20,40,60,80). Can be a list space delimited")
    parser.add_argument("-model", "--model", help="Select which model to run: all, one_layer, four_decr, four_same")

    args = parser.parse_args()

    if args.good_path:
        good_path = args.good_path
    else:
        print("Needs Good Path with -gp or --good_path")
        sys.exit()

    if args.mal_path:
        mal_path = args.mal_path
    else:
        print("Needs Malware Path with -mp or --mal_path")
        sys.exit()

    if args.adverse:
        adverse = True
    else:
        adverse = False

    if args.mode == "grid":
        mode = "grid"
        print("Mode is %s" % mode)
    else:
        mode = "full"
        print("Mode is %s" % mode)

    if args.model == "all" :
        model = ["one_layer", "four_decr", "four_same"]
    elif args.model in ["one_layer", "four_decr", "four_same"] :
        model = args.model
    else :
        print("Defaulting to All models")
        model = ["one_layer", "four_decr", "four_same"]

    if args.test_ratio :
        ratios = args.test_ratio
    else :
        print("Defaulting to testing all ratios")
        ratios = [20,40,60,80]

    features, labels = vectorize(good_path, mal_path, adverse)
    if mode == "grid" :
        for m in model :
            grid_search_EpochBatch(m, features, labels)
            
    if mode == "full" :
        for m in model :
            for r in ratios :
                full_run(m, features, labels, r)

    return 0


def vectorize(good_path, mal_path, adverse):
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

    count_vect = CountVectorizer(input=u'content', analyzer=u'word',
                                 token_pattern='(\\b(:?uses-|optional-)?permission:\s[^\s]*)')
    time0 = timeit.default_timer()
    features = count_vect.fit_transform(perms)
    features = features.todense()
    features = np.array(features)

    if adverse:
        print("Adversarial Learning")
        count1 = 0
        count2 = 0
        gdprmsize = np.size(gdprm, 0)
        mlprmszie = np.size(mlprm, 0)
        for i in range(0, gdprmsize // 10):
            if labels[i] == 0:
                count1 += 1
                labels[i] = 1
        print("Good Permissions Changed: %d" % count1)
        for i in range(gdprmsize, gdprmsize + mlprmszie // 10):
            if labels[i] == 1:
                count2 += 1
                labels[i] = 0
        print("Malware Permissions Changed: %d" % count2)
        print("Total Permissions Changed: %d" % count1 + count2)

    print("Done Vectorizing Data")
    return features, labels


def full_run(modelName, features, labels, test_ratio):
    epochs = 16
    batch_size = 10
    neurons = 45
    optimizer = 'Nadam'
    weight_constraint = 5
    dropout_rate = 0.1
    percent = 1 - test_ratio

    model_params = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, optimizer=optimizer,
                        weight_constraint=weight_constraint, dropout_rate=dropout_rate)
    fit_params = dict(batch_size=batch_size, epochs=epochs)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    if modelName == "oneLayer":
        model = KerasClassifier(build_fn=create_one_layer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "binaryDecrease":
        model = KerasClassifier(build_fn=create_binaryDecrease, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "fourSame":
        model = KerasClassifier(build_fn=create_fourSameLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "fourDecr":
        model = KerasClassifier(build_fn=create_fourDecrLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=0)
    cv_result = cross_validate(model, features, labels, cv=sss, fit_params=fit_params, return_train_score=True,
                               scoring=scoring, verbose=2)

    df = pandas.DataFrame(cv_result)
    try:
        # path1 = '/home/lab309/pythonScripts/testResults/deep_results/finalCV' + str(percent) + modelName + '.csv'
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/adverse/' + str(percent) + modelName + '.csv'
        file1 = open(path1, "a+")
    except:
        # path1 = "gridSearch" + modelName + ".csv"
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
    percent = (1 - test_ratio) * 100

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
        file1 = open(path1, "w+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0


if __name__ == "__main__":
    main()
