import numpy as np
import timeit
import pandas
import datetime
import argparse

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from .keras_models import create_one_layer, create_dualInputSimple, create_dualInputLarge

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Nadam

def main():

    args = parse_arguments()

    perm_inputs, feat_inputs, labels = vectorize(args["good_path"], args["mal_path"])
    perm_width = len(perm_inputs[0])
    feat_width = len(feat_inputs[0])

    grid_search(args, perm_inputs, feat_inputs, labels)
    return
    #comb_width = len(comb_inputs[0])

    avg = 0
    for x in split_acc:
        avg+=x
    print 'avg acc is :' + str(avg/5)

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
    #comb_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=comb_pattern))

    time0 = timeit.default_timer()
    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

    #comb_inputs_sparse = comb_vect.fit_transform(samples)
    #comb_inputs_dense = comb_inputs_sparse.todense()
    #comb_inputs = np.array(comb_inputs_dense)

    return perm_inputs, feat_inputs, labels

def grid_search(args, perm_inputs, feat_inputs, labels):
    '''
    The below method is a modified implementation of the gridsearch method in
    KTFBinClass.py that manually iterates through all params via nested loops
    this method has been created to allow for multi_input neural networks
    '''
    perm_width = int(len(perm_inputs[0]))
    feat_width = int(len(feat_inputs[0]))
    print type(perm_width)
    input_ratios = args["input_ratio"]
    splits = args["splits"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    neurons = args["neurons"]
    modelName = args["model"]
    print modelName



    for m in modelName:
        for r in args["train_ratio"]:
            percent=float(r)/100
            print percent
            sss = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=percent)
            for ir in input_ratios:
                for epoch in epochs:
                    for batch in batch_size:
                        for size in neurons:
                            print str(r) + ' ' + str(epoch) + ' ' + str(batch)
                            for train_index, test_index in sss.split(perm_inputs, labels):
                                print m
                                if m == "oneLayer":
                                    print 'here'
                                    model = create_one_layer(optimizer='nadam', data_width=perm_width)
                                elif m == "dual_simple":
                                    model = create_dualInputSimple(input_ratio=.125, perm_width=perm_width, feat_width=feat_width)
                                elif m == "dual_large":
                                    model = create_dualInputLarge(dropout_rate=.1, input_ratio=.125, perm_width=perm_width, feat_width=feat_width)

                                perm_train, perm_test = perm_inputs[train_index], perm_inputs[test_index]
                                feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
                                labels_train, labels_test = labels[train_index], labels[test_index]
                                if m != "oneLayer":
                                    print "multi_input: " + str(m)
                                    model.fit([perm_train, feat_train], labels_train, epochs=epoch, batch_size=batch)
                                    labels_pred = model.predict([perm_test, feat_test], batch_size=batch)

                                elif m == "oneLayer":
                                    print "single_input: " + str(m)
                                    model.fit(perm_train, labels_train, epochs=epoch, batch_size=batch)
                                    labels_pred = model.predict(perm_test, batch_size=batch)

                                labels_pred = (labels_pred > 0.5)
                                cm = confusion_matrix(labels_test, labels_pred)
                                print "acc: " + str(float((cm[0][0] + cm[1][1]))/len(labels_pred))




                                #split_acc = np.append(split_acc, model.evaluate([perm_test, feat_test], labels_test)[1])
                                #comb_train, comb_test, = comb_inputs[train_index], comb_inputs[test_index]
                                #model.fit(comb_train, labels_train, epochs=8, batch_size=64)
                                #split_acc = np.append(split_acc, model.evaluate(comb_test, labels_test)[1])

    return


    print("%s Best: %f using %s" % (modelName, grid_fit.best_score_, grid_fit.best_params_))

    d = datetime.datetime.today()
    month = str( '%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    min = str('%02d' % d.minute)

    df = pandas.DataFrame(grid_fit.cv_results_)
    try:
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/gridSearch' + modelName + month + day + hour + min + '.csv'
        file1 = open(path1, "w+")
    except:
        path1 = "gridSearch" + modelName + ".csv"
        file1 = open(path1, "w+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gp", "--good_path", help="Good File Path")
    parser.add_argument("-mp", "--mal_path", help="Malware File Path")
    parser.add_argument("-ad", "--adverse", help="Turns on Adversarial Learning")
    parser.add_argument("-m", "--mode", help="Choose mode: full, grid")
    parser.add_argument("-e", "--epochs", help="Number of Epochs, can be list for grid search", type=int, nargs="*")
    parser.add_argument("-tr", "--train_ratio", nargs="*", type=int,
                        help="Set Test Ratios. Enter as a percent (20,40,60,80). Can be a list space delimited")
    parser.add_argument("-bs", "--batch_size", nargs="*", type=int,
                        help="Batch size. Can be a list space delimited")
    parser.add_argument("-n", "--neurons", nargs="*", type=int,
                        help="Number of Neurons. Can be a list space delimited")
    parser.add_argument("-o", "--optimizer", nargs="*",
                        help="Optimizers. Can be a list space delimited")
    parser.add_argument("-w", "--weight_constraint", nargs="*", type=int,
                        help="Weight Constraint. Can be a list space delimited")
    parser.add_argument("-d", "--dropout", nargs="*", type=int,
                        help="Dropout. Enter as percent (10,20,30,40...). Can be a list space delimited.")
    parser.add_argument("-model", "--model", help="Select which model to run: all \
    , one_layer, four_decr, four_same")
    parser.add_argument("-s", "--splits", help="Number of Splits for SSS", type=int)
    parser.add_argument("-ir", "--input_ratio", help="ratio of layer width between \
     features and permissions layers", type=float, nargs="*")

    args = parser.parse_args()

    arguments = {}

    if args.good_path:
        good_path = args.good_path
        arguments["good_path"] = good_path
    else:
        print("Needs Good Path with -gp or --good_path")
        sys.exit()

    if args.mal_path:
        mal_path = args.mal_path
        arguments["mal_path"] = mal_path
    else:
        print("Needs Malware Path with -mp or --mal_path")
        sys.exit()

    if args.adverse:
        adverse = True
    else:
        adverse = False
    arguments["adverse"] = adverse

    if args.mode == "grid":
        mode = "grid"
        print("Mode is %s" % mode)
    else:
        mode = "full"
        print("Mode is %s" % mode)
    arguments["mode"] = mode

    if args.model == "all":
        model = ["oneLayer", "dual_simple", "dual_large"]
    elif args.model in ["oneLayer", "fourDecr", "fourSame", "dual_inputLarge", "dual_inputSimple"]:
        model = [args.model]
    else:
        print("Defaulting to All models")
        model = ["oneLayer", "fourDecr", "fourSame"]
    arguments["model"] = model

    if args.epochs:
        epochs = args.epochs
    else:
        print("Defaulting to 16 epochs")
        epochs = [16]
    arguments["epochs"] = epochs
    if args.train_ratio:
        train_ratio = args.train_ratio
    else:
        print("Defaulting to testing all ratios")
        train_ratio = [20, 40, 60, 80]
    arguments["train_ratio"] = train_ratio

    if args.batch_size:
        batch_size = args.batch_size
    else:
        print("Defaulting to Batch Size 10")
        batch_size = [10]
    arguments["batch_size"] = batch_size

    if args.neurons:
        neurons = args.neurons
    else:
        print("Defaulting to 45 Neurons")
        neurons = [45]
    arguments["neurons"] = neurons

    if args.optimizer:
        optimizer = args.optimizer
    else:
        print("Defaulting to NADAM Optimizer")
        optimizer = "Nadam"
    arguments["optimizer"] = optimizer

    if args.weight_constraint:
        weight_constraint = args.weight_constraint
    else:
        print("Defaulting to weight constraint 5")
        weight_constraint = [5]
    arguments["weight_constraint"] = weight_constraint

    if args.dropout:
        dropout = args.dropout
    else:
        print("Defaulting to dropout of 10%")
        dropout = [10]
    arguments["dropout"] = dropout

    if args.splits:
        splits = args.splits
    else:
        print("Defaulting to 1 SSS Split")
        splits = [1]
    arguments["splits"] = splits

    if args.input_ratio:
        input_ratio = args.input_ratio
    else:
        print("default to .25 input ratio")
        input_ratio = [.25]
    arguments["input_ratio"] = input_ratio

    return arguments


if __name__ == "__main__":
    main()
