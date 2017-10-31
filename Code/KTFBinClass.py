import numpy as np
import pandas
import sklearn
import random
import time
import timeit
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from timeit import default_timer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder



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
	
	print "Grid Search for One Layer"
	grid_search_EpochBatch("oneLayer", features, labels)
	print "Grid Search for Binary Decreasing Layers"
	grid_search_EpochBatch("binaryDecrease", features, labels)
	print "Grid Search for Four equal Layers"
	grid_search_EpochBatch("fourSame", features, labels)
	print "Grid Search for Four Decr Layers"
	grid_search_EpochBatch("fourDecr", features, labels)
	
	return 0

	
def grid_search_EpochBatch(modelName, features, labels):
	epochs = [1, 2, 4, 8, 16, 32]
	batch_size = [10, 100, 200, 500, 1000, 1500, 2000, 5000]
	paramGrid = dict(epochs=epochs, batch_size=batch_size)
	test_ratio=.8
		
	
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
	path1 = '/home/lab309/pythonScripts/testResults/deep_results/epochBatchTrain' + modelName + '.csv'
	file1 = open(path1, "w+")
	df.to_csv(file1, index=True)
	file1.close()
	
	return 0
	
	
	
def cross_val(modelName, features, labels, length, test_ratio, splits, tensorSize, epoch, batch):
	
	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=0)
	if modelName == "one" :
		print "\nOne layer with size %d" % tensorSize
		model = KerasClassifier(build_fn=pickle.loads(create_one_layer)(tensorSize), epochs=epoch, batch_size=batch, verbose=0)
	elif modelName == "binaryDecrease":
		print "\nbinary decr layers with size %d" % tensorSize
		model = KerasClassifier(build_fn=pickle.loads(create_binaryDecrease)(tensorSize), epochs=epoch, batch_size=batch, verbose=0)
	elif modelName == "fourSame":
		print "\nfour layers with size %d" % tensorSize
		model = KerasClassifier(build_fn=pickle.loads(create_fourSameLayer)(tensorSize), epochs=epoch, batch_size=batch, verbose=0)
	elif modelName == "fourDecr":
		print "\nfour decr layers with size %d" % tensorSize
		model = KerasClassifier(build_fn=pickle.loads(create_fourDecrLayer)(tensorSize), epochs=epoch, batch_size=batch, verbose=0)
		
	start_time = time.time()
	results = cross_val_score(model, features, labels, cv=sss, scoring='accuracy')	
	
	print(1-test_ratio, results.mean(), (time.time()-start_time)/splits)

	
	
def standard_test():
	epoch = 5
	batch=1000
	ratios = [.2, .4, .6, .8]
	
	for r in ratios:
		#estimator = KerasClassifier(build_fn=create_one_layer(), nb_epoch=epoch, batch_size=batch, verbose=2)
		#model = create_one_layer()
		#test_model("one", features, labels, r, epoch, batch)
		cross_val("one", features, labels, inputSize, r, 5, 25, epoch, batch)
		
	for r in ratios:
		#estimator = KerasClassifier(build_fn=create_binaryDecrease(), nb_epoch=epoch, batch_size=batch, verbose=2)
		#model = create_binaryDecrease()
		#test_model("binaryDecrease", features, labels, r, epoch, batch)
		cross_val("binaryDecrease", features, labels,inputSize, r, 5, 25, epoch, batch)
		
	for r in ratios:
		#estimator = KerasClassifier(build_fn=create_four5kLayer(), nb_epoch=epoch, batch_size=batch, verbose=2)
		#model = create_four5kLayer()
		#test_model("four5k", features, labels, r, epoch, batch)
		cross_val("fourSame", features, labels,inputSize, r, 5, 25, epoch, batch)
		
	for r in ratios:
		#estimator = KerasClassifier(build_fn=create_fourDecrLayer(), nb_epoch=epoch, batch_size=batch, verbose=2)
		#model = create_fourDecrLayer()
		#test_model("fourDecr", features, labels, r, epoch, batch)
		cross_val("fourDecr", features, labels,inputSize, r, 5, 25, epoch, batch)
		
	return 0
	
	
def create_one_layer(size=25):
	#baseline Model
	model = Sequential()
	#The first param in Dense is the number of neurons in the first hidden layer
	model.add(Dense(size, input_dim=22300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

	
def create_binaryDecrease(size=25):
	model = Sequential()
	#The first param in Dense is the number of neurons in the first hidden layer
	model.add(Dense(size, input_dim=22300, kernel_initializer='normal', activation='relu'))
	while (size/2 >=1):
		model.add(Dense(size/2, kernel_initializer='normal', activation='relu'))
		size/=2
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
def create_fourSameLayer(size=25):
	#baseline Model
	model = Sequential()
	#The first param in Dense is the number of neurons in the first hidden layer
	model.add(Dense(size/3, input_dim=22300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
def create_fourDecrLayer(size=25):
	#baseline Model
	model = Sequential()
	#The first param in Dense is the number of neurons in the first hidden layer
	model.add(Dense(size, input_dim=22300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(size/4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
	
	
if __name__ == "__main__":
	main()
	
