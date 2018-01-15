from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm



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
    n2 = neurons // 2 if neurons // 2 > 0 else 1
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
