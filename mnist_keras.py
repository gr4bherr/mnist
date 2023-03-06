#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 4

(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# one hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
 
X_train = X_train/255 
X_test = X_test/255
 
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

model = Sequential()
model.add(Dense(64, input_dim=28*28, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)

score = model.evaluate(X_test, y_test, verbose=1)
print(type(score))
print('Test score:', score[0])
print('Test accuracy:', score[1])
 
