#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:39 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Learnpy imports:
from learnpy.misc import csv_to_array
from learnpy.supervised import svm

# %% Simple example.
    
# Get the data (in [cm, kg]):
training_data = csv_to_array('../../dataset/2d_training.csv')
testing_data = csv_to_array('../../dataset/2d_testing.csv')
n_input = 2
n_train = len(training_data)
n_test = len(testing_data)
X_train = training_data[:, :n_input]
Y_train = training_data[:, -1]
X_test = testing_data[:, :n_input]
Y_test = testing_data[:, -1]

# Plot the training data:
fig = plt.figure()
for i in range(n_train):
    if Y_train[i] == 0:
        color = '.r'
    else:
        color = '.b'
    plt.plot(X_train[i,0], X_train[i,1], color)
    
# Plot the testing data:
for i in range(n_test):
    if Y_test[i] == 0:
        color = 'xr'
    else:
        color = 'xb'
    plt.plot(X_test[i,0], X_test[i,1], color)
    
# Intialize:
classifier = svm(n_input, n_train)

# Train:
classifier.train(X_train, Y_train)

# Classify:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')

# Plot decision boundary:
W = classifier.params['W']
W0 = classifier.params['W0']
xx = np.linspace(np.min(X_test[:,0]), np.max(X_test[:,0]), 100)
plt.plot(xx, -W[0]/W[1]*xx - W0/W[1], 'k')

# %% More complicated example from SKLEARN.
    
# Get the data:
n_train = 100
n_test = 50
n_samples = n_train + n_test
X, Y = datasets.make_classification(n_samples = n_samples, random_state = 123)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]
    
# Intialize:
n_input = len(X[0])
classifier = svm(n_input, n_train)

# Train:
classifier.train(X_train, Y_train)

# Predict:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')