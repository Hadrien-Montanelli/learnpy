#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:25 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# System imports:
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Standard library imports:
import matplotlib.pyplot as plt
from sklearn import datasets

# Learnpy imports:
import misc
import supervised

# %% Simple example.
    
# Get the data (in [cm, kg]):
training_data = misc.csv_to_array('../../dataset/2d_training.csv')
testing_data = misc.csv_to_array('../../dataset/2d_testing.csv')
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
k = 1
classifier = supervised.knns(n_input, n_train, k)

# Train:
classifier.train(X_train, Y_train)

# Classify:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')
    
# %% More complicated example from SKLEARN.
    
# Get the data:
n_train = 1000
n_test = 500
n_samples = n_train + n_test
X, Y = datasets.make_classification(n_samples = n_samples, random_state = 123)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]
    
# Intialize:
n_input = len(X[0])
k = 1
classifier = supervised.knns(n_input, n_train, k)

# Train:
classifier.train(X_train, Y_train)

# Predict:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')