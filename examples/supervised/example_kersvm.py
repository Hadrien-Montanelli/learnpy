#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:11:15 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# System imports:
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Standard library imports:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Learnpy imports:
import supervised

# %% Simple example.

# Get the data:
training_data = np.zeros([50, 3])
testing_data = np.zeros([50, 3])
for i in range(50):
    training_data[i, 0] = -1 + 2*np.random.uniform()
    training_data[i, 1] = -1 + 2*np.random.uniform()
    testing_data[i, 0] = -1 + 2*np.random.uniform()
    testing_data[i, 1] = -1 + 2*np.random.uniform()
    if (training_data[i, 0]**2 + training_data[i, 1]**2 < 0.4):
        training_data[i, 2] = 0
    else:
        training_data[i, 2] = 1
    if (testing_data[i, 0]**2 + testing_data[i, 1]**2 < 0.4):
        testing_data[i, 2] = 0
    else:
        testing_data[i, 2] = 1
n_input = 2
n_train = len(training_data)
n_test = len(testing_data)
X_train = training_data[:, :n_input]
Y_train = training_data[:, -1]
X_test = testing_data[:, :n_input]
Y_test = testing_data[:, -1]

# Plot the training data:
number_rows = len(training_data)
fig = plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal')
for i in range(number_rows):
    if training_data[i,-1] == 0:
        color = '.r'
    else:
        color = '.b'
    plt.plot(training_data[i,0], training_data[i,1], color)
    
# Plot the testing data:
number_rows = len(testing_data)
for i in range(number_rows):
    if testing_data[i,-1] == 0:
        color = 'xr'
    else:
        color = 'xb'
    plt.plot(testing_data[i,0],testing_data[i,1],color)

# Intialize:
kernel = lambda x,y: (1 + x @ y)**2
classifier = supervised.kersvm(n_input, n_train, kernel)

# Train:
classifier.train(X_train, Y_train)

# Classify:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')

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
kernel = lambda x,y: (1 + x @ y)**3
classifier = supervised.kersvm(n_input, n_train, kernel)

# Train:
classifier.train(X_train, Y_train)

# Predict:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')