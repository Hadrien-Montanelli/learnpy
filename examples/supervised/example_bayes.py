#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
from sklearn import datasets

# Learnpy imports:
from learnpy.misc import csv_to_array
from learnpy.supervised import bayes

# %% Simple example.
    
# Get the data (in [cm, kg]):
training_data = csv_to_array('../../datasets/2d_training.csv')
testing_data = csv_to_array('../../datasets/2d_testing.csv')
n_input = 2
n_train = len(training_data)
n_test = len(testing_data)
X_train = training_data[:, :n_input]
Y_train = training_data[:, -1]
X_test = testing_data[:, :n_input]
Y_test = testing_data[:, -1]

# Intialize:
model = 'normal'
classifier = bayes(n_input, n_train, model)

# Train:
classifier.train(X_train, Y_train)

# Classify:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')

randvar = classifier.params['randvar']
randvar_cm_0 = randvar[0][0]
randvar_cm_1 = randvar[0][1]
randvar_kg_0 = randvar[1][0]
randvar_kg_1 = randvar[1][1]
plt.figure()
randvar_cm_0.plot('b')
randvar_cm_1.plot('r')
plt.figure()
randvar_kg_0.plot('b')
randvar_kg_1.plot('r')

# %% More complicated example from SKLEARN.
    
# Get the data:
n_train = 4000
n_test = 1000
n_samples = n_train + n_test
X, Y = datasets.make_classification(n_samples = n_samples, random_state = 123)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]
    
# Intialize:
n_input = len(X[0])
model = 'normal'
classifier = bayes(n_input, n_train, model)

# Train:
classifier.train(X_train, Y_train)

# Predict:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')