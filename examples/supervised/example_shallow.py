#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:15:36 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from sklearn import datasets

# Learnpy imports:
from learnpy.supervised import shallow

# %% Example from SKLEARN.

# Get the data:
n_features = 20
n_train = 4000
n_test = 1000
n_samples = n_train + n_test
X, Y = datasets.make_classification(n_samples = n_samples, 
                                    n_features = n_features, 
                                    random_state = 123)
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]
    
# Intialize:
n_input = len(X[0])
n_neurons = 20
options = {'acc': 'entropy', 'cost': 'entropy', 'disp': True, 'jtol': 0.2, 
            'lr': 1, 'maxiter': 2000, 'output': 'sigmoid'}
classifier = shallow(n_input, n_train, n_neurons, options)

# Train:
classifier.train(X_train, Y_train)

# Predict:
Y_hat = classifier.classify(X_test)

# Compute accuracy:
acc = classifier.accuracy(Y_test, Y_hat)
print(f'Accuracy: {acc}%')