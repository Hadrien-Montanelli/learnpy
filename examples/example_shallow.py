#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:57:17 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../supervised')
sys.path.append('../misc')
from shallow import shallow
from sklearn import datasets

# Get data:
X, y = datasets.make_classification(n_samples=5000, random_state=123)

X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]
    
# Classify the data with a shallow network:
N = 100
acc = shallow(X_train, X_test, y_train, y_test, N)

# Print accuracy:
print(f'Accuracy: {acc*100}%')