#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:39 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../supervised')
sys.path.append('../misc')
from perceptron import perceptron
from utilities import csv_to_array
import matplotlib.pyplot as plt
import numpy as np

# Get the data (in [cm, kg]):
training_data = csv_to_array('../dataset/2d_training.csv')
testing_data = csv_to_array('../dataset/2d_testing.csv')
training_data[:,1] = training_data[:,1]
testing_data[:,1] = testing_data[:,1]

# Plot the training data:
number_rows = len(training_data)
fig = plt.figure()
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
    plt.plot(testing_data[i,0], testing_data[i,1], color)
    
# Classify the data with the perceptron algorithm:
output = perceptron(training_data, testing_data)

# Print error and weights, and plot decision boundary:
error = output[2]
print('Error:  ', sum(error))
w = output[1]
w_0 = output[0]
xx = np.linspace(np.min(testing_data[:,0]), np.max(testing_data[:,0]), 100)
plt.plot(xx, -w[0]/w[1]*xx - w_0/w[1], 'k')
print('Weights:', [w_0, w])