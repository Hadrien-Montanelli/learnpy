#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:39 2020

@author: montanelli
"""
import sys
sys.path.append('../')
from IPython import get_ipython
from perceptron import perceptron
from utilities import csv_to_array
import matplotlib.pyplot as plt
import numpy as np

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Use the Perceptron algorithm on the following data (in [cm, kg]):
training_data = csv_to_array('../data/heights_weights_training.csv')
testing_data = csv_to_array('../data/heights_weights_testing.csv')
bias = -200
training_data[:,1] = training_data[:,1] + bias
testing_data[:,1] = testing_data[:,1] + bias
output = perceptron(training_data, testing_data)
number_rows = len(training_data)
fig = plt.figure()
for i in range(number_rows):
    if training_data[i,-1] == 0:
        color = '.r'
    else:
        color = '.b'
    plt.plot(training_data[i,0],training_data[i,1],color)
number_rows = len(testing_data)
for i in range(number_rows):
    if testing_data[i,-1] == 0:
        color = 'xr'
    else:
        color = 'xb'
    plt.plot(testing_data[i,0],testing_data[i,1],color)
error = output[2]
print('Error:', sum(error), '\n')
w = output[1]
w0 = output[0]
xx = np.linspace(np.min(testing_data[:,0]), np.max(testing_data[:,0]), 100)
plt.plot(xx, -w[0]/w[1]*xx - w0/w[1], 'k')
print([w0,w])