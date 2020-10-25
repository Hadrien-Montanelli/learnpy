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
import pylab

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# # Use the Perceptron algorithm on the following data (in [cm, kg]):
training_data = csv_to_array('../data/heights_weights_training.csv')
testing_data = csv_to_array('../data/heights_weights_testing.csv')
output = perceptron(training_data, testing_data)
w0 = output[0]
w = output[1]
error = output[2]
print('Error (1D):', sum(error), '\n')

number_rows = len(training_data)
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
xx = np.linspace(150, 200, 100)
pylab.plot(xx, -w[0]/w[1]*xx - w0/w[1], 'k')