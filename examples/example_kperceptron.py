#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:45:24 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../supervised')
sys.path.append('../misc')
from kperceptron import kperceptron
import matplotlib.pyplot as plt
import numpy as np

# Get the data:
N = 40
training_data = np.zeros([N, 3])
testing_data = np.zeros([N, 3])
for i in range(N):
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
    
# Classify the data with the kernelized perceptron algorithm:
kernel = lambda x,y: (1 + x @ y)**2
output = kperceptron(training_data, testing_data, kernel)
    
# Print error and weights:
error = output[2]
print('Error:  ', sum(error))
w = output[1]
w_0 = output[0]
print('Weights:', [w_0, w])