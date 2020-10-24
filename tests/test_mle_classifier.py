#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

@author: montanelli
"""
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
from mle_classifier import mle_classifier
from utilities import csv_to_array

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Fit the following data (in cm) with a normal distribution & MLE:
training_data = csv_to_array('../data/heights_training.csv')
testing_data = csv_to_array('../data/heights_testing.csv')
prior = np.array([0.5, 0.5])
output = mle_classifier(training_data, testing_data, prior, 'normal')

# Post-processing:
randvar_0 = output[0]
randvar_1 = output[1]   
error = output[4]   
print('Error (1D):', sum(error), '\n')
randvar_0.plot('b')
randvar_1.plot('r')
for k in range(len(testing_data)):
    if training_data[k,1] == 0:
        color = 'xb'
    else:
        color = 'xr'
    plt.plot(testing_data[k,0], 0, color)
    if error[k] != 0:
        plt.plot(testing_data[k,0], 0, '+k')
        
# Fit the following data (in cm) with a normal distribution & MLE:
training_data = csv_to_array('../data/heights_weights_training.csv')
testing_data = csv_to_array('../data/heights_weights_testing.csv')
prior = np.array([0.5, 0.5])
output = mle_classifier(training_data, testing_data, prior, 'normal')

# Post-processing:
randvar_0 = output[0]
randvar_1 = output[1]
error = output[4]   
print('Error (2D):', sum(error), '\n')