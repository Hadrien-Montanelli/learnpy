#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../supervised')
sys.path.append('../misc')
from mle import mle
from utilities import csv_to_array
import numpy as np

# Classify the following data (in cm) with the MLE algorithm:
training_data = csv_to_array('../dataset/heights_training.csv')
testing_data = csv_to_array('../dataset/heights_testing.csv')
prior = np.array([0.5, 0.5])
output = mle(training_data, testing_data, prior, 'normal')
randvar_0 = output[0]
randvar_1 = output[1]   
error = output[2]   
print('Error (1D):', error, '\n')
randvar_0.plot('b')
randvar_1.plot('r')
        
# Classify the following data (in [cm, kg]) with the MLE algorithm:
training_data = csv_to_array('../dataset/heights_weights_training.csv')
testing_data = csv_to_array('../dataset/heights_weights_testing.csv')
prior = np.array([0.5, 0.5])
output = mle(training_data, testing_data, prior, 'normal')
randvar_0 = output[0]
randvar_1 = output[1]
error = output[2]   
print('Error (2D):', error, '\n')
randvar_0.plot()
randvar_1.plot()