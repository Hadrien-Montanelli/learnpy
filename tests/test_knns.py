#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:25 2020

@author: montanelli
"""
import sys
sys.path.append('../')
from IPython import get_ipython
from knns import knns
from utilities import csv_to_array

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Use the k-NNs algorithm on the following data (in cm):
training_data = csv_to_array('../dataset/heights_training.csv')
testing_data = csv_to_array('../dataset/heights_testing.csv')
output = knns(training_data, testing_data, 1)
error = output
print('Error (1D):', sum(error), '\n')

# Use the k-NNs algorithm on the following data (in [cm, kg]):
training_data = csv_to_array('../dataset/heights_weights_training.csv')
testing_data = csv_to_array('../dataset/heights_weights_testing.csv')
output = knns(training_data, testing_data, 1)

# Post-processing:
output = knns(training_data, testing_data, 1)
error = output
print('Error (2D):', sum(error), '\n')