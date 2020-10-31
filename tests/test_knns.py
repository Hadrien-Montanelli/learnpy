#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:25 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../supervised')
sys.path.append('../misc')
from knns import knns
from utilities import csv_to_array

# Classify the following data (in cm) with the k-NNs algorithm:
training_data = csv_to_array('../dataset/heights_training.csv')
testing_data = csv_to_array('../dataset/heights_testing.csv')
output = knns(training_data, testing_data, 1)
print('Error (1D):', output, '\n')

# Classify the following data (in [cm, kg]) with the k-NNs algorithm:
training_data = csv_to_array('../dataset/heights_weights_training.csv')
testing_data = csv_to_array('../dataset/heights_weights_testing.csv')
output = knns(training_data, testing_data, 1)
print('Error (2D):', output, '\n')