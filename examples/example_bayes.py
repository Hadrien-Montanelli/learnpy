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
from bayes import bayes
from utilities import csv_to_array
import matplotlib.pyplot as plt
import numpy as np

# Classify the following data (in cm) with the MLE algorithm:
training_data = csv_to_array('../dataset/1d_training.csv')
testing_data = csv_to_array('../dataset/1d_testing.csv')
prior = np.array([0.5, 0.5])
randvar, error = bayes(training_data, testing_data, prior, 'normal')
randvar_0 = randvar[0][0]
randvar_1 = randvar[0][1]  
print('Error (1D):', error, '\n')
randvar_0.plot('b')
randvar_1.plot('r')
        
# Classify the following data (in [cm, kg]) with the MLE algorithm:
training_data = csv_to_array('../dataset/2d_training.csv')
testing_data = csv_to_array('../dataset/2d_testing.csv')
prior = np.array([0.5, 0.5])
randvar, error = bayes(training_data, testing_data, prior, 'normal')
randvar_cm_0 = randvar[0][0]
randvar_cm_1 = randvar[0][1]
randvar_kg_0 = randvar[1][0]
randvar_kg_1 = randvar[1][1]
print('Error (2D):', error, '\n')
plt.figure()
randvar_cm_0.plot('b')
randvar_cm_1.plot('r')
plt.figure()
randvar_kg_0.plot('b')
randvar_kg_1.plot('r')