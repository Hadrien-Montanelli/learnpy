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

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Fit the following data (in cm) with a normal distribution & MLE:
training_data = np.array([[170,1], [172,1], [180,1], [169,0], [157,0], [166,0]])
testing_data = np.array([[175,1], [185,1], [150,1], [159,0], [160,0], [180,0]])
prior = np.array([0.5, 0.5])
output = mle_classifier(training_data, testing_data, prior, 'normal')
print(output)