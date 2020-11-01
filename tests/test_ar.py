#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:21:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from utilities import csv_to_array
import matplotlib.pyplot as plt
import numpy as np
from ar import ar

# Test AR(1):
data = csv_to_array('../dataset/time_series_ar1.csv')
plt.plot(data, '.-')
p = 1
alpha, beta = ar(data, p)
print([alpha, beta])
prediction = np.zeros(len(data))
prediction[0] = data[0]
for k in range(len(data)-1):
    prediction[k+1] = alpha + beta[0]*data[k]
plt.plot(prediction, '.-')

# Test AR(2):
data = csv_to_array('../dataset/time_series_ar2.csv')
plt.figure()
plt.plot(data, '.-')
p = 2
alpha, beta = ar(data, p)
print([alpha, beta])
prediction = np.zeros(len(data))
prediction[0] = data[0]
prediction[1] = data[1]
for k in range(len(data)-2):
    prediction[k+2] = alpha + beta[0]*data[k+1] + beta[1]*data[k]
plt.plot(prediction, '.-')