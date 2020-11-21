#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:21:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# System imports:
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Standard library imports:
import matplotlib.pyplot as plt
import numpy as np

# Learnpy imports:
import misc
import timeseries

# %% Examples.

# Test AR(1):
series = misc.csv_to_array('../../dataset/time_series_ar1.csv')
plt.plot(series, '.-')
p = 1
alpha, beta = timeseries.ar(series, p)
print([alpha, beta])
prediction = np.zeros(len(series))
prediction[0] = series[0]
for k in range(len(series)-1):
    prediction[k+1] = alpha + beta[0]*series[k]
plt.plot(prediction, '.-')

# Test AR(2):
series = misc.csv_to_array('../../dataset/time_series_ar2.csv')
plt.figure()
plt.plot(series, '.-')
p = 2
alpha, beta = timeseries.ar(series, p)
print([alpha, beta])
prediction = np.zeros(len(series))
prediction[0] = series[0]
prediction[1] = series[1]
for k in range(len(series)-2):
    prediction[k+2] = alpha + beta[0]*series[k+1] + beta[1]*series[k]
plt.plot(prediction, '.-')