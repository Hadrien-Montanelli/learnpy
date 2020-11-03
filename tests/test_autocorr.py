#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:17:42 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from utilities import csv_to_array
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from autocorr import autocorr

# AR(1)-type time series:
series = csv_to_array('../dataset/time_series_ar1.csv')
sample_acf = autocorr(series)
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf

# MA(1)-type time series:
series = csv_to_array('../dataset/time_series_ma1.csv')
sample_acf = autocorr(series)
plt.figure()
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf