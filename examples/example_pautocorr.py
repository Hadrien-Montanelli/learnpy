#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:59:21 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from utilities import csv_to_array
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from pautocorr import pautocorr

# AR(1)-type time series:
series = csv_to_array('../dataset/time_series_ar1.csv')
sample_pacf = pautocorr(series)
plt.plot(sample_pacf[1:], '-')
sample_pacf_statsmodels = pacf(series, nlags=len(series)-1, method='ywm')
plt.plot(sample_pacf_statsmodels[1:], '--')
error = np.linalg.norm(sample_pacf - sample_pacf_statsmodels)
print('Error: ', error) # compare with statsmodels' pacf

# MA(1)-type time series:
series = csv_to_array('../dataset/time_series_ma1.csv')
sample_pacf = pautocorr(series)
plt.figure()
plt.plot(sample_pacf[1:], '-')
sample_pacf_statsmodels = pacf(series, nlags=len(series)-1, method='ywm')
plt.plot(sample_pacf_statsmodels[1:], '--')
error = np.linalg.norm(sample_pacf - sample_pacf_statsmodels)
print('Error: ', error) # compare with statsmodels' pacf