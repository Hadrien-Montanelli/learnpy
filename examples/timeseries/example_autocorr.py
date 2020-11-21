#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:17:42 2020

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
from statsmodels.tsa.stattools import acf

# Learnpy imports:
import misc
import timeseries

# %% Examples.

# AR(1)-type time series:
series = misc.csv_to_array('../../dataset/time_series_ar1.csv')
sample_acf = timeseries.autocorr(series)
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf

# MA(1)-type time series:
series = misc.csv_to_array('../../dataset/time_series_ma1.csv')
sample_acf = timeseries.autocorr(series)
plt.figure()
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf