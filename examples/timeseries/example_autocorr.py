#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:17:42 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

# Learnpy imports:
from learnpy.misc import csv_to_array
from learnpy.timeseries import autocorr

# %% Examples.

# AR(1)-type time series:
series = csv_to_array('../../dataset/time_series_ar1.csv')
sample_acf = autocorr(series)
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf

# MA(1)-type time series:
series = csv_to_array('../../dataset/time_series_ma1.csv')
sample_acf = autocorr(series)
plt.figure()
plt.plot(sample_acf[1:], '-')
sample_acf_statsmodels = acf(series, nlags=len(series), fft=False)
plt.plot(sample_acf_statsmodels[1:], '--')
error = np.linalg.norm(sample_acf - sample_acf_statsmodels)
print('Error: ', error) # compare with statsmodels' acf