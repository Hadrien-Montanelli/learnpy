#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:59:21 2020

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
from statsmodels.tsa.stattools import pacf

# Learnpy imports:
import misc
import timeseries as ts

# %% Examples.

# AR(1)-type time series:
series = misc.csv_to_array('../../dataset/time_series_ar1.csv')
sample_pacf = ts.pautocorr(series)
plt.plot(sample_pacf[1:], '-')
sample_pacf_statsmodels = pacf(series, nlags=len(series)-1, method='ywm')
plt.plot(sample_pacf_statsmodels[1:], '--')
error = np.linalg.norm(sample_pacf - sample_pacf_statsmodels)
print('Error: ', error) # compare with statsmodels' pacf

# MA(1)-type time series:
series = misc.csv_to_array('../../dataset/time_series_ma1.csv')
sample_pacf = ts.pautocorr(series)
plt.figure()
plt.plot(sample_pacf[1:], '-')
sample_pacf_statsmodels = pacf(series, nlags=len(series)-1, method='ywm')
plt.plot(sample_pacf_statsmodels[1:], '--')
error = np.linalg.norm(sample_pacf - sample_pacf_statsmodels)
print('Error: ', error) # compare with statsmodels' pacf