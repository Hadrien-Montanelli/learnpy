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
import matplotlib.pyplot as plt
from autocorr import autocorr

# AR(1)-type time series:
series = csv_to_array('../dataset/time_series_ar1.csv')
sample_acf = autocorr(series)
plt.plot(sample_acf[1:], '.-')

# MA(1)-type time series:
series = csv_to_array('../dataset/time_series_ma1.csv')
sample_acf = autocorr(series)
plt.figure()
plt.plot(sample_acf[1:], '.-')