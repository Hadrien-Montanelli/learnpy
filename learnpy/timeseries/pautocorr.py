#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:00:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

# Learnpy imports:
from .autocorr import autocorr

def pautocorr(series):
    '''Compute the sample partial autocorrelation function of a series.
    
    Input
    -----
    series : numpy.ndarray
        A time series as a Tx1 array for T data points.
    
    Output
    ------
    output : numpy.ndarray
        The Tx1 array of sample partial autocorrelation.
        
    Example
    -------
        import numpy as np
        from learnpy.timeseries import pautocorr
        
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = pautocorr(x_t)
        print(output)
        
    See also the 'example_pautocorr' file.
    '''
    # Get the number of data points:
    T = len(series)

    # Compute the sample autocorrelation function:
    sample_acf = autocorr(series)
    
    # Compute the partial sample autocorrelation function:
    sample_pacf = np.zeros(T)
    sample_pacf[0] = 1
    for h in range(1, T):
        gamma = np.zeros([h, h])
        for i in range(h):
            for j in range(h):
                gamma[i, j] = sample_acf[i-j]
        sample_pacf[h] = (np.linalg.inv(gamma) @ sample_acf[1:h+1])[-1]
        
    return sample_pacf