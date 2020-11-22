#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:19:21 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def autocorr(series):
    '''Compute the sample autocorrelation function of a series.
    
    Input
    -----
    series : numpy.ndarray
        A time series as a Tx1 array for T data points.
    
    Output
    ------
    output : numpy.ndarray
        The Tx1 array of sample autocorrelation.
        
    Example
    -------
        import numpy as np
        import timeseries as ts
        
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = ts.autocorr(x_t)
        print(output)
        
    See also the 'example_autocorr' file.
    '''
    # Get the number of data points:
    T = len(series)

    # Compute the sample mean and sample variance:
    sample_mean = 1/T*sum(series)
    sample_var = 1/T*sum((series - sample_mean)**2)
    
    # Compute the sample autocorrelation function:
    sample_acf = np.zeros(T)
    sample_acf[0] = 1
    for h in range(1, T):
        sample_acf[h] = 1/T*sum((series[h:T] - sample_mean)
                                * (series[:T-h] - sample_mean))
        sample_acf[h] /= sample_var
        
    return sample_acf