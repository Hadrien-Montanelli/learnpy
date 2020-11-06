#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:19:21 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def autocorr(series):
    '''Compute the sample autocorrelation function.
    
    Input
    -----
    series : numpy array
        A time series stored as a Tx1 vector for T observations.
    
    Output
    ------
    The Tx1 vector of sample autocorrelation.
    
    Example
    -------
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = autocorr(x_t)
        print(output)
        
    See also the 'example_autocorr' file.
    '''
    T = len(series)
    sample_acf = np.zeros(T)
    sample_acf[0] = 1
    sample_mean = 1/T*sum(series)
    sample_var = 1/T*sum((series - sample_mean)**2)
    for h in range(1, T):
        sample_acf[h] = 1/T*sum((series[h:T] - sample_mean)
                                  *(series[:T-h] - sample_mean))
        sample_acf[h] /= sample_var
        
    return sample_acf