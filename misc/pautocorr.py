#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:00:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from autocorr import autocorr

def pautocorr(series):
    '''Compute the sample partial autocorrelation function.
    
    Input
    -----
    series : numpy array
        A time series stored as a Tx1 vector for T observations.
    
    Output
    ------
    The Tx1 vector of sample partial autocorrelation.
    
    Example
    -------
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = pautocorr(x_t)
        print(output)
        
    See also the 'test_pautocorr' file.
    '''
    T = len(series)
    sample_pacf = np.zeros(T)
    sample_pacf[0] = 1
    sample_acf = autocorr(series)
    sample_mean = 1/T*sum(series)
    sample_var = 1/T*sum((series - sample_mean)**2)
    for h in range(1, T):
        gamma = np.zeros([h, h])
        for i in range(h):
            for j in range(h):
                gamma[i,j] = sample_var*sample_acf[i-j]
        sample_pacf[h] = (np.linalg.inv(gamma) @ sample_acf[1:h+1])[-1]
        sample_pacf[h] *= sample_var
        
    return sample_pacf