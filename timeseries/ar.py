#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:27:18 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def ar(series, p):
    '''Autoregression AR(p) of data.
    
    Inputs
    ------
    series : numpy array
        A time series stored as a Tx1 vector for T observations.
        
    p : int
        Parameter in the AR(p) model.
    
    Outputs
    -------
    The first output is the bias while the second output is the rest of the 
    model parameters.
    
    Example
    -------
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = ar(x_t, 2)
        print(output)

    See also the 'example_ar' file.
    '''
    # Assemble the autoregression matrix:
    T = len(series)
    y = series[p:]
    x = np.zeros([T-1, p+1])
    for t in range(T-1):
        x[t, 0] = 1
        for i in range(1, p+1):
            if i - 1 <= t:
                x[t, i] = series[t-i+1]
    x = x[p-1:, :]

    # Least squares:
    phi = np.linalg.inv(x.T @ x) @ x.T @ y
 
    return phi[0], phi[1:]