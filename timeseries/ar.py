#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:27:18 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def ar(series, p):
    '''Autoregression AR(p) of a series.
    
    Inputs
    ------
    series : numpy.ndarray
        A time series as a Tx1 array for T data points.
        
    p : int
        Parameter in the AR(p) model.
    
    Outputs
    -------
    output[0] : float
        The bias.    
    
    output[1] : numpy.ndarray
        The rest of the model parameters as a px1 array.
        
    Example
    -------
        import numpy as np
        import timeseries as ts
        
        x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = ts.ar(x_t, 2)
        print(output)
        
    See also the 'example_ar' file.
    '''
    # Get the number of data points:
    T = len(series)
    
    # Assemble the RHS for least squares:
    Y = series[p:]
    
    # Assemble the autoregression matrix for least squares:
    X = np.zeros([T-1, p+1])
    for t in range(T-1):
        X[t, 0] = 1
        for i in range(1, p+1):
            if i - 1 <= t:
                X[t, i] = series[t-i+1]
    X = X[p-1:, :]

    # Least squares:
    phi = np.linalg.inv(X.T @ X) @ X.T @ Y
 
    return phi[0], phi[1:]