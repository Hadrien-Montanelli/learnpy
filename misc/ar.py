#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:27:18 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def ar(data, p):
    '''Autoregression AR(p) of data.
    
    Inputs
    ------
    data : numpy array
        A time series as a Nx1 vector for N observations.
        
    p : int
        Parameter in the AR(p) model.
    
    Outputs
    -------
    The first output is the bias while the second output is the rest of the 
    model parameters.
    
    Example
    -------
        x = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)
        output = ar(x, 2)
        print(output)

    See also the test_ar file.
    '''
    # Assemble the autoregression matrix:
    n = len(data)
    y = data[p:]
    x = np.zeros([n-1, p+1])
    for k in range(n-1):
        x[k, 0] = 1
        for i in range(1, p+1):
            if i - 1 <= k:
                x[k, i] = data[k-i+1]
    x = x[p-1:, :]

    # Least squares:
    phi = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
 
    return phi[0], phi[1:]