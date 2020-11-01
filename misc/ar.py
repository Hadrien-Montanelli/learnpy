#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:27:18 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from regression import regression

def ar(data, p):
    '''Autoregression AR(p) of data.
    
    Inputs
    ------
    data : numpy array
        
    p : int
    
    Outputs
    -------
    
    Example
    -------

    See also the test_ar file.
    '''
    
    if p == 1:
        # TO IMPROVE: implement p=1 case.
        raise ValueError("The ar algorithm doesn't support p=1 for now.")
        
    else:
        # Assemble the autoregression matrix:
        n = len(data)
        y = np.zeros(n-1)
        y = data[p:]
        x = np.zeros([n-1, p])
        for k in range(n-1):
            for i in range(p):
                if i <= k:
                    x[k, i] = data[k-i]
        x = x[p-1:, :]

        # Call the regression method:
        phi_0, phi = regression(x, y, 'linear')

    return phi_0, phi