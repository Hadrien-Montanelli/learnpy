#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:36:47 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def regression(x, y, model):
    '''Regression of y on x.
    
    Inputs
    ------
    x : numpy array
        The dependent variables stored as a NxD matrix for N observations in
        dimension D.
        
    y : numpy array
        The independent variable stored as Nx1 vector.
        
    model : str
        The model for regression. Only 'linear' is supported.
    
    Outputs
    -------
    The first output is the bias while the second output is the rest of the 
    model parameters.
    
    Example
    -------
    This is an example in 1D.
    
        x = np.linspace(-1, 1, 100)
        y = 2*x + 6 + 5e-1*np.random.randn(100)
        output = regression(x, y, 'linear')
        print(output)
    
    See also the test_regression file.
    '''
    # Get the number of data and the dimension:
    n = len(x)
    if len(x.shape) == 1:
        dimension = 1
    else:
        dimension = len(x[0])
        
    # TO IMPROVE: add logistic regression.
    if model == 'linear':
        
        # One-dimensional case.
        if dimension == 1:
            x_bar = 1/n*sum(x)
            y_bar = 1/n*sum(y)
            beta = sum((x - x_bar)*y)/sum((x - x_bar)**2)
            alpha = y_bar - beta*x_bar
            return alpha, beta
            
        # Higher dimensions.
        else:
            z = np.zeros([n, dimension + 1])
            for i in range(n):
                z[i, 0] = 1 # bias
                z[i, 1:] = x[i, :]
            beta = np.linalg.inv(np.transpose(z) @ z) @ np.transpose(z) @ y
            return beta[0], beta[1:]