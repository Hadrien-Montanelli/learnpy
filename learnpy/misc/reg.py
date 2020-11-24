#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:36:47 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def reg(X, Y, model):
    '''Regression of Y on X.
    
    Inputs
    ------
    X : numpy.ndarray
        The independent variables as a nxd array for n data points in 
        dimension d.
        
    Y : numpy.ndarray
        The dependent variable as a nx1 array.
        
    model : str
        The model for regression. Only 'linear' is supported.
    
    Outputs
    -------
    output[0] : float
        The bias.
        
    output[1] : numpy.ndarray
        The rest of the model parameters as a dx1 array.
    
    Example
    -------
    This is an example in 1d.
    
        import numpy as np
        from learnpy.misc import regression
    
        x = np.linspace(-1, 1, 100)
        y = 2*x + 6 + 5e-1*np.random.randn(100)
        output = regression(x, y, 'linear')
        print(output)
    
    See also the 'example_regression' file.
    '''
    # Get the number of data points n and the dimension d:
    n = len(X)
    if (len(X.shape) == 1):
        d = 1
    else:
        d = len(X[0])
        
    # TO IMPROVE: add logistic regression.
    if (model == 'linear'):
        
        # One-dimensional case:
        if (d == 1):
            X_bar = 1/n*sum(X)
            Y_bar = 1/n*sum(Y)
            beta = sum((X - X_bar) * Y)/sum((X - X_bar)**2)
            alpha = Y_bar - beta*X_bar
            return alpha, beta
            
        # Higher dimensions:
        else:
            
            # Add a column of 1's for the bias:
            Z = np.zeros([n, d+1])
            for i in range(n):
                Z[i, 0] = 1 # bias
                Z[i, 1:] = X[i, :]
            
            # Least squares:
            beta = np.linalg.inv(Z.T @ Z) @ Z.T @ Y
            
            return beta[0], beta[1:]