#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:36:47 2020

@author: montanelli
"""
import numpy as np

def regression(x, y, model):
    '''Regression of y on x. Models include 'normal'.
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
            return [alpha, beta]
            
        # Higher dimensions.
        else:
            z = np.zeros([n, dimension + 1])
            for i in range(n):
                z[i, 0] = 1 # bias
                z[i, 1] = x[i, 0]
                z[i, 2] = x[i, 1]
            beta = np.linalg.inv(np.transpose(z) @ z) @ np.transpose(z) @ y
            return beta