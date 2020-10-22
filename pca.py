#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:19:02 2020

@author: montanelli
"""
import numpy as np
from numpy import linalg as LA

def pca(data):
    """Return the principal components of data."""
    # Get dimensions:
    number_rows = len(data)
    number_cols = len(data[0])
    
    # Compute the sample mean and centre the data:
    sample_mean = 1/number_rows*np.sum(data, 0)
    data = np.array([data[i,:] - sample_mean for i in range(number_rows)])
    
    # Eigenvalue decomposition:
    if number_rows < number_cols:
        D, V = LA.eig(1/(number_rows-1)*(data @ np.transpose(data)))
    else:
        D, V = LA.eig(1/(number_rows-1)*(np.transpose(data) @ data))
        
    return D, V