#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:19:02 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from numpy import linalg as LA

def pca(data):
    """Return the principal components of data.
    
    Input
    -----
    data : numpy arrray
        The data stored as a NxD matrix for N observations in dimension D.
    
    Outputs
    -------
    The outputs are the matrices of eigenvalues and eigenvectors of the 
    sample covariance matrix.
    
    Example
    -------
    This is an example with 2D data.
    
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        D, V = pca(data)
    
    See also the 'test_pca' file.
    
    """
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