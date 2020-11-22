#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:19:02 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from numpy import linalg as LA

def pca(X):
    """Return the principal components of X.
    
    Input
    -----
    X : numpy.ndarrray
        The data as a nxd array for n data points in dimension d.
    
    Outputs
    -------
    output[0] : numpy.ndarrray
        The eigenvalues of the sample covariance matrix as a dx1 array.  
    
    output[1] : numpy.ndarrray
        THe eigenvectors of the sample covariance matrix as a dxd array. 
    
    Example
    -------
    This is an example with 2d data.
    
        import numpy as np
        from learnpy.misc import pca
        
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        D, V = pca(data)
    
    See also the 'example_pca' file.
    
    """
    # Get the number of data points n and the dimension d:
    n = len(X)
    d = len(X[0])
    
    # Compute the sample mean and center the data:
    sample_mean = 1/n*np.sum(X, 0)
    X = np.array([X[i,:] - sample_mean for i in range(n)])
    
    # Eigenvalue decomposition of the sample covariance matrix:
    if (n < d):
        D, V = LA.eig(1/(n-1)*(X @ X.T))
    else:
        D, V = LA.eig(1/(n-1)*(X.T @ X))
        
    return D, V