#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:57:38 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from math import exp, sqrt, pi
from RandVar import RandVar
from RandVar2 import RandVar2

def compute_mle(data, model):
    """ 
    Compute the maximum likelihood estimate of a model using some data.
    
    Inputs
    ------
    data : numpy array
        The data stored as a NxD matrix for N observations in dimension D. 
        Only 1D and 2D data are supported.
        
    model : str
        The model for the probability distribution. Only 'normal' is 
        supported.
    
    Output
    ------
    The output is a RandVar (1D) or RandVar2 (2D). See the documentation for
    RandVar and RandVar2 for details.
    
    Example
    -------
    This is an example with 2D data.
    
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        output = compute_mle(data, 'normal')
        output.plot()  
        output.display()
      
    See also the 'test_compute_mle' file.
    """
    # Get the number of data and the dimension:
    n = len(data)
    if len(data.shape) == 1:
        dimension = 1
    else:
        dimension = len(data[0])
    
    # To IMPROVE (1): implement the algorithm for higher dimensions.
    # To IMPROVE (2): add more probability models.
    # One-dimensional case:
    if dimension == 1:
        if model == 'normal':
            mean = 1/n*sum(data)
            var = 1/(n-1)*sum([(x - mean)**2 for x in data])
            pdf = lambda x: 1/sqrt(2*pi*var)*exp(-1/(2*var)*(x-mean)**2)
            left_bound = min(data) - 3*sqrt(var)
            right_bound = max(data) + 3*sqrt(var)
            domain = np.array([left_bound, right_bound])
            randvar = RandVar(pdf, domain)
        return randvar
    
    # Two-dimensional case:
    if dimension == 2:
        if model == 'normal':
            mean_x = 1/n*sum(data)[0]
            mean_y = 1/n*sum(data)[1]
            mean = np.array([mean_x, mean_y])
            covar_xx = 1/(n-1)*sum([(x - mean_x)**2 for x in data[:,0]])
            covar_yy = 1/(n-1)*sum([(x - mean_y)**2 for x in data[:,1]])
            covar_xy = 1/(n-1)*sum((data[:,0]-mean_x)*(data[:,1]-mean_y))
            covar = np.array([[covar_xx, covar_xy], [covar_xy, covar_yy]])
            covar_inv = np.linalg.inv(covar)
            determinant = np.linalg.det(covar)
            scl = (2*pi)**(-1)*determinant**(-1/2)
            a, b = covar_inv[0,0:2]
            c, d = covar_inv[1,0:2]
            fun = lambda x,y: exp(-1/2*((x-mean_x)*(a*(x-mean_x)+b*(y-mean_y)) 
                                   + (y-mean_y)*(c*(x-mean_x)+d*(y-mean_y))))
            pdf = lambda x,y: scl*fun(x,y)
            left_x_bound = min(data[:,0]) - 3*sqrt(covar_xx)
            right_x_bound = max(data[:,0]) + 3*sqrt(covar_xx)
            left_y_bound = min(data[:,1]) - 3*sqrt(covar_yy)
            right_y_bound = max(data[:,1]) + 3*sqrt(covar_yy)
            domain = np.array([left_x_bound, right_x_bound, 
                               left_y_bound, right_y_bound])
            randvar2 = RandVar2(pdf, domain)
        return randvar2