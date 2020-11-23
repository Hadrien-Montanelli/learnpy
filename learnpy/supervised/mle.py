#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:57:38 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
from math import exp, sqrt, pi
import numpy as np

# Learnpy imports:
from .RandVar import RandVar
from .RandVar2 import RandVar2

def mle(X, model):
    """ 
    Compute the maximum likelihood estimate.
    
    Inputs
    ------
    X : numpy.ndarray
        The data as a nxd matrix for n data points in dimension d.
        
    model : str
        The model for the probability distribution. Only 'normal' is
        supported.
    
    Output
    ------
    output : RandVar (1d), RandVar2 (2d) or a function (d>2).
    
    Example
    -------
    This is an example with 2d data.
    
        import numpy as np
        from learnpy.supervised import mle
    
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        output = mle(data, 'normal')
        output.plot()  
        output.display()
      
    See also the 'example_mle' file.
    """
    # Get the number of data points and the dimension:
    n = len(X)
    if len(X.shape) == 1:
        d = 1
    else:
        d = len(X[0])
    
    # To IMPROVE: add more probability models.
    # 1d case:
    if (d == 1):
        if (model == 'normal'):
            mean = 1/n*sum(X)
            var = 1/n*sum([(x - mean)**2 for x in X])
            pdf = lambda x: 1/sqrt(2*pi*var)*exp(-1/(2*var)*(x-mean)**2)
            left_bound = min(X) - 3*sqrt(var)
            right_bound = max(X) + 3*sqrt(var)
            domain = np.array([left_bound, right_bound])
            randvar = RandVar(pdf, domain)
        return randvar
    
    # To IMPROVE: add more probability models.
    # Dimension d>1:
    else:
        if (model == 'normal'):
            mean = 1/n*sum(X)
            covar = np.zeros([d, d])
            for i in range(n):
                covar += 1/n*np.outer((X[i,:]-mean), (X[i,:]-mean).T)
            covar_inv = np.linalg.inv(covar)
            det = np.linalg.det(covar)
            scl = (2*pi)**(-d/2)*det**(-1/2)
            fun = lambda x: (x - mean).T @ covar_inv @ (x - mean)
            pdf = lambda x: scl*exp(-1/2*fun(x))
            
            # In 2d, return a RANDVAR2:
            if (d == 2):
                pdf_2d = lambda x,y: pdf(np.array([x, y]))
                left_x_bound = min(X[:,0]) - 3*sqrt(covar[0,0])
                right_x_bound = max(X[:,0]) + 3*sqrt(covar[0,0])
                left_y_bound = min(X[:,1]) - 3*sqrt(covar[1,1])
                right_y_bound = max(X[:,1]) + 3*sqrt(covar[1,1])
                domain = np.array([left_x_bound, right_x_bound, 
                                   left_y_bound, right_y_bound])
                randvar2 = RandVar2(pdf_2d, domain)
                return randvar2
            
            # For d>2, return the probability distribution:
            else:
                return pdf