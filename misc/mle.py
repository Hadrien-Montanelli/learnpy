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

def mle(data, model):
    """ 
    Compute the maximum likelihood estimate of a model using some data.
    
    Inputs
    ------
    data : numpy array
        The data stored as a NxD matrix for N observations in dimension D. 
        
    model : str
        The model for the probability distribution. Only 'normal' is 
        supported.
    
    Output
    ------
    The output is a RandVar (1D), a RandVar2 (2D) or a lambda (D>2). See the 
    documentation for RandVar and RandVar2 for details.
    
    Example
    -------
    This is an example with 2D data.
    
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        output = mle(data, 'normal')
        output.plot()  
        output.display()
      
    See also the 'example_mle' file.
    """
    # Get the number of data and the dimension:
    n = len(data)
    if len(data.shape) == 1:
        dimension = 1
    else:
        dimension = len(data[0])
    
    # To IMPROVE: add more probability models.
    # 1D case:
    if dimension == 1:
        if model == 'normal':
            mean = 1/n*sum(data)
            var = 1/n*sum([(x - mean)**2 for x in data])
            pdf = lambda x: 1/sqrt(2*pi*var)*exp(-1/(2*var)*(x-mean)**2)
            left_bound = min(data) - 3*sqrt(var)
            right_bound = max(data) + 3*sqrt(var)
            domain = np.array([left_bound, right_bound])
            randvar = RandVar(pdf, domain)
        return randvar
    
    # To IMPROVE: add more probability models.
    # Dimension D>1:
    else:
        if model == 'normal':
            mean = 1/n*sum(data)
            covar = np.zeros([dimension, dimension])
            for i in range(n):
                covar += 1/n*np.outer((data[i,:]-mean), np.transpose(data[i,:]-mean))
            covar_inv = np.linalg.inv(covar)
            det = np.linalg.det(covar)
            scl = (2*pi)**(-dimension/2)*det**(-1/2)
            fun = lambda x: np.transpose(x - mean) @ covar_inv @ (x - mean)
            pdf = lambda x: scl*exp(-1/2*fun(x))
            
            # In 2D, return a RANDVAR2:
            if dimension == 2:
                pdf_2d = lambda x,y: pdf(np.array([x, y]))
                left_x_bound = min(data[:,0]) - 3*sqrt(covar[0,0])
                right_x_bound = max(data[:,0]) + 3*sqrt(covar[0,0])
                left_y_bound = min(data[:,1]) - 3*sqrt(covar[1,1])
                right_y_bound = max(data[:,1]) + 3*sqrt(covar[1,1])
                domain = np.array([left_x_bound, right_x_bound, 
                                   left_y_bound, right_y_bound])
                randvar2 = RandVar2(pdf_2d, domain)
                return randvar2
            
            # For D>2, return the probability distribution:
            else:
                return pdf