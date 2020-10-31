#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:38:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
import sys
sys.path.append('../misc')
from compute_mle import compute_mle

def mle(training_data, testing_data, prior, model):
    """
    Use the maximum likelihood estimation algorithm for binary classification.
    
    Inputs
    ------
    training_data : numpy array
        The training data stored as a Nx(D+1) matrix for N observations in
        dimension D. Only 1D and 2D data are supported. The last column 
        represents the label (0 or 1).
        
    testing_data : numpy_array
        The testing data stored as a Nx(D+1) matrix for N observations in 
        dimension D. Only 1D and 2D data are supported. The last column 
        represents the label (0 or 1).
        
    prior : numpy array
        The prior probablilites stored as a 2x1 vector.
        
    model : str
        The model for the probability distribution. Only 'normal' is 
        supported.

    Outputs
    -------
    The first two outputs are a RandVar (1D) or RandVar2 (2D) that correspond 
    to the probability models for the data labelled with 0 and 1. See the 
    documentation for RandVar and RandVar2 for details. The third output is 
    the error on the testing data.
    
    Example
    -------
    This is an example with 1D data.

        training_data = np.array([[160, 0], [155, 0], [172, 1], [180, 1]])
        testing_data = np.array([[165, 0], [162, 0], [176, 1], [189, 1]])
        prior = np.array([1/2, 1/2])
        output = mle(training_data, testing_data, prior, 'normal')
        output[0].plot()
        output[1].plot()
        print(output[2])
      
    See also the 'test_mle' file.
    """
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_cols = len(testing_data[0])
    dimension = number_cols - 1
    
    # Separate labels 0 and 1:
    training_data_0 = training_data[training_data[:,-1]==0][:,0:dimension]
    training_data_1 = training_data[training_data[:,-1]==1][:,0:dimension]
    
    # Compute the MLE for training:
    randvar_0 = compute_mle(training_data_0, model)
    randvar_1 = compute_mle(training_data_1, model)

    # Testing:
    error = []
    for k in range(number_rows_testing):
        
        # One-dimensional case:
        if dimension == 1:
            p_0 = randvar_0.pdf(testing_data[k,0])*prior[0]
            p_1 = randvar_1.pdf(testing_data[k,0])*prior[1]
            
        # Two-dimensional case:
        if dimension == 2:
            p_0 = randvar_0.pdf(testing_data[k,0],testing_data[k,1])*prior[0]
            p_1 = randvar_1.pdf(testing_data[k,0],testing_data[k,1])*prior[1]
            
        label = float(p_0 < p_1)
        error.append(1/number_rows_testing*float(label != testing_data[k,-1]))
            
    # Outputs:
    return randvar_0, randvar_1, sum(error)