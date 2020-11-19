#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:38:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
import sys
sys.path.append('../misc')
from mle import mle

def bayes(training_data, testing_data, prior, model):
    """
    Use a naive Bayes classifier for binary classification.
    
    Inputs
    ------
    training_data : numpy array
        The training data stored as a Nx(D+1) matrix for N observations in
        dimension D. The last column represents the label (0 or 1).
        
    testing_data : numpy_array
        The testing data stored as a Nx(D+1) matrix for N observations in 
        dimension D. The last column represents the label (0 or 1).
        
    prior : numpy array
        The prior probablilites stored as a 2x1 vector.
        
    model : str
        The model for the probability distribution. Only 'normal' is 
        supported.

    Outputs
    -------
    The first two outputs are a RandVar that corresponds to the probability 
    models for the data labelled with 0 and 1. See the documentation for 
    RandVar for details. The third output is the error on the testing data.
    
    Example
    -------
    This is an example with 1D data.

        training_data = np.array([[160, 0], [155, 0], [172, 1], [180, 1]])
        testing_data = np.array([[165, 0], [162, 0], [176, 1], [189, 1]])
        prior = np.array([1/2, 1/2])
        output = bayes(training_data, testing_data, prior, 'normal')
        output[0].plot()
        output[1].plot()
        print(output[2])
      
    See also the 'example_bayes' file.
    """
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_cols = len(testing_data[0])
    dimension = number_cols - 1
    
    # Separate labels 0 and 1:
    training_data_0 = training_data[training_data[:,-1]==0][:,0:dimension]
    training_data_1 = training_data[training_data[:,-1]==1][:,0:dimension]
    
    # Compute the 1D MLEs for each feature and each label:
    randvar = []
    for j in range(dimension):
        randvar_0 = mle(training_data_0[:,j], model)
        randvar_1 = mle(training_data_1[:,j], model)
        randvar.append([randvar_0, randvar_1])
    
    # Testing:
    error = []
    for k in range(number_rows_testing):
        p_0 = prior[0]
        p_1 = prior[1]
        for j in range(dimension):
            p_0 *= randvar[j][0].pdf(testing_data[k,j])
            p_1 *= randvar[j][1].pdf(testing_data[k,j])  
        label = float(p_0 < p_1)
        error.append(1/number_rows_testing*float(label != testing_data[k,-1]))
            
    # Outputs:
    return randvar, sum(error)