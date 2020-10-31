#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:38:07 2020

@author: montanelli
"""
import numpy as np
from mle import mle

def mle_classifier(training_data, testing_data, prior, model):
    """ 
    Use maximum likelihood estimation for binary classification.
    
    The first columns of the training and testing data represent the data as 
    numpy arrays, while the last colum represents the label (0 or 1). The data
    can be 1D or 2D. Models include 'normal'.
    
    For example, with 2D data:
        
        import numpy as np
        from mle_classifier import mle_classifier
        
        training_data = np.array([[165,0],[174,1],[155,0],[184,1]])
        testing_data = np.array([[162,0],[170,1],[160,0],[182,1]])
        prior = np.array([0.5, 0.5])
        mle_classifier(training_data, testing_data, prior, 'normal')
    """
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_cols = len(testing_data[0])
    dimension = number_cols - 1
    
    # Separate labels 0 and 1:
    training_data_0 = training_data[training_data[:,-1]==0][:,0:dimension]
    training_data_1 = training_data[training_data[:,-1]==1][:,0:dimension]
    
    # Use the MLE for training:
    randvar_0 = mle(training_data_0, model)
    randvar_1 = mle(training_data_1, model)

    # Testing:
    error = []
    proba_0 = []
    proba_1 = []
    for k in range(number_rows_testing):
        
        # To IMPROVE: implement algorithm for higher dimensions.
        # One-dimensional case:
        if dimension == 1:
            p_0 = randvar_0.pdf(testing_data[k,0])*prior[0]
            p_1 = randvar_1.pdf(testing_data[k,0])*prior[1]
            
        # Two-dimensional case:
        if dimension == 2:
            p_0 = randvar_0.pdf(testing_data[k,0],testing_data[k,1])*prior[0]
            p_1 = randvar_1.pdf(testing_data[k,0],testing_data[k,1])*prior[1]
            
        label = float(p_0 < p_1)
        proba_0.append(p_0)
        proba_1.append(p_1)
        error.append(1/number_rows_testing*float(label != testing_data[k,-1]))
            
    # Outputs:
    return randvar_0, randvar_1, proba_0, proba_1, error