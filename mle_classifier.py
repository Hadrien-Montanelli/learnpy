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
    Use a Maximum Likelihood Estimator-based learning algorithm.
    
    The first column of the training and testing data represents the data, 
    while the second colum represents the label (0 or 1).
    """
    testing_data_len = len(testing_data)
    dimension = len(training_data.shape) - 1
    
    # One-dimensional case:
    if dimension == 1:
        
        # Separate labels 0 and 1:
        training_data = np.array(training_data)
        training_data_0 = training_data[training_data[:,1]==0][:,0]
        training_data_1 = training_data[training_data[:,1]==1][:,0]
        
        # Use the MLE for training:
        randvar_0 = mle(training_data_0, model)
        randvar_1 = mle(training_data_1, model)
        
        # Testing:
        error = []
        proba_0 = []
        proba_1 = []
        for k in range(testing_data_len):
            p_0 = randvar_0.pdf(testing_data[k,0])*prior[0]
            p_1 = randvar_1.pdf(testing_data[k,0])*prior[1]
            label = float(p_0 < p_1)
            proba_0.append(p_0)
            proba_1.append(p_1)
            error.append(1/testing_data_len*float(label != testing_data[k,1]))
                
        # Outputs:
        return randvar_0, randvar_1, proba_0, proba_1, error