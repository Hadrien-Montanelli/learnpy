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
    Use a MLE-based learning algorithm.
    
    The data can be 1D or 2D; models include 'normal'.
    """
    n = len(training_data)
    m = len(testing_data)
    dimension = len(training_data.shape) - 1
    
    if dimension == 1:
        
        # Training:
        training_data_0 = []
        training_data_1 = []
        for k in range(n):
            if training_data[k,1] == 0:
                training_data_0.append(training_data[k,0])
            else:
                training_data_1.append(training_data[k,0])
        training_data_0 = np.array(training_data_0)
        training_data_1 = np.array(training_data_1)
        randvar_0 = mle(training_data_0, model)
        randvar_1 = mle(training_data_1, model)
        
        # Testing:
        error = 0
        for k in range(m):
            proba_0 = randvar_0.pdf(testing_data[k,0])*prior[0]
            proba_1 = randvar_1.pdf(testing_data[k,0])*prior[1]
            if proba_0 > proba_1:
                type = 0
            else:
                type = 1
        if type != training_data[k,1]:
                error += 1
        error = 1/n*error
        return error