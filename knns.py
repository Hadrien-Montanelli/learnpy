#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:56:55 2020

@author: montanelli
"""
import numpy as np
from numpy import linalg as LA

def knns(training_data, testing_data, k):
    """
    Use the k-nearest neighbours algorithm to classify data.
    
    The first columns of the training and testing data represent the data as 
    numpy arrays, while the last colum represents the label (0 or 1). The data
    can be 1D or 2D.
    """
    # TO IMPROVE: the following implements k=1, extend to k>2.
    if k != 1:
        raise ValueError("The knns algorithm only supports k=1 for now.")
        
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_rows_training = len(training_data)
    number_cols = len(testing_data[0])
    
    # Separate labels 0 and 1:
    training_data_0 = training_data[training_data[:,-1]==0][:,0:number_cols-1]
    training_data_1 = training_data[training_data[:,-1]==1][:,0:number_cols-1]
    
    # Training:
    dist = compute_distance(training_data_0, training_data_1)
    
    # Testing: 
    error = []
    for j in range(number_rows_testing):
        
        dist_to_training = np.zeros(number_rows_training)
        
        for i in range(number_rows_training):
            x_j = testing_data[j,0:number_cols-1]
            x_i = training_data[i,0:number_cols-1]
            dist_to_training[i] = dist(x_j, x_i)
            
        pos_min = np.argmin(dist_to_training)
        label = testing_data[int(pos_min), -1]
        error.append(1/number_rows_testing*float(label != testing_data[j,-1]))
        
    return error
    
def compute_distance(training_data_0, training_data_1): 
    # TO IMPROVE: use BFGS to find the distance that best separates the data.
    dist = lambda x,y: LA.norm(x-y)
    return dist