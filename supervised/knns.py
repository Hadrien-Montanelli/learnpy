#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:56:55 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from numpy import linalg as LA

def knns(training_data, testing_data, k):
    """Use the k-nearest neighbours algorithm for binary classification.
    
    Inputs
    ------
    training_data : numpy array
        The training data stored as a Nx(D+1) matrix for N observations in
        dimension D. The last column represents the label (0 or 1).
        
    testing_data : numpy_array
        The testing data stored as a Nx(D+1) matrix for N observations in 
        dimension D. The last column represents the label (0 or 1).
        
    k : int
        The number of neighbours. Only k = 1 is supported.
    
    Output
    ------
    The output is the error on the testing data.

    Example
    -------
    This is an example with 2D data.

        training_data = np.array([[160, 60, 0], [172, 90, 1], [180, 90, 1]])
        testing_data = np.array([[165, 66, 0], [176, 86, 1], [189, 99, 1]])
        output = knns(training_data, testing_data, 1)
        print(output)
      
    See also the 'example_knns' file.
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
        
    return sum(error)
    
def compute_distance(training_data_0, training_data_1): 
    # TO IMPROVE: use BFGS to find the distance that best separates the data.
    dist = lambda x,y: LA.norm(x-y)
    return dist