#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:45:47 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def kperceptron(training_data, testing_data, kernel = lambda x,y: 1 + x @ y):
    """
    Use the kernelized perceptron algorithm for binary classification.
    
    Inputs
    ------
    training_data : numpy array
        The training data stored as a Nx(D+1) matrix for N observations in
        dimension D. The last column represents the label (0 or 1).
        
    testing_data : numpy_array
        The testing data stored as a Nx(D+1) matrix for N observations in 
        dimension D. The last column represents the label (0 or 1).
        
    kernel : lambda
        The kernel.
    
    Outputs
    -------
    The first output is the bias while the second output is the rest of the
    weights. The third output is the error on the testing data.
      
    See the 'example_kperceptron' file.
    """
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_rows_training = len(training_data)
    number_cols = len(testing_data[0])
    
    # Change labels {0,1} to {-1,1}:
    training_data = np.array(training_data)
    testing_data = np.array(testing_data)
    training_data[:,-1] = 2*training_data[:,-1] - 1
    testing_data[:,-1] = 2*testing_data[:,-1] - 1

    # Training:
    test = 0
    alpha = np.zeros(number_rows_training)
    while test == 0:
        test = 1
        for i in range(number_rows_training):
            label = 0
            x_i = training_data[i,0:number_cols-1]
            y_i = training_data[i,-1]
            for j in range(number_rows_training):
                x_j = training_data[j,0:number_cols-1]
                y_j = training_data[j,-1]
                label += alpha[j]*y_j*(kernel(x_j, x_i))
            label = np.sign(label)
            if label != y_i:
                alpha[i] += 1
                test = 0
    
    # Testing: 
    error = []
    for i in range(number_rows_testing):
        label = 0
        x_i = testing_data[i,0:number_cols-1]
        y_i = testing_data[i, -1]
        for j in range(number_rows_training):
            x_j = training_data[j,0:number_cols-1]
            y_j = training_data[j,-1]
            label += alpha[j]*y_j*(kernel(x_j, x_i))
        label = np.sign(label)
        error.append(1/number_rows_testing*float(label != y_i))
        
    # Get the weights:
    w_0 = 0
    w = np.zeros(number_cols-1)
    for i in range(number_rows_training):
        x_i = training_data[i,0:number_cols-1]
        y_i = training_data[i,-1]
        w += alpha[i]*y_i*x_i
        w_0 += alpha[i]*y_i
        
    return w_0, w, error