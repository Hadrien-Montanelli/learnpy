#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:36:48 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def perceptron(training_data, testing_data):
    """
    Use the perceptron algorithm for binary classification.
    
    Inputs
    ------
    training_data : numpy array
        The training data stored as a Nx(D+1) matrix for N observations in
        dimension D. The last column represents the label (0 or 1).
        
    testing_data : numpy_array
        The testing data stored as a Nx(D+1) matrix for N observations in 
        dimension D. The last column represents the label (0 or 1).
    
    Outputs
    -------
    The first output is the bias while the second output is the rest of the
    weights. The third output is the error on the testing data.

    Example
    -------
    This is an example with 2D data.

        training_data = np.array([[160, 60, 0], [172, 90, 1], [180, 90, 1]])
        testing_data = np.array([[165, 66, 0], [176, 86, 1], [189, 99, 1]])
        output = perceptron(training_data, testing_data)
        print(output)
      
    See also the 'example_perceptron' file.
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
    w0 = 0
    w = np.zeros(number_cols-1)
    while test == 0:
        test = 1
        for i in range(number_rows_training):
            label = np.sign(w @ training_data[i,0:number_cols-1] + w0)
            if label != training_data[i,-1]:
                w += training_data[i,-1]*training_data[i,0:number_cols-1]
                w0 += training_data[i,-1]
                test = 0
    
    # Testing: 
    error = []
    for i in range(number_rows_testing):
        label = np.sign(w @ testing_data[i,0:number_cols-1] + w0)
        error.append(1/number_rows_testing*float(label != testing_data[i,-1]))
        
    return w0, w, error