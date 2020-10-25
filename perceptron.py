#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:36:48 2020

@author: montanelli
"""
import numpy as np

def perceptron(training_data, testing_data):
    """
    Use the perceptron algorithm to classify data.
    
    The first columns of the training and testing data represent the data as 
    numpy arrays, while the last colum represents the label (0 or 1).
    """
    # Get dimensions:
    number_rows_testing = len(testing_data)
    number_rows_training = len(training_data)
    number_cols = len(testing_data[0])
    
    # Change labels {0,1} to {-1,1}:
    training_data = np.array(training_data)
    testing_data = np.array(training_data)
    training_data[:,-1] = 2*training_data[:,-1]  - 1
    testing_data[:,-1] = 2*testing_data[:,-1] - 1

    # Training:
    iter = 0
    iter_max = 100
    kernel = compute_kernel(training_data)
    alpha = np.zeros(number_rows_training)
    while iter < iter_max:
        for i in range(number_rows_training):
            label = 0
            x_i = training_data[i,:]
            for j in range(number_rows_training):
                x_j = training_data[j,:] 
                y_j = training_data[j,-1] 
                label += alpha[j]*y_j*(kernel(x_i,i) @ kernel(x_j,j) + 1)
            label = np.sign(label)
            if label != training_data[i,-1]:
                alpha[i] += 1
        iter += 1

    # Testing: 
    error = np.zeros(number_rows_testing)
    for i in range(number_rows_testing):
        label = 0
        x_i = testing_data[i,0:number_cols-1] 
        for j in range(number_rows_training):
            x_j = training_data[j,0:number_cols-1] 
            y_j = training_data[j,-1]
            label += alpha[j]*y_j*(x_i @ x_j + 1)
        label = np.sign(label)
        if label != testing_data[i,-1]:
            error[i] = 1/number_rows_testing
        
    return alpha, error
    
def compute_kernel(data): 
    # TO IMPROVE: Implement the kernel trick.
    kernel = lambda x,idx: x[0:len(data)-1]
    return kernel