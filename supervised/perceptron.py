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