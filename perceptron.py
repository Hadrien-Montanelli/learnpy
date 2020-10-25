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
    
    # Change labels 0 and 1 to -1 and 1:
    training_data = np.array(training_data)
    testing_data = np.array(training_data)
    training_data[:,-1] = 2*training_data[:,-1]  - 1
    testing_data[:,-1] = 2*testing_data[:,-1] - 1
    
    # Training:
    iter = 0
    iter_max = 500
    w0 = 0
    w = np.zeros(number_cols-1)
    while iter < iter_max:
        for i in range(number_rows_training):
            label = np.sign(w @ training_data[i,0:number_cols-1] + w0)
            if label != training_data[i,-1]:
                w += training_data[i,-1]*training_data[i,0:number_cols-1]
                w0 += training_data[i,-1]
        iter += 1

    # Testing: 
    error = []
    for i in range(number_rows_testing):
        label = np.sign(w @ testing_data[i,0:number_cols-1] + w0)
        error.append(1/number_rows_testing*float(label != testing_data[i,-1]))
        
    return w0, w, error

    # # Training:
    # kernel = compute_kernel(training_data)
    # iter = 0
    # iter_max = 50
    # alpha = np.zeros([number_rows_training, 1])
    # while iter < iter_max:
    #     for i in range(number_rows_training):
    #         temp = 0
    #         for j in range(number_rows_training):
    #             temp += alpha[j]
    # *np.dot(kernel(training_data[i,:],i),
    #                       kernel(training_data[j,:],j))
    #         if np.sign(temp) != training_data[i,-1]:
    #             alpha[i] += 1
    #     iter += 1
    # print(alpha)

    # # Testing: 
    # error = []
    # for j in range(number_rows_testing):
        
    #     label = 0
    #     for i in range(number_rows_training):
    #         label += alpha[i]*training_data[i,-1]
    # *np.dot(testing_data[j,0:number_cols-1], training_data[i, 0:number_cols-1])
    #     error.append(1/number_rows_testing*float(np.sign(label) 
    #!= testing_data[j,-1]))
        
    # return error
    
# def compute_kernel(training_data): 
#     # TO IMPROVE: Implement the kernel trick.
#     #kernel = lambda x,idx: x[-1]*np.array([1.0 if i == idx else 0.0 for 
#i in range(len(training_data))])
#     kernel = lambda x,idx: x[0:len(training_data[0])-1]
#     return kernel