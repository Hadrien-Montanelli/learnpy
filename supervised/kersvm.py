#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:11:15 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

def kersvm(training_data, testing_data, kernel):
    """
    Use the kernel upport vector machines algorithm for binary classification.
    
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
    coefficients. The third output is the error on the testing data.
      
    See the 'example_kersvm' file.
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
    eq_cons = {'type': 'eq',
                 'fun' : lambda  x: sum(x*training_data[:,-1]),
                 'jac' : lambda x: training_data[:,-1]}
    lower_bounds = [0 for _ in range(number_rows_training)]
    upper_bounds = [np.inf for _ in range(number_rows_training)]
    bounds = Bounds(lower_bounds, upper_bounds)
    alpha = np.zeros(number_rows_testing)
    res = minimize(func, alpha, args = (training_data, kernel), 
                   method = 'SLSQP', jac = func_grad, bounds = bounds,
                   constraints = eq_cons, 
                   options = {'ftol': 1e-8, 'disp': False})
    alpha = res.x
    
    # Get bias:
    idx = np.argmax(alpha)
    x_idx = training_data[idx,0:number_cols-1]
    y_idx = training_data[idx,-1]
    w0 = y_idx
    for i in range(number_rows_training):
        x_i = training_data[i,0:number_cols-1]
        y_i = training_data[i,-1]
        w0 -= alpha[i]*y_i*kernel(x_i,x_idx)
        
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
        label = np.sign(label + w0)
        error.append(1/number_rows_testing*float(label != y_i))
        
    return w0, alpha, error

def func(alpha, data, kernel):
    num_rows = len(data)
    num_cols = len(data[0])
    aux = 0
    for i in range(num_rows):
        x_i = data[i,0:num_cols-1]
        y_i = data[i,-1]
        for j in range(num_rows):
            x_j = data[j,0:num_cols-1]
            y_j = data[j,-1]
            aux += alpha[i]*alpha[j]*y_i*y_j*kernel(x_j, x_i)        
    return -(sum(alpha) - 1/2*aux)

def func_grad(alpha, data, kernel):
    num_rows = len(data)
    num_cols = len(data[0])
    grad = np.ones(num_rows)
    for i in range(num_rows):
        x_i = data[i,0:num_cols-1]
        y_i = data[i,-1]
        for j in range(num_rows):
            x_j = data[j,0:num_cols-1]
            y_j = data[j,-1]
            grad[i] += -1/2*alpha[j]*y_i*y_j*kernel(x_j, x_i)
    return -grad