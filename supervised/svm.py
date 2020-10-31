#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:41:31 2020

@author: montanelli
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

def svm(training_data, testing_data):
    """
    Use support vector machines for binary classification.
    
    The first columns of the training and testing data represent the data as 
    numpy arrays, while the last colum represents the label (0 or 1).
    
    Example:
        
        
        
        
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
    constraint_mat = np.zeros([number_rows_training, number_cols])
    for k in range(number_rows_training):
        data = training_data[k, 0:number_cols-1]
        label = training_data[k,-1]
        constraint_mat[k, 0:number_cols-1] = label*data
        constraint_mat[k,-1] = label*1 # bias
    lower_bnd = [1 for _ in range(number_rows_training)]
    upper_bnd = [np.inf for _ in range(number_rows_training)]
    linear_constraint = LinearConstraint(constraint_mat, lower_bnd, upper_bnd)
    w = np.zeros(number_cols)
    res = minimize(func, w, method='trust-constr',
                   jac=func_grad, hess=func_hess, 
                   constraints = linear_constraint,
                   options={'xtol': 1e-8, 'disp': True})
    w = res.x
    w0 = w[-1]
    w = w[:-1]
    
    # Testing: 
    error = []
    for i in range(number_rows_testing):
        label = np.sign(w @ testing_data[i,0:number_cols-1] + w0)
        error.append(1/number_rows_testing*float(label != testing_data[i,-1]))
        
    return w0, w, error

# Note that the last entry of the weights represents the bias.
def func(w):
    return 1/2*sum(w[:-1]**2.0)

def func_grad(w):
    grad = np.zeros(len(w))
    grad[:-1] = w[:-1]
    return grad

def func_hess(w):
    hess = np.eye(len(w))
    hess[-1,-1] = 0
    return hess