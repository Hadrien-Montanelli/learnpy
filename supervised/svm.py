#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:41:31 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

# Learnpy imports:
from .classifier import classifier

class svm(classifier):
    """
    Class for representing support vector machines.
    """
    def train(self, X, Y):
        """
        Train support vector machines.
        
        Inputs
        ------
        X : numpy.ndarray
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy.ndarray
            The labels as a 1xn array. Labels are {0,1}.
        """
        # Get dimensions:
        d = self.n_input
        n = self.n_train

        # Change labels {0,1} to {-1,1}:
        Y = 2*Y - 1
        
        # Initialize parameters:
        W = np.zeros(d+1) # bias W0 stored as last entry
           
        # Cost function:
        def func(w):
            return 1/2*sum(w[:-1]**2.0)

        # Gradient:
        def func_grad(w):
            grad = np.zeros(len(w))
            grad[:-1] = W[:-1]
            return grad

        # Hessian times a vector:
        def func_hess(w, p):
            Hp = np.ones_like(w)
            Hp[:-1] = p[:-1]
            Hp[-1] = 0
            return Hp
    
        # Optimization:
        constraint_mat = np.zeros([n, d+1])
        for k in range(n):
            constraint_mat[k,:d] = Y[k]*X[k,:]
            constraint_mat[k,-1] = Y[k] # bias
        lower_bnd = [1 for _ in range(n)]
        upper_bnd = [np.inf for _ in range(n)]
        lin_constr = LinearConstraint(constraint_mat, lower_bnd, upper_bnd)
        res = minimize(func, W, method = 'trust-constr',
                       jac = func_grad, hessp = func_hess, 
                       constraints = lin_constr,
                       options = {'xtol': 1e-8, 'disp': False})         
        
        # Store parameters:
        self.params['W0'] = res.x[-1]
        self.params['W'] = res.x[:-1]

    def classify(self, X):
        """
        Classify data.
        
        Inputs
        ------
        X : numpy.ndarray
            The testing data as a nxd array for n data points in dimension d.
            
        Output
        ------
        Y_hat : numpy.ndarray
            Predicted labels as a 1xn array. Labels are {0,1}.
        """
        # Get the parameters:
        W0 = self.params['W0']
        W = self.params['W']
            
        # Predict:
        Y_hat = np.sign(W @ X.T + W0)
        
        # Change labels {-1,1} to {0,1}:
        Y_hat = 1/2*(Y_hat + 1)
        
        return Y_hat