#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:11:15 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

# Learnpy imports:
from .classifier import classifier

class kersvm(classifier):
    """
    Class for representing kernel support vector machines.
    """
    def __init__(self, n_input, n_train, kernel):
        super().__init__(n_input, n_train)
        self.kernel = kernel
        
    def train(self, X, Y):
        """
        Train kernel support vector machines.
        
        Inputs
        ------
        X : numpy array
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy array
            The labels as an 1xn array. Labels are {0,1}.
        """
        # Get dimensions:
        n = self.n_train

        # Get the kernel:
        K = self.kernel

        # Change labels {0,1} to {-1,1}:
        Y = 2*Y - 1
        
        # Initialize parameters:
        alpha = np.zeros(n)
           
        # Cost function:
        def func(alpha, X, Y, K):
            n = len(X)
            aux = 0
            for i in range(n):
                for j in range(n):
                    aux += alpha[i]*alpha[j]*Y[i]*Y[j]*K(X[j,:], X[i,:])        
            return -(sum(alpha) - 1/2*aux)
        
        # Gradient:
        def func_grad(alpha, X, Y, K):
            n = len(X)
            grad = np.ones(n)
            for i in range(n):
                for j in range(n):
                    grad[i] += -1/2*alpha[j]*Y[i]*Y[j]*K(X[j,:], X[i,:])        
            return -grad
    
        # Optimization:
        eq_cons = {'type': 'eq',
                     'fun' : lambda alpha: sum(alpha*Y),
                     'jac' : lambda alpha: Y}
        lower_bounds = [0 for _ in range(n)]
        upper_bounds = [np.inf for _ in range(n)]
        bounds = Bounds(lower_bounds, upper_bounds)
        res = minimize(func, alpha, args = (X, Y, K), 
                       method = 'SLSQP', jac = func_grad, bounds = bounds,
                       constraints = eq_cons, 
                       options = {'ftol': 1e-8, 'disp': False})    
            
        # Get bias:
        idx = np.argmax(alpha)
        W0 = Y[idx]
        for i in range(n):
            W0 -= alpha[i]*Y[i]*K(X[i,:], X[idx,:])
        
        # Store parameters:
        self.params['W0'] = W0
        self.params['alpha'] = res.x
        self.params['X_train'] = X
        self.params['Y_train'] = Y

    def classify(self, X):
        """
        Classify data.
        
        Inputs
        ------
        X : numpy array
            The data to classify as a nxd array for n data in dimension d.
            
        Output
        ------
        Y_hat : numpy array
            Predicted labels as a 1xn array. Labels are {0,1}.
        """
        # Get the dimenions:
        n = self.n_train
        m = len(X)
        
        # Get the parameters:
        W0 = self.params['W0']
        alpha = self.params['alpha']
        X_train = self.params['X_train']
        Y_train = self.params['Y_train']
                
        # Get the kernel:
        K = self.kernel
            
        # Predict:
        Y_hat = np.zeros(m)
        for i in range(m):
            for j in range(n):
                Y_hat[i] += alpha[j]*Y_train[j]*K(X_train[j,:], X[i, :])
            Y_hat[i] = np.sign(Y_hat[i] + W0)
        
        # Change labels {-1,1} to {0,1}:
        Y_hat = 1/2*(Y_hat + 1)
        
        return Y_hat