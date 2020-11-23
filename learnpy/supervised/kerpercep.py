#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:45:47 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

# Learnpy imports:
from .classifier import classifier

class kerpercep(classifier):
    """
    Class for representing the kernel perceptron classifier.
    """
    def __init__(self, n_input, n_train, kernel):
        super().__init__(n_input, n_train)
        self.kernel = kernel
        
    def train(self, X, Y):
        """
        Train the kernel perceptron classifier.
        
        Inputs
        ------
        X : numpy.ndarray
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy.ndarray
            The labels as a 1xn array. Labels are {0,1}.
        """
        # Get dimension:
        n = self.n_train
        
        # Get the kernel:
        K = self.kernel

        # Change labels {0,1} to {-1,1}:
        Y = 2*Y - 1
        
        # Initialize parameters:
        alpha = np.zeros(n)
           
        # Main loop:
        test = 0
        while (test == 0):
            test = 1
            for i in range(n):
                Y_hat = 0
                for j in range(n):
                    Y_hat += alpha[j]*Y[j]*K(X[j,:], X[i,:])
                Y_hat = np.sign(Y_hat)
                if (Y_hat != Y[i]):
                    alpha[i] += 1
                    test = 0
                
        # Store parameters:
        self.params['alpha'] = alpha
        self.params['X_train'] = X
        self.params['Y_train'] = Y

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
        # Get the dimenions:
        n = self.n_train
        m = len(X)
        
        # Get the parameters:
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
            Y_hat[i] = np.sign(Y_hat[i])
            
        # Change labels {-1,1} to {0,1}:
        Y_hat = 1/2*(Y_hat + 1)
        
        return Y_hat