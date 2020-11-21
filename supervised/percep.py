#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:43:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

# Learnpy imports:
from .classifier import classifier

class percep(classifier):
    """
    Class for representing the perceptron classifier.
    """
    def train(self, X, Y):
        """
        Train the perceptron classifier.
        
        Inputs
        ------
        X : numpy array
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy array
            The labels as an 1xn array. Labels are {0,1}.
        """
        # Get dimensions:
        d = self.n_input
        n = self.n_train

        # Change labels {0,1} to {-1,1}:
        Y = 2*Y - 1
        
        # Initialize parameters:
        W0 = 0
        W = np.zeros(d)
           
        # Main loop:
        test = 0
        while (test == 0):
            test = 1
            for i in range(n):
                Y_hat = np.sign(W @ X[i, :] + W0)
                if (Y_hat != Y[i]):
                    W += Y[i]*X[i, :]
                    W0 += Y[i]
                    test = 0
                
        # Store parameters:
        self.params['W0'] = W0
        self.params['W'] = W

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
        # Get the parameters:
        W0 = self.params['W0']
        W = self.params['W']
            
        # Predict:
        Y_hat = np.sign(W @ X.T + W0)
        
        # Change labels {-1,1} to {0,1}:
        Y_hat = 1/2*(Y_hat + 1)
        
        return Y_hat