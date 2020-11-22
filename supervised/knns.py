#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:56:55 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

# Learnpy imports:
from .classifier import classifier

class knns(classifier):
    """
    Class for representing the k-nearest neighbors classifier.
    """
    def __init__(self, n_input, n_train, n_neigbours):
        super().__init__(n_input, n_train)
        # TO IMPROVE: the following implements k=1; extend to k>2.
        if (n_neigbours != 1):
            raise ValueError("The knns algorithm only supports k=1 for now.")
        self.n_neigbours = n_neigbours
        
    def train(self, X, Y):
        """
        Train the k-nearest neighbors classifier.
        
        Inputs
        ------
        X : numpy.ndarray
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy.ndarray
            The labels as a 1xn array. Labels are {0,1}.
        """
        # Get dimensions:
        d = self.n_input
        
        # TO IMPROVE: find the distance that best separates the training data.
        W = np.ones([1, d])
        
        # Store parameters:
        self.params['W'] = W
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
        W = self.params['W']
        X_train = self.params['X_train']
        Y_train = self.params['Y_train']

        # Define the distance:
        distance = lambda x,y: np.sqrt(np.sum((x - y) * W * (x - y)))
        
        # Predict:
        Y_hat = np.zeros(m)
        for i in range(m):
            dist_to_training = np.zeros(n)
            for j in range(n):
                dist_to_training[j] = distance(X[i,:], X_train[j,:])
            pos_min = np.argmin(dist_to_training)
            Y_hat[i] = Y_train[int(pos_min)]
        
        return Y_hat