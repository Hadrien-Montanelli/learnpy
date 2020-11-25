#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:37:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
from abc import ABC, abstractmethod
import numpy as np

class classifier(ABC):
    """
    Abstract class for representing binary classifiers.
    """
    def __init__(self, n_input, n_train):
        super().__init__()
        self.n_input = n_input
        self.n_train = n_train
        self.params = {}
        
    @abstractmethod
    def train(self, X, Y):
        pass
    
    @abstractmethod
    def classify(self, X):
        pass

    def accuracy(self, Y, Y_hat):
        """Compute the accuracy of a classifier.
        
        Inputs
        ------
        Y : numpy.ndarray
            Exact values as a 1xn array.
        
        Y_hat : numpy.ndarray
            Predicted values as a 1xn array.
            
        Output
        ------
        output : float
            The accuracy of the classifier in %.
        """
        def _to_binary(x):
            return 1 if x > .5 else 0
        
        Y_hat = np.vectorize(_to_binary)(Y_hat)
        acc = float(Y @ Y_hat.T + (1 - Y) @ (1 - Y_hat.T))/Y.size
        acc = round(100*acc, 2)
        
        return acc
    
    def cost(self, Y, Y_hat):
        """Compute the value of the cost function used during training.
        
        Inputs
        ------
        Y : numpy.ndarray
            Exact values as a 1xn array.
        
        Y_hat : numpy.ndarray
            Predicted values as a 1xn array.
            
        Output
        ------
        output : float
            The value of the cost function.
        """
        cost = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))/Y.size
        
        return cost