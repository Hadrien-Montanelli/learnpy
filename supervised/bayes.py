#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:38:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

# Learnpy imports:
from .classifier import classifier
from .mle import mle

class bayes(classifier):
    """
    Class for representing the naive Bayes classifier.
    """
    def __init__(self, n_input, n_train, model):
        super().__init__(n_input, n_train)
        self.model = model
    
    def train(self, X, Y):
        """
        Train the naive Bayes classifier.
        
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
        
        # Get the model:
        model = self.model
        
        # Separate labels 0 and 1:
        X0 = X[Y==0]
        X1 = X[Y==1]

        # Initialize parameters:
        randvar = []
        priors = np.array([len(X0)/n, len(X1)/n])
        
        # Main loop:
        for j in range(d):
            randvar0 = mle(X0[:,j], model)
            randvar1 = mle(X1[:,j], model)
            randvar.append([randvar0, randvar1])
                
        # Store parameters:
        self.params['randvar'] = randvar
        self.params['priors'] = priors

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
        # Get dimensions:
        d = self.n_input
        m = len(X)
        
        # Get the parameters:
        randvar = self.params['randvar']
        priors = self.params['priors']
        
        # Predict:
        Y_hat = []
        for k in range(m):
            p0 = priors[0]
            p1 = priors[1]
            for j in range(d):
                p0 *= randvar[j][0].pdf(X[k,j])
                p1 *= randvar[j][1].pdf(X[k,j])  
            Y_hat.append(float(p0 < p1))
        Y_hat = np.array(Y_hat)
        
        return Y_hat