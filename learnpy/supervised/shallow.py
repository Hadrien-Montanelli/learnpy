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

class shallow(classifier):
    """
    Class for representing shallow networks.
    """
    def __init__(self, n_input, n_train, n_neurons, 
                 options = {'acc': 'entropy', 'cost': 'entropy', 'disp': False, 
                            'jtol': 0.25, 'lr': 1, 'maxiter': 200}):
        super().__init__(n_input, n_train)
        self.n_neurons = n_neurons
        self.options = options
        
    def train(self, X, Y):
        """
        Train the shallow network using gradient descent.
        
        Inputs
        ------
        X : numpy.ndarray
            The training data as a nxd array for n data in dimension d.
        
        Y : numpy.ndarray
            The labels as a 1xn array. Labels are {0,1}.
        """
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        
        def dsigmoid(x):
            grad = sigmoid(x) * (1 - sigmoid(x))
            return grad
    
        def relu(x):
            return np.maximum(x, 0)
        
        def drelu(x):
            grad = np.zeros(x.shape)
            grad[x>0] = 1
            return grad
               
        # Get the options:
        cost = self.options['cost']
        disp = self.options['disp']
        jtol = self.options['jtol']
        lr = self.options['lr']
        maxiter = self.options['maxiter']
        
        # Get the dimensions:
        d = self.n_input
        n = self.n_train
        N = self.n_neurons
        
        # Initialize parameters:
        W10 = np.zeros([N, 1])
        W1 = 0.01*np.random.randn(d, N)
        W20 = np.zeros([1, 1])
        W2 = 0.01*np.random.randn(N, 1)

        # Main loop:
        iteration = 0
        J0, J = 0, 1
        while (iteration < maxiter and J > jtol*J0):     

            # Forward:
            Z1 = W1.T @ X.T + W10  
            A1 = relu(Z1)
            Z2 = W2.T @ A1 + W20
            A2 = sigmoid(Z2)
    
            # Backward:
            if (cost == 'mse'):
                dA2 = 2 * (A2 - Y)/Y.size
            elif (cost == 'entropy'):
                dA2 = (A2 - Y) / (A2 * (1 - A2))
            dZ2 = dA2 * dsigmoid(Z2)
            dW2 = 1/n * (A1 @ dZ2.T)
            dW20 = 1/n * np.sum(dZ2, axis = 1, keepdims = True)
            dA1 = W2 @ dZ2
            dZ1 = dA1 * drelu(Z1)
            dW1 = 1/n * (X.T @ dZ1.T)
            dW10 = 1/n * np.sum(dZ1, axis = 1, keepdims = True)
            
            # Upadte parameters:
            W10 -= lr * dW10
            W1 -= lr * dW1
            W20 -= lr * dW20
            W2 -= lr * dW2

            # Compute the cost function:            
            J = self.cost(Y, A2)
            if (iteration == 0):
                J0 = self.cost(Y, A2)
                
            # Display informations:
            if (disp == True and iteration % 100 == 0):
                print('Iteration:', iteration)
                print('Cost func:', J)
            
            # Update iteration count:
            iteration += 1
        
        # Store parameters:
        self.params['W10'] = W10
        self.params['W1'] = W1
        self.params['W20'] = W20
        self.params['W2'] = W2
    
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
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        
        def relu(x):
            return np.maximum(x, 0)
        
        # Get the parameters:
        W10 = self.params['W10']
        W1 = self.params['W1']
        W20 = self.params['W20']
        W2 = self.params['W2']
        
        # Predict:
        Z1 = W1.T @ X.T + W10 
        A1 = relu(Z1) 
        Z2 = W2.T @ A1 + W20
        Y_hat = sigmoid(Z2)
        
        return Y_hat