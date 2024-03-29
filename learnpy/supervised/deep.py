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

class deep(classifier):
    """
    Class for representing deep networks.
    """
    def __init__(self, n_input, n_train, n_layers, n_neurons, 
                 options = {'acc': 'entropy', 'cost': 'entropy', 'disp': False, 
                            'jtol': 0.25, 'lr': 1, 'maxiter': 200, 
                            'output': 'sigmoid'}):
        super().__init__(n_input, n_train)
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.options = options
        
    def train(self, X, Y):
        """
        Train the deep network using gradient descent.
        
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
        output = self.options['output']

        # Get the dimensions:
        d = self.n_input
        n = self.n_train
        L = self.n_layers
        N = self.n_neurons 
        
        # Initialize parameters:
        for l in range(2, L+1):
            self.params['W' + str(l) + str(0)] = np.zeros([N[l-1], 1])
            self.params['W' + str(l)] = 0.01*np.random.randn(N[l-2], N[l-1])      
        self.params['W' + str(1) + str(0)] = np.zeros([N[0], 1])
        self.params['W' + str(1)] = 0.01*np.random.randn(d, N[0])
        self.params['W' + str(L+1) + str(0)] = np.zeros([1, 1])
        self.params['W' + str(L+1)]  = 0.01*np.random.randn(N[-1], 1)
           
        # Main loop:
        iteration = 0
        J0, J = 0, 1
        while (iteration < maxiter and J > jtol*J0):  
                 
            # Forward:
            values = {}
            values['A' + str(0)] = X.T
            A = values['A' + str(0)]
            for l in range(1, L+1):
                W = self.params['W' + str(l)]
                W0 = self.params['W' + str(l) + str(0)] 
                Z = W.T @ A + W0
                A = relu(Z)
                values['Z' + str(l)] = Z
                values['A' + str(l)] = A
            W = self.params['W' + str(L+1)]
            W0 = self.params['W' + str(L+1) + str(0)] 
            Z = W.T @ A + W0
            if (output == 'sigmoid'):
                A = sigmoid(Z)
            elif (output == 'linear'):
                A = Z
            values['Z' + str(L+1)] = Z
            values['A' + str(L+1)] = A

            # Backward:
            for l in range(L+1, 0, -1):
                A, A_prev = values['A' + str(l)], values['A' + str(l-1)]
                Z = values['Z' + str(l)]
                if (l == L+1):
                    if (cost == 'mse'):
                        dA = 2 * (A - Y)/Y.size
                    elif (cost == 'entropy'):
                        dA = (A - Y) / (A * (1 - A)) 
                    if (output == 'sigmoid'):
                        dZ = dA * dsigmoid(Z)
                    elif (output == 'linear'):
                        dZ = dA
                else:
                    dA = W @ dZ
                    dZ = dA * drelu(Z)
                W = self.params['W' + str(l)]
                dW = 1/n * (A_prev @ dZ.T)
                dW0 = 1/n * np.sum(dZ, axis = 1, keepdims = True)               
                self.params['W' + str(l)] -= lr * dW
                self.params['W' + str(l) + str(0)] -= lr * dW0
            
            # Compute the cost function:     
            A = values['A' + str(L+1)]
            J = self.cost(Y, A)
            if (iteration == 0):
                J0 = self.cost(Y, A)
                
            # Display informations:
            if (disp == True and iteration % 100 == 0):
                print('Iteration:', iteration)
                print('Cost func:', J)
            
            # Update iteration count:
            iteration += 1

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
        
        # Get the dimensions and output:
        L = self.n_layers
        output = self.options['output']

        # Predict:
        A = X.T
        for l in range(1, L+1):
            W = self.params['W' + str(l)]
            W0 = self.params['W' + str(l) + str(0)] 
            Z = W.T @ A + W0
            A = relu(Z)
        W = self.params['W' + str(L+1)]
        W0 = self.params['W' + str(L+1) + str(0)] 
        Z = W.T @ A + W0
        if (output == 'sigmoid'):
            Y_hat = sigmoid(Z)
        elif (output == 'linear'):
            Y_hat = Z
        
        return Y_hat