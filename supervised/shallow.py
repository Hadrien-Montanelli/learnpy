#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:54:29 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def shallow(X_train, X_test, y_train, y_test, N):
    """
    Use a shallow network for binary classification.
    
    Inputs
    ------
    
    Outputs
    -------
      
    See the 'example_shallow' file.
    """
    # Get dimensions:
    n = len(X_train)
    d = len(X_train[0])
    
    # Initialize weights:
    W1 = 0.01*np.random.randn(d, N)
    W10 = np.zeros([N, 1])
    W2 = 0.01*np.random.randn(N, 1)
    W20 = np.zeros([1, 1])
    
    # Reshape data:
    X = X_train.T
    Y = np.zeros([1, n])
    for k in range(n):
        Y[0, k] = y_train[k]
    # print('X:', [len(X), len(X[0])])
    # print('Y:', [len(Y), len(Y[0])])
       
    # Main loop:
    max_iter = 100
    iter = 0
    while iter < max_iter:
                
        # Forward:
        Z1 = W1.T @ X + W10
        A1 = sigmoid(Z1)   
        Z2 = W2.T @ A1 + W20
        A2 = sigmoid(Z2)
        # print('Z1:', [len(Z1), len(Z1[0])])
        # print('A1:', [len(A1), len(A1[0])])
        # print('Z2:', [len(Z2), len(Z2[0])])
        # print('A2:', [len(A2), len(A2[0])])
        print('Cost:', loss(Y, A2))

        # Backward:
        dZ2 = A2 - Y
        dW2 = 1/n* (A1 @ dZ2.T)
        dW20 = 1/n*np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2 @ dZ2 * A1 * (1 - A1)
        dW1 = 1/n*(X @ dZ1.T)
        dW10 = 1/n*np.sum(dZ1, axis=1, keepdims=True)
        # print('dW2:', [len(dW2), len(dW2[0])])
        # print('dW20:', [len(dW20), len(dW20[0])])
        # print('dW1:', [len(dW1), len(dW1[0])])
        # print('dW10:', [len(dW10), len(dW10[0])])
        
        # Upadte weights:
        eta = 1
        W10 = W10 - eta*dW10
        W1 = W1 - eta*dW1
        W20 = W20 - eta*dW20
        W2 = W2 - eta*dW2
        
        iter += 1
        
    # Testing: 
    X = X_test.T
    Y = np.zeros([1, len(X[0])])
    for k in range(len(X[0])):
        Y[0, k] = y_test[k]
    Z1 = W1.T @ X + W10
    A1 = sigmoid(Z1)   
    Z2 = W2.T @ A1 + W20
    A2 = sigmoid(Z2)
    
    return accuracy(Y, A2)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def loss(Y, Y_hat):
    """
    Y: vector of true value
    Y_hat: vector of predicted value
    """
    n = len(Y[0])
    s = Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
    loss = -np.sum(s)/n
    return loss

def accuracy(Y, Y_pred):
    """
    Y: vector of true value
    Y_pred: vector of predicted value
    """
    def _to_binary(x):
        return 1 if x > .5 else 0

    Y_pred = np.vectorize(_to_binary)(Y_pred)
    acc = float(np.dot(Y, Y_pred.T) + np.dot(1 - Y, 1 - Y_pred.T))/Y.size
    return acc