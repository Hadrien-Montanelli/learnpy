#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:18:53 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np

def montecarlo(function, domain, N=1000):
    """Monte Carlo integrtion of a function on a rectangular domain.
    
    Input
    -----
    function : lambda
        The function to integrate.
    
    domain : numpy array
        A rectangluar domain stored as a (2xD)x1 array in dimenion D.
    
    N : int
        The number of sample points.
        
    Output
    ------
    The approximate integral of the function.
    
    Example
    -------
    This is an example in 2D.
    
        f = lambda x: x[0]**2*cos(x[1])
        dom = np.array([0, 2, -1, 1])
        I = montecarlo(f, dom)
    
    See also the 'example_montecarlo' file.
    
    """
    # Get dimension and volume:
    dimension = int(len(domain)/2)
    V = 1
    for j in range(dimension):
        V *= domain[2*j+1] - domain[2*j]
    
    # Sample the domain uniformly:
    points = np.zeros([N, dimension])
    for i in range(N):
        for j in range(dimension):
            a = domain[2*j]
            b = domain[2*j+1]
            points[i, j] = a + (b - a)*np.random.uniform()
        
    # Compute the integral:
    S = 0
    for point in points:
        S += function(point)
    I = V/N*S
        
    return I