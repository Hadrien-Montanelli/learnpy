#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:18:53 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def montecarlo(f, domain, N=1000):
    """Monte Carlo integrtion of a function on a rectangular domain.
    
    Input
    -----
    f : function
        The function to integrate.
    
    domain : numpy array
        A rectangluar domain as a (2*d)x1 array in dimenion d.
    
    N : int
        The number of sample points.
        
    Output
    ------
    output ; float
        The approximate value of the integral.
    
    Example
    -------
    This is an example in 2d.
    
        import numpy as np
        from learnpy.misc import montecarlo
        
        f = lambda x: x[0]**2*np.cos(x[1])
        dom = np.array([0, 2, -1, 1])
        I = montecarlo(f, dom)
    
    See also the 'example_montecarlo' file.
    
    """
    # Get the dimension:
    d = int(len(domain)/2)
    
    # Compute the volume of the domain:
    V = 1
    for j in range(d):
        V *= domain[2*j+1] - domain[2*j]
    
    # Sample the domain uniformly:
    points = np.zeros([N, d])
    for i in range(N):
        for j in range(d):
            a = domain[2*j]
            b = domain[2*j+1]
            points[i, j] = a + (b - a)*np.random.uniform()
        
    # Compute the integral:
    S = 0
    for point in points:
        S += f(point)
    I = V/N*S
        
    return I