#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:05:24 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from numpy import linalg as LA

def kmeans(X, k):
    """Return k clusters in X using the k-means algorithm.
    
    Inputs
    ------
    X : numpy.ndarray
        The data as a nxd array for n data points in dimension d. 
        
    k : int
        The number of clusters to find in X.
    
    Output
    ------
    output : list
        A list containing k lists of indicdes for each of the k clusters.
                
    Example
    -------
    This is an example with 2d data.
    
        import numpy as np
        from learnpy.unsupervised import kmeans
        
        data = np.array([[170, 80], [172, 90], [180, 68], [169, 77]])
        output = kmeans(data, 2)
        print(output)
      
    See also the 'example_kmeans' file.
    """
    # Get the number of data points:
    n = len(X)
    
    # Assign each observation to one of the clusters:
    cluster_index = initialize_cluster_index(k, n)
    
    # Iterate the following 2 steps until cluster assignments stop changing:
    test = 0
    while (test == 0):
        
        # For each cluster compute the cluster mean:
        cluster_mean = []
        for i in range(k):
            cluster_mean.append(compute_cluster_mean(X, cluster_index[i]))

        # Re-assign all observations to the cluster whose mean is closest:
        new_cluster_index = [[] for _ in range(k)]
        for i in range(n):
            dist_to_cluster = np.zeros(k)
            for j in range(k):
                dist_to_cluster[j] = LA.norm(X[i, :] - cluster_mean[j])**2  
            pos_min = np.argmin(dist_to_cluster)
            new_cluster_index[pos_min].append(i)

        # Stop if clusters haven't changed:
        if (new_cluster_index == cluster_index):
            test = 1
        else:
            cluster_index = new_cluster_index.copy()
        
    return cluster_index

def compute_cluster_mean(X, cluster_index):
    """Compute the cluster mean of a cluster. 
    
    Inputs
    ------
    X : numpy.ndarray
        The data stored as a nxd array for n observations in dimension d.
        
    cluster_index : list
        The list of indices of points in X that belong to the cluster whose 
        mean will be computed.
    
    Output
    ------
    ouptput : numpy.ndarray
        The cluster mean as a dx1 array.
    """
    number_rows = len(cluster_index)
    cluster_mean = 1/number_rows*np.sum(X[cluster_index, :], 0)
    return cluster_mean

def initialize_cluster_index(k, n): 
    """Initialize the cluster asisgnment randomly.
    
    Inputs
    ------
    k : int
        The number of clusters.    
    
    n : int
        The number of data points.
        
    Output
    ------
    output : list
        A list containing k lists of indicdes for each of the k clusters. 
    """
    # Assign one point to each cluster:
    cluster_index = [[] for _ in range(k)]
    for i in range(k):
        cluster_index[i].append(i)
    
    # Assign the rest randomly:
    for i in range(k, n):
        temp = np.random.randint(k)
        cluster_index[temp].append(i)

    return cluster_index