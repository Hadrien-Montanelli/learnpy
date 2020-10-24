#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:05:24 2020

@author: montanelli
"""
import numpy as np
from numpy import linalg as LA

def kmeans(data, k):
    """Return k clusters of data using the k-means algorithm."""
    # Get dimension:
    number_rows = len(data)
    
    # Assign each observation to one of the clusters:
    cluster_index = initialise_cluster_index(k, number_rows)
    
    # Iterate the following 2 steps until cluster assignments stop changing:
    test = 0
    while test == 0:
        
        # For each cluster compute the cluster mean:
        cluster_mean = []
        for i in range(k):
            cluster_mean.append(compute_cluster_mean(data, cluster_index[i]))
            
        # Re-assign all observations to the cluster whose mean is closest:
        new_cluster_index = [[] for _ in range(k)]
        for i in range(number_rows):
            dist_to_cluster = np.zeros([k, 1])
            for j in range(k):
                dist_to_cluster[j] = LA.norm(data[i,:] - cluster_mean[j])**2  
            pos_min = np.argmin(dist_to_cluster)
            new_cluster_index[pos_min].append(i)

        # Stop if clusters haven't changed:
        if new_cluster_index == cluster_index:
            test = 1
        else:
            cluster_index = new_cluster_index.copy()
        
    return cluster_index

def compute_cluster_mean(data, cluster_index):
    number_rows = len(cluster_index)
    cluster_mean = 1/number_rows*np.sum(data[cluster_index, :], 0)
    return cluster_mean

def initialise_cluster_index(k, number_rows): 
    # TO IMPROVE: do a random assignment.
    cluster_index = []
    for i in range(k-1):
        cluster_index.append([i])
    cluster_index.append([i for i in range(k-1, number_rows)])
    return cluster_index