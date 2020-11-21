#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:06:00 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# System imports:
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Standard library imports:
import matplotlib.pyplot as plt

# Learnpy imports:
import misc
import unsupervised

# %% Simple example.

# Find clusters on the following data (in [cm, kg]) with k-means algorithm:
data = misc.csv_to_array('../../dataset/2d_training.csv')
clusters = unsupervised.kmeans(data[:,0:2], 2)
print(clusters)
number_rows = len(data)
for i in range(number_rows):
    if i in clusters[0]:
        color = '.r'
    else:
        color = '.b'
    plt.plot(data[i,0], data[i,1], color)