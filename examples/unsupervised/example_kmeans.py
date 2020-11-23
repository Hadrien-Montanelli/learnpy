#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:06:00 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt

# Learnpy imports:
from learnpy.misc import csv_to_array
from learnpy.unsupervised import kmeans

# %% Simple example.

# Find clusters on the following data (in [cm, kg]) with k-means algorithm:
data = csv_to_array('../../datasets/2d_training.csv')
n_clusters = 2
clusters = kmeans(data[:,0:2], n_clusters)
print(clusters)
number_rows = len(data)
for i in range(number_rows):
    if i in clusters[0]:
        color = '.r'
    else:
        color = '.b'
    plt.plot(data[i,0], data[i,1], color)