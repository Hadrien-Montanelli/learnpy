#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:06:00 2020

@author: montanelli
"""
# Imports:
import sys
sys.path.append('../unsupervised')
sys.path.append('../misc')
from kmeans import kmeans
from utilities import csv_to_array
import matplotlib.pyplot as plt

# Use k-means:
data = csv_to_array('../dataset/heights_weights_training.csv')
clusters = kmeans(data[:,0:2], 2)
print(clusters)
number_rows = len(data)
for i in range(number_rows):
    if i in clusters[0]:
        color = '.r'
    else:
        color = '.b'
    plt.plot(data[i,0],data[i,1],color)