#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:22:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from pca import pca
from utilities import list_to_array
import csv
import matplotlib.pyplot as plt

# Load data:
data = list(csv.reader(open('../dataset/crabs.csv')))
data.pop(0)
for row in data:
    del row[0]
    del row[0]
    del row[0]
X = list_to_array(data)

# PCA:
D, V = pca(X)

# Post-processing:
Z = X @ V
plt.plot(Z[:,1], Z[:,2], '.')