#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:22:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# System imports:
import os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Standard library imports:
import matplotlib.pyplot as plt
import csv

# Learnpy imports:
import misc

# %% Example.

# Load data:
data = list(csv.reader(open('../../dataset/crabs.csv')))
data.pop(0)
for row in data:
    del row[0]
    del row[0]
    del row[0]
X = misc.list_to_array(data)

# PCA:
D, V = misc.pca(X)

# Plot two principal components:
Z = X @ V
plt.plot(Z[:,1], Z[:,2], '.')