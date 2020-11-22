#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:22:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
import csv

# Learnpy imports:
from learnpy.misc import list_to_array
from learnpy.misc import pca

# %% Example.

# Load data:
data = list(csv.reader(open('../../dataset/crabs.csv')))
data.pop(0)
for row in data:
    del row[0]
    del row[0]
    del row[0]
X = list_to_array(data)

# PCA:
D, V = pca(X)

# Plot two principal components:
Z = X @ V
plt.plot(Z[:,1], Z[:,2], '.')