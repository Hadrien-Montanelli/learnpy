#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:21:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from pylab import meshgrid 

# Learnpy imports:
from learnpy.misc import regression

# %% Examples.

# Try linear model with noise in 1D:
np.random.seed(2)
n = 100
x = np.linspace(-1, 1, 100)
y = 2*x + 6 + 5e-1*np.random.randn(n)
plt.plot(x, y, '.r')
alpha, beta = regression(x, y, 'linear')
print([alpha, beta])
plt.plot(x, alpha + beta*x, 'k')

# Try linear model with noise in 2D:
n = 10
x_1 = np.linspace(-1, 1, n)
x_2 = np.linspace(-1, 1, n)
x = np.zeros([n**2, 2])
for i in range(n):
    for j in range(n):
        idx = i + n*j
        x[idx, 0] = x_1[i]
        x[idx, 1] = x_2[j]
y = 10 + 0*x[:,0] + 3*x[:,1] + 2e-1*np.random.randn(n**2)
fig = plt.figure()
ax = plt.gca(projection='3d')
for k in range(x.shape[0]):
    ax.scatter(x[k,0], x[k,1], y[k], c='r')   
alpha, beta = regression(x, y, 'linear')
print([alpha, beta])
X_1, X_2 = meshgrid(x[:,0], x[:,1])
Y = alpha + beta[0]*X_1 + beta[1]*X_2
surf = ax.plot_surface(X_1, X_2, Y, cmap = cm.binary, alpha=0.2)