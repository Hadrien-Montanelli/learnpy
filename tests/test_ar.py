#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:21:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
import numpy as np
import matplotlib.pyplot as plt
from ar import ar
from math import sqrt

# Example with p=2:
np.random.seed(2)
n = 100
p = 2
x = np.zeros(n)
phi_0 = 1
phi_1 = 0.9
phi_2 = -0.8
x[0] = phi_0
x[1] = phi_0 + phi_1*x[0]
var = 1e-2
for k in range(n-2):
    x[k+2] = phi_0 + phi_1*x[k+1] + phi_2*x[k]
x = x + sqrt(var)*np.random.randn(n)
plt.plot(x, '.-r')

# Compute AR(p):
alpha, beta = ar(x, p)
print(alpha, beta)

# Plot results:
xx = np.zeros(n)
xx[0] = x[0]
xx[1] = xx[0]
for k in range(n-2):
    xx[k+2] = alpha + beta[0]*xx[k+1] + beta[1]*xx[k]
plt.plot(xx, '.-b')