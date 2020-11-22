#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:18:53 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import cos, sin, exp
import numpy as np

# Learnpy imports:
from learnpy.misc import montecarlo

# %% Examples.

# Test in 1D:
f = lambda x: cos(x)
dom = np.array([1, 3])
I = montecarlo(f, dom)
I_ex = sin(3) - sin(1)
print('Error in 1D:', abs(I-I_ex)/abs(I_ex))

# Test in 2D:
f = lambda x: x[0]**2*cos(x[1])
dom = np.array([0, 2, -1, 1])
I = montecarlo(f, dom)
I_ex = 16/3*sin(1)
print('Error in 2D:', abs(I-I_ex)/abs(I_ex))

# Test in 3D:
f = lambda x: x[0]*x[1]*exp(x[2])
dom = np.array([-2, 1, 0, 1, 1, 2])
I = montecarlo(f, dom)
I_ex = -3/4*(exp(2) - exp(1))
print('Error in 3D:', abs(I-I_ex)/abs(I_ex))