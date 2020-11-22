#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:21:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import exp, pi, sqrt
import numpy as np

# Learnpy imports:
from learnpy.supervised import RandVar2

# %% Examples.

# Try normal distribution:
left_x_bound = -5
right_x_bound = 5
left_y_bound = -5
right_y_bound = 5
rho = 0.1
scaling = 1/(2*pi*sqrt(1-rho**2))
pdf = lambda x,y: scaling*exp(-(x**2-2*rho*x*y+y**2)/(2*(1-rho**2)))
domain = np.array([left_x_bound, right_x_bound, left_y_bound, right_y_bound])
X = RandVar2(pdf, domain)
X.display()
X.plot()