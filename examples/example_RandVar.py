#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:58:00 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from math import exp, sqrt, pi
import numpy as np
from RandVar import RandVar

# Define X as a normal distribution:
mean = 0.21
left_bound = -10 + mean
right_bound = 10 + mean
var = 1.5
pdf = lambda x: 1/sqrt(2*pi*var)*exp(-1/(2*var)*(x-mean)**2)
domain = np.array([left_bound, right_bound])
X = RandVar(pdf, domain)
X.display()
X.plot('b')

# Define Y as the standard normal distribution by scaling X:
Y = RandVar.scale(X, 1/sqrt(var), -mean/sqrt(var))
Y.display()
Y.plot('r')

# Define Z as the sum X+Y:
a = 1.1
b = 1.4
Z = RandVar.plus(RandVar.scale(X, a, 0), RandVar.scale(Y, b, 0))
Z.display()
print('Mean should be ', round(a*X.mean()+b*Y.mean(),10), '.', sep='')
print('Variance should be ', round(a**2*X.var()+b**2*Y.var(),10), '.', sep='')
Z.plot('k')