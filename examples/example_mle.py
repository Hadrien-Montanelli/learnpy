#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Imports:
import sys
sys.path.append('../misc')
from mle import mle
import numpy as np
import matplotlib.pyplot as plt

# Compute MLE for the normal distribution and following data (in cm):
data = np.array([170, 172, 180, 169, 175])
normal_mle = mle(data, 'normal')
normal_mle.display()
normal_mle.plot()

# Compute MLE for the normal distribution and following data (in [cm, kg]):
data = np.array([[170,80], [172,90], [180, 68], [169, 77], [175, 100]])
normal_mle = mle(data, 'normal')
normal_mle.display()
normal_mle.plot()
ax = plt.gca(projection='3d')

# Compute MLE for the normal distribution and following data (in [cm, kg, IQ]):
data = np.array([[170,80,100], [172,90,80], [180, 68, 90], [169, 77, 140]])
normal_mle = mle(data, 'normal')