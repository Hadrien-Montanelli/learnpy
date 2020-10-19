#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

@author: montanelli
"""
from RandVar import RandVar
from maxlikest import maxlikest
import matplotlib.pyplot as plt

# Fit the following heights (cm) with a normal:
heights = [170, 172, 180, 169, 175]
normal_fit = maxlikest(heights, 'normal')
normal_fit.plot()
plt.plot(heights, [normal_fit.pdf(x) for x in heights], '.r')