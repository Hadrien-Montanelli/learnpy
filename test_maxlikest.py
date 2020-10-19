#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:56:46 2020

@author: montanelli
"""
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
from maxlikest import maxlikest

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Fit the following data (in cm) with a normal distribution & MLE:
data = np.array([170, 172, 180, 169, 175])
normal_mle = maxlikest(data, 'normal')
normal_mle.display()
normal_mle.plot()
plt.plot(data, np.vectorize(normal_mle.pdf)(data), '.k')

# Fit the following data (in [cm, kg]) with a normal distribution & MLE:
data = np.array([[170,80], [172,90], [180, 68], [169, 77], [175, 100]])
normal_mle = maxlikest(data, 'normal')
normal_mle.display()
normal_mle.plot()
ax = plt.gca(projection='3d')
for k in range(data.shape[0]):
    ax.scatter(data[k,0], data[k,1], 
               normal_mle.pdf(data[k,0], data[k,1]), c='k')