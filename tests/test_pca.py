
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:22:50 2020

@author: montanelli
"""
import sys
sys.path.append('../')
from IPython import get_ipython
from pca import pca
from utilities import list_to_array
import csv
import matplotlib.pyplot as plt

# Clear workspace:
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# Load data:
data = list(csv.reader(open('../data/crabs.csv')))
data.pop(0)
for row in data:
    del row[0]
    del row[0]
    del row[0]
X = list_to_array(data)

# PCA:
D,V = pca(X)

# Post-processing:
Z = X @ V
plt.plot(Z[:,1],Z[:,2], '.')
