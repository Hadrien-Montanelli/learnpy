#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:37:13 2020

@author: montanelli
"""
import numpy as np

def list_to_array(data_list):
    """Convert a list to a numpy array."""
    number_rows = len(data_list)
    number_cols = len(data_list[0])
    data_array = np.zeros([number_rows,number_cols])
    for k in range(number_rows):
        for l in range(number_cols):
            data_array[k,l] = data_list[k][l]
    return data_array