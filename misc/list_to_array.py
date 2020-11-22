#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:34:15 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def list_to_array(data_list):
    """Convert a list to a numpy.ndarray."""
    number_rows = len(data_list)
    number_cols = len(data_list[0])
    data_array = np.zeros([number_rows, number_cols])
    for k in range(number_rows):
        for l in range(number_cols):
            data_array[k, l] = data_list[k][l]
    return data_array