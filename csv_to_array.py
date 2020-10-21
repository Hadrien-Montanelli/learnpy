#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:46:13 2020

@author: montanelli
"""
import csv
import numpy as np

def csv_to_array(csv_file):
    """Convert a csv file to a numpy array."""
    data_list = list(csv.reader(open(csv_file)))
    number_rows = len(data_list)
    number_cols = len(data_list[0])
    data_array = np.zeros([number_rows,number_cols])
    for k in range(number_rows):
        for l in range(number_cols):
            data_array[k,l] = data_list[k][l]
    return data_array