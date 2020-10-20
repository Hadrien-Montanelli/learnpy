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
    data_len = len(data_list)
    data_array = np.zeros([data_len,2])
    for k in range(data_len):
        for l in range(2):
            data_array[k,l] = data_list[k][l]
    return data_array