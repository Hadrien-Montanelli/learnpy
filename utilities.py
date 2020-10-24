#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:34:15 2020

@author: montanelli
"""
import numpy as np
import csv

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

def list_to_array(data_list):
    """Convert a list to a numpy array."""
    number_rows = len(data_list)
    number_cols = len(data_list[0])
    data_array = np.zeros([number_rows,number_cols])
    for k in range(number_rows):
        for l in range(number_cols):
            data_array[k,l] = data_list[k][l]
    return data_array