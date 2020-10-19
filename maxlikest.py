#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:57:38 2020

@author: montanelli
"""
from math import *
from RandVar import RandVar

def maxlikest(data, model):
    """ 
    Fit the given data with a prescribed model using MLE.
    
    Models include 'normal'.
    """
    n = len(data)
    if model == 'normal':
        mean = 1/n*sum(data)
        var = 1/n*sum([(x - mean)**2 for x in data])
        pdf = lambda x: 1/sqrt(2*pi*var)*exp(-1/2*(x-mean)**2)
        randvar = RandVar(pdf, [min(data), max(data)])
        return randvar
