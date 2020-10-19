#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:55:16 2020

@author: montanelli
"""
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt

class RandVar:
    """
    Class for representing continuous random variables.
    """
    def __init__(self, pdf, domain):
        """Construct a RandVar from a pdf and a domain."""
        self.pdf = pdf
        self.domain = domain
    
    def norm(self):
        """Return the norm of the pdf of self."""
        left_bound = self.domain[0]
        right_bound = self.domain[1]
        integrand = lambda x: self.pdf(x)
        output = intg.quad(integrand, left_bound, right_bound)
        return output[0]
    
    def mean(self):
        """Return the mean of self."""
        left_bound = self.domain[0]
        right_bound = self.domain[1]
        integrand = lambda x: x*self.pdf(x)
        output = intg.quad(integrand, left_bound, right_bound)
        return output[0]
    
    def var(self):
        """Return the variance of self."""
        left_bound = self.domain[0]
        right_bound = self.domain[1]
        integrand_1 = lambda x: x*self.pdf(x)
        output_1 = intg.quad(integrand_1, left_bound, right_bound)
        integrand_2 = lambda x: x**2*self.pdf(x)
        output_2 = intg.quad(integrand_2, left_bound, right_bound)
        return output_2[0] - (output_1[0])**2
    
    def display(self):
        """Display informatons about self."""
        print('------------------')
        print('norm:', round(self.norm(), 10))
        print('mean:', round(self.mean(), 10))
        print('var: ', round(self.var(), 10),  '\n')

    def scale(self, scaling, shift):
        """Return scaling*self + shift."""
        self_scaled = RandVar([], self.domain)
        self_scaled.pdf = lambda x: 1/scaling*self.pdf((x-shift)/scaling)
        return self_scaled
        
    def plus(self, randvar):
        """Return self + randvar."""
        self_plus_randvar = RandVar([], self.domain)
        left_bound = self.domain[0]
        right_bound = self.domain[1]
        integrand = lambda x,z: self.pdf(x)*randvar.pdf(z-x)
        self_plus_randvar.pdf = lambda z: intg.quad(lambda x: integrand(x, z), 
                                                     left_bound,
                                                     right_bound)[0]
        return self_plus_randvar
        
    def minus(self, randvar):
        """Return self - randvar."""
        return RandVar.plus(self, RandVar.scale(randvar, -1, 0))
         
    def plot(self):
        """Plot the pdf of self."""
        left_bound = self.domain[0]
        right_bound = self.domain[1]
        number_points = int(round(100*(right_bound - left_bound)))
        x = np.linspace(left_bound, right_bound, number_points)
        y = [self.pdf(i) for i in x]
        plt.plot(x, y)