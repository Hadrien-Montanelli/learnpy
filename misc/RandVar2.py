#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:13:13 2020

Copyright 2020 by Hadrien Montanelli.
"""
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import meshgrid
from RandVar import RandVar

class RandVar2:
    """
    Class for representing jointly continuous random variables in 2D.
    """
    def __init__(self, pdf, domain):
        """Construct a RandVar2 from a pdf and a domain."""
        self.pdf = pdf
        self.domain = domain
    
    def norm(self):
        """Return the norm of the pdf of self."""
        left_x_bound = self.domain[0]
        right_x_bound = self.domain[1]
        left_y_bound = self.domain[2]
        right_y_bound = self.domain[3]
        integrand = lambda x,y: self.pdf(x, y)
        output = intg.dblquad(integrand, left_y_bound, right_y_bound, 
                           lambda y: left_x_bound, right_x_bound)
        return output[0]
    
    def mean(self):
        """Return the mean of self."""
        left_x_bound = self.domain[0]
        right_x_bound = self.domain[1]
        left_y_bound = self.domain[2]
        right_y_bound = self.domain[3]
        randvar_x = RandVar([], [left_x_bound, right_x_bound])
        randvar_x.pdf = lambda x: intg.quad(lambda y: self.pdf(x, y), 
                                    left_y_bound,
                                    right_y_bound)[0]
        randvar_y = RandVar([], [left_y_bound, right_y_bound])
        randvar_y.pdf = lambda y: intg.quad(lambda x: self.pdf(x, y), 
                                    left_x_bound,
                                    right_x_bound)[0]
        mean_x = randvar_x.mean()
        mean_y = randvar_y.mean()
        return [mean_x, mean_y]
    
    def covar(self):
        """Return the covariance matrix of self."""
        left_x_bound = self.domain[0]
        right_x_bound = self.domain[1]
        left_y_bound = self.domain[2]
        right_y_bound = self.domain[3]
        randvar_x = RandVar([], [left_x_bound, right_x_bound])
        randvar_x.pdf = lambda x: intg.quad(lambda y: self.pdf(x, y), 
                                    left_y_bound,
                                    right_y_bound)[0]
        randvar_y = RandVar([], [left_y_bound, right_y_bound])
        randvar_y.pdf = lambda y: intg.quad(lambda x: self.pdf(x, y), 
                                    left_x_bound,
                                    right_x_bound)[0]
        
        integrand = lambda x,y: x*y*self.pdf(x, y)
        mean_xy = intg.dblquad(integrand, left_y_bound, right_y_bound, 
                           lambda y: left_x_bound, right_x_bound)[0]
        mean_x = randvar_x.mean()
        mean_y = randvar_y.mean()
        covar_xx = randvar_x.var()
        covar_yy = randvar_y.var()
        covar_xy = mean_xy - mean_x*mean_y
        return [[covar_xx, covar_xy], [covar_xy, covar_yy]]
    
    def display(self):
        """Display informatons about self."""
        tol = 6
        print('---------------------------------')
        print('norm:  ', round(self.norm(), tol))
        print('mean:  ', np.vectorize(round)(self.mean(), tol))
        print('covar: ', np.vectorize(round)(self.covar()[0], tol))
        print('       ', np.vectorize(round)(self.covar()[1], tol), '\n')
         
    def plot(self):
        """Plot the pdf of self."""
        left_x_bound = self.domain[0]
        right_x_bound = self.domain[1]
        left_y_bound = self.domain[2]
        right_y_bound = self.domain[3]
        number_x_points = int(10*(right_x_bound - left_x_bound))
        number_y_points = int(10*(right_y_bound - left_y_bound))
        x = np.linspace(left_x_bound, right_x_bound, number_x_points)
        y = np.linspace(left_y_bound, right_y_bound, number_y_points)
        X,Y = meshgrid(x,y)
        Z = np.vectorize(self.pdf)(X,Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, 
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5)