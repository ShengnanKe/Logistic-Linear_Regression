# -*- coding: utf-8 -*-
"""
Class: DSCI552
Assignment 4
@author: Anli Shengnan Ke
"""

import numpy as np
import pandas as pd
from math import *


class LinearRegression():
    def __init__(self, file):
        data = np.loadtxt(file, delimiter=',')
        self.N = data.shape[0]
        self.dim = 2
        self.X = np.concatenate((np.ones((self.N,1)), data[:,:2]), axis=1)
        self.y = data[:,-1]
        self.D = self.X.T
        
    def runLinearRegression(self):
        self.w = np.linalg.inv(self.D.dot(self.D.T)).dot(self.D).dot(self.y)
        
    def getW(self):
        return self.w
    

model_linear_regre = LinearRegression('linear-regression.txt')
model_linear_regre.runLinearRegression()
print('Weights: \n', model_linear_regre.w)
