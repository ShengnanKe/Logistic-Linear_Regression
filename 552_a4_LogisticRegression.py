# -*- coding: utf-8 -*-
"""
Class: DSCI552
Assignment 4
@author: Anli Shengnan Ke
"""

import numpy as np
import pandas as pd
from math import *


class LogisticRegression():
    def __init__(self, file, max_iter, eta):
        data = np.loadtxt(file, delimiter=',')
        data[:,3] = 1  # padding
        self.X = data[:, [3, 0, 1, 2]]
        self.y = data[:, 4]
        self.N = data.shape[0]  # number of data points
        self.dim = 3  # number of dimensions
        self.max_iteration = max_iter
        self.w = np.ones([self.X.shape[1],1])  # initialize all w to 1
        self.eta = eta  # learning rate
       
    def runLogisticRegression(self):
        for _ in range(self.max_iteration):
            dw_sum = 0
            for i in range(self.N):
                y_w_x = self.y[i] * np.dot(self.w.T, self.X[i])
                e_y_w_x = exp(y_w_x)
                denominator = e_y_w_x + 1
                nominator = self.y[i] * (self.X[i])
                tmp = nominator / denominator
                dw_sum += tmp
                # if i == 100:
                #     print('nominator: ', nominator)
                #     print('denominator: ', denominator)
            dw = -1 * dw_sum / self.N
            # print('dw: ', dw)
            # print('w: ', self.w)
            self.w -= self.eta*(dw.reshape(4,1))
            if (_+1) % 1000 == 0:
                print('finished', _+1, 'iterations')  
    
    def getW(self):
        return self.w
    
    def sigmoid(x):
        return exp(x)/(1 + exp(x))
            
    def getAccuracy(self):
        self.prediction = []
        for i in range(self.N):
            tmp = np.dot(self.w.T, self.X[i])
            probability = LogisticRegression.sigmoid(tmp)
            self.prediction.append(1 if probability >= 0.5 else -1)
        accuracy = np.array(self.prediction) == self.y
        return str(accuracy.mean()*100.0)+"%"

    
        
        
model_lr = LogisticRegression('classification.txt', 7000, 0.1)
model_lr.runLogisticRegression()
print('Weights:\n', model_lr.getW())
print('Accuracy: ', model_lr.getAccuracy())
