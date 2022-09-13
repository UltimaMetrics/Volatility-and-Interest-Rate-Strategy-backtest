# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:41:22 2022

@author: sigma
"""

from statsmodels.tsa.stattools import adfuller

def stationarity(data, cutoff=0.05):
    if adfuller(data)[1] < cutoff:
        print('The series is stationary')
        print('p-value = ', adfuller(data)[1])
    else:
        print('The series is NOT stationary')
        print('p-value = ', adfuller(data)[1])
        
        