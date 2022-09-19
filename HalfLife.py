# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:13:33 2022

@author: sigma
"""

import numpy as np

def estimate_half_life(spread):
    x=spread.shift().iloc[1:].to_frame().assign(const=1)
    y=spread.diff().iloc[1:]
    beta=(np.linalg.inv(x.T@x)@x.T@y).iloc[0]
    halflife=int(round(-np.log(2)/beta,0))
    return max(halflife,1)

