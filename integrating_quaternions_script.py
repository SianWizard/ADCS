# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:27:14 2023

@author: sina
"""

import numpy as np
import math
import Quaternion_funcs as qf



w = lambda t: np.array([20*3.14/180*math.sin(0.1*t),20*3.14/180*0.01,20*3.14/180*math.cos(0.1*t)])
b0 = np.array([0.408248,0,0.408248,0.816497])
time = np.arange(0.,43.0,0.01)
time_length = time.size
b_series = np.zeros((4,time_length))
b_series[0:4,0]=b0
for i in range(0,time_length):
    if i == 0: continue
    w_t0 = w(time[i-1])
    b_t0 = b_series[:,i-1]
    delta_t = time[i]-time[i-1]
    db = qf.EP_diffEq(b_t0, w_t0)
    b_t = b_t0 + db*delta_t
    b_series[:,i]=b_t

b42 = b_series[:,np.where(time==42)[0]]

print('The sum is:')

print((b42[1]**2+b42[2]**2+b42[3]**2)**0.5)