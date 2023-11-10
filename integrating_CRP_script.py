# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:06:35 2023

@author: sina
"""
import math
import CRP_funcs as crp
import numpy as np


w = lambda t: 3*3.14/180*np.array([math.sin(0.1*t),0.01,math.cos(0.1*t)])
q0 = np.array([0.4,0.2,-0.1])
time = np.arange(0.,43.0,0.01)
time_length = time.size
q_series = np.zeros((3,time_length))
q_series[:,0]=q0
for i in range(0,time_length):
    if i == 0: continue
    w_t0 = w(time[i-1])
    q_t0 = q_series[:,i-1]
    delta_t = time[i]-time[i-1]
    dq = crp.CRP_diffEq(q_t0, w_t0)
    q_t = q_t0 + dq*delta_t
    q_series[:,i]=q_t

q42 = q_series[:,np.where(time==42)[0]]

print('The sum is:')

print((q42[0]**2+q42[1]**2+q42[2]**2)**0.5)