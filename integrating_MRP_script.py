# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:27:14 2023

@author: sina
"""

import numpy as np
import MRP_funcs as mrpf

# s1 = np.array([0.1,0.2,0.3])

# dcm1 = mrpf.MRP2DCM(s1)

# dcm2 = np.array([[0.763314,0.0946746,-0.639053],[-0.568047,-0.372781,-0.733728],[-0.307692,0.923077,-0.230769]])

# mrp2 = mrpf.DCM2MRP(dcm2)

# s_BN1 = np.array([0.1,0.2,0.3])
# s_RB1 = np.array([-0.1,0.3,0.1])

# s_RN1 = mrpf.MRP_sum(s_BN1, s_RB1)

# s_BN2 = np.array([0.1,0.2,0.3])
# s_RN2 = np.array([0.5,0.3,0.1])

# s_NR2 = -s_RN2

# s_BR2= mrpf.MRP_sum(s_NR2, s_BN2)

# Integration part

import math

w = lambda t: 20*3.14/180*np.array([math.sin(0.1*t),0.01,math.cos(0.1*t)])
s0 = np.array([0.4,0.2,-0.1])
time = np.arange(0.,42.0,0.01)
time_length = time.size
s_series = np.zeros((3,time_length))
s_series[:,0]=s0
for i in range(0,time_length):
    if i == 0: continue
    w_t0 = w(time[i-1])
    s_t0 = s_series[:,i-1]
    
    if np.dot(s_t0,s_t0)>1:
        s_t0 = -s_t0/np.dot(s_t0,s_t0)
        
    delta_t = time[i]-time[i-1]
    ds = mrpf.MRP_diffEq(s_t0, w_t0)
    s_t = s_t0 + ds*delta_t
    
    if np.dot(s_t,s_t)>1:
        s_t = -s_t/np.dot(s_t,s_t)
        
    
    s_series[:,i]=s_t

s42 = s_series[:,-1]


print('The sum is:')

print((s42[0]**2+s42[1]**2+s42[2]**2)**0.5)