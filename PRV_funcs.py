# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:05:32 2023

@author: sina
"""

import numpy as np
import math

def DCM2PRV (dcm):
    phi = math.acos(0.5*(dcm[0][0]+dcm[1][1]+dcm[2][2]-1))
    phi_p = phi - 2*math.pi
    
    set1 = 1/math.sin(phi)*np.array((dcm[1][2]-dcm[2][1],dcm[2][0]-dcm[0][2],dcm[0][1]-dcm[1][0]))
    set2 = 1/math.sin(phi_p)*np.array((dcm[1][2]-dcm[2][1],dcm[2][0]-dcm[0][2],dcm[0][1]-dcm[1][0]))
    
    phis = np.row_stack((phi,phi_p))
    es = np.row_stack((set1,set2))
    return phis,es