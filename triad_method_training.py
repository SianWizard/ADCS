# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:55:32 2023

@author: sina
"""

import numpy as np
import Attitude_det_funcs as atdet
from PRV_funcs import DCM2PRV


b1 = np.array((0.8273,0.5541,-0.0920))
b2 = np.array((-0.8285,0.5522,-0.0955))

n1 = np.array((-0.1517,-0.9669,0.2050))
n2 = np.array((-0.8393,0.4494,-0.3044))

BN_est = atdet.triad(b1, b2, n1, n2)

orthogonality = np.matmul(BN_est,np.transpose(BN_est))

BN_est_2 = np.array(([0.969846,0.17101,0.173648],[-0.200706,0.96461,0.17101],[-0.138258,-0.200706,0.969846]))
BN_tru_2 = np.array(([0.963592,0.187303,0.190809],[-0.223042,0.956645,0.187303],[-0.147454,-0.223042,0.963592]))

phis,es = DCM2PRV(np.matmul(BN_est_2,np.transpose(BN_tru_2)))

phis_deg = 180/3.1415 * phis