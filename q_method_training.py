# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:54:53 2023

@author: sina
"""

import numpy as np
import Attitude_det_funcs as atdet

b = np.column_stack((np.array([0.8273,0.5541,-0.0920]),np.array([-0.8285,0.5522,-0.0955])))
n = np.column_stack((np.array([-0.1517,-0.9669,0.2050]),np.array([-0.8393,0.4494,-0.3044])))
w = np.array([1,1])

dcm = atdet.q_method(b, n, w)

dcm2 = atdet.QUEST_method(b, n, w) # not super accurate yet