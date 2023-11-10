# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:55:43 2023

@author: sina
"""
# Classical Rodriguez Parameter (CRP) parameters

import numpy as np

def CRP2DCM(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    
    C = 1/(1+np.matmul(np.transpose(q),q))*\
        np.array([[1+q1**2-q2**2-q3**2,2*(q1*q2+q3),2*(q1*q3-q2)],\
                  [2*(q2*q1-q3),1-q1**2+q2**2-q3**2,2*(q2*q3+q1)],\
                  [2*(q3*q1+q2),2*(q3*q2-q1),1-q1**2-q2**2+q3**2]])
    return C

def EP2CRP(ep):
    q = np.zeros((3,))
    q[0] = ep[1]/ep[0]
    q[1] = ep[2]/ep[0]
    q[2] = ep[3]/ep[0]
    return q

def DCM2CRP(dcm):
    from Quaternion_funcs import DCM2EP
    ep = DCM2EP(dcm)
    ep_one_set = ep[1,:]
    q = EP2CRP(ep_one_set)
    return q

def CRP_sum(q_BN,q_FB): #to find q_FB
    q_FB = 1/(1-np.dot(q_BN, q_FB))*\
        (q_BN+q_FB-np.cross(q_FB, q_BN))
    return q_FB

def CRP_diffEq (q,w):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    
    w = w.reshape((3,))
    
    Qmat = np.array([[1+q1**2,q1*q2-q3,q1*q3+q2],[q2*q1+q3,1+q2**2,q2*q3-q1],[q3*q1-q2,q3*q2+q1,1+q3**2]])
    
    dq = 0.5*np.matmul(Qmat,w)
    
    return dq