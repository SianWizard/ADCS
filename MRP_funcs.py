# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:05:50 2023

@author: sina
"""
# MRP_funcs

import numpy as np
import Quaternion_funcs as qf

def SkewSym(a):
    A = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return A

def MRP2DCM (s):
    C = np.eye(3)+(1/(1+np.dot(s,s))**2)*((8*np.matmul(SkewSym(s),SkewSym(s))-4*(1-np.dot(s,s))*SkewSym(s)))
    return C
    
def EP2MRP(ep):
    MRP_set = np.zeros((2,3))
    for i in [0,1]:
        mrp1 = ep[i][1]/(1+ep[i][0])
        mrp2 = ep[i][2]/(1+ep[i][0])
        mrp3 = ep[i][3]/(1+ep[i][0])
        MRP_set[i,:]=np.array([mrp1,mrp2,mrp3])
    return MRP_set

def DCM2MRP(dcm):
    ep = qf.DCM2EP(dcm)
    mrp_set = EP2MRP(ep)
    return mrp_set
    
def MRP_sum(s_BN,s_RB): # to get s_RN
    s1 = s_BN.reshape((3,))
    s2 = s_RB.reshape((3,))
    
    denum = 1 + np.dot(s1,s1)*np.dot(s2,s2)-2*np.dot(s1,s2)
    
    if np.abs(denum)<0.001:
        if (np.dot(s1,s1))**0.5>1:
            s1 = -(1/np.dot(s1,s1))*s1
        else:
            s2 = -(1/np.dot(s2,s2))*s2
        denum = 1 + np.dot(s1,s1)*np.dot(s2,s2)-2*np.dot(s1,s2)
        
    num = (1-np.dot(s1,s1))*s2+(1-np.dot(s2,s2))*s1-2*np.cross(s2,s1)
    s_RN = num/denum
    return s_RN
    
def MRP_diffEq(s,w):
    s_temp = np.array([[s[0],s[1],s[2]]]);
    B = (1-np.linalg.norm(s)**2)*np.eye(3)+2*SkewSym(s)+2*np.matmul(np.transpose(s_temp),s_temp) ## Such a hassle to transpose a 1D array!
    ds = 1/4*np.matmul(B,w)
    return ds