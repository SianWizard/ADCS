# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:51:51 2023

@author: sina
"""

# This program is for Euler Parameters (EP - Quaternions)

import numpy as np

def EP2DCM (ep):
    c11 = ep[0]**2+ep[1]**2-ep[2]**2-ep[3]**2
    c12 = 2*(ep[1]*ep[2]+ep[0]*ep[3])
    c13 = 2*(ep[1]*ep[3]-ep[0]*ep[2])
    c21 = 2*(ep[1]*ep[2]-ep[0]*ep[3])
    c22 = ep[0]**2-ep[1]**2+ep[2]**2-ep[3]**2
    c23 = 2*(ep[3]*ep[2]+ep[0]*ep[1])
    c31 = 2*(ep[1]*ep[3]+ep[0]*ep[2])
    c32 = 2*(ep[3]*ep[2]-ep[0]*ep[1])
    c33 = ep[0]**2-ep[1]**2-ep[2]**2+ep[3]**2
    
    C = np.array([[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]])
    
    return C

def DCM2EP (dcm):
    tr = dcm[0][0]+dcm[1][1]+dcm[2][2]
    b0_2 = 1/4*(1+tr)
    b1_2 = 1/4*(1+2*dcm[0][0]-tr)
    b2_2 = 1/4*(1+2*dcm[1][1]-tr)
    b3_2 = 1/4*(1+2*dcm[2][2]-tr)
    
    b_2 = [b0_2,b1_2,b2_2,b3_2]
    
    max_b2 = max(b_2)
    index_max_b2 = b_2.index(max_b2)
    
    b = np.zeros((2,4))
    
    if index_max_b2 == 0:
        b0_values = [max_b2**0.5,-1*max_b2**0.5]
        for i in [0,1]:
            b0 = b0_values[i]
            b1 = (dcm[1][2]-dcm[2][1])/(4*b0)
            b2 = (dcm[2][0]-dcm[0][2])/(4*b0)
            b3 = (dcm[0][1]-dcm[1][0])/(4*b0)
            b[i][:] = np.array([b0,b1,b2,b3])
            
    elif index_max_b2 == 1:
        b1_values = [max_b2**0.5,-1*max_b2**0.5]
        for i in [0,1]:
            b1 = b1_values[i]
            b0 = (dcm[1][2]-dcm[2][1])/(4*b1)
            b2 = (dcm[1][0]+dcm[0][1])/(4*b1)
            b3 = (dcm[0][2]+dcm[2][0])/(4*b1)
            b[i][:] = np.array([b0,b1,b2,b3])
            
    elif index_max_b2 == 2:
        b2_values = [max_b2**0.5,-1*max_b2**0.5]
        for i in [0,1]:
            b2 = b2_values[i]
            b0 = (dcm[2][0]-dcm[0][2])/(4*b2)
            b1 = (dcm[1][0]+dcm[0][1])/(4*b2)
            b3 = (dcm[1][2]+dcm[2][1])/(4*b2)
            b[i][:] = np.array([b0,b1,b2,b3])
            
    else:
        b3_values = [max_b2**0.5,-1*max_b2**0.5]
        for i in [0,1]:
            b3 = b3_values[i]
            b0 = (dcm[0][1]-dcm[1][0])/(4*b3)
            b1 = (dcm[2][0]+dcm[0][2])/(4*b3)
            b2 = (dcm[1][2]+dcm[2][1])/(4*b3)
            b[i][:] = np.array([b0,b1,b2,b3])

    
    return b # gives 2 sets


def EP_sum (BN,FB): # for FN = FB + BN in quaternions
    B1 = BN
    B2 = FB
    
    B1 = B1.reshape((4,))
        
    matB2 = np.array([[B2[0],-B2[1],-B2[2],-B2[3]],[B2[1],B2[0],B2[3],-B2[2]],[B2[2],-B2[3],B2[0],B2[1]],[B2[3],B2[2],-B2[1],B2[0]]])
    FN = np.matmul(matB2,B1)
    return FN

def EP_sub (FN,BN): # for FN - BN = FB in quaternions
    B1 = FN
    B2 = BN
    B1 = B1.reshape((4,))
        
    matB2 = np.array([[B2[0],-B2[1],-B2[2],-B2[3]],[B2[1],B2[0],-B2[3],B2[2]],[B2[2],B2[3],B2[0],-B2[1]],[B2[3],-B2[2],B2[1],B2[0]]])
    b = np.matmul(np.transpose(matB2),B1)
    return b

def EP_diffEq (b,w):
    w = w.reshape((3,))
    bMat = np.array([\
                     [-b[1],-b[2],-b[3]],\
                     [b[0],-b[3],b[2]],\
                     [b[3],b[0],-b[1]],\
                     [-b[2],b[1],b[0]]])
    db = 0.5*np.matmul(bMat,w)
    return db