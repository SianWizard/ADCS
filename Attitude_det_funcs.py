# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:44:19 2023

@author: sina
"""
import numpy as np
from Quaternion_funcs import EP2DCM

# Triad method
def triad (b1,b2,n1,n2):
    #b1 is the more accurate measurement
    b1 = b1/np.linalg.norm(b1)
    b2 = b2/np.linalg.norm(b2)
    n1 = n1/np.linalg.norm(n1)
    n2 = n2/np.linalg.norm(n2)
    
    t1b = b1
    t2b = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
    t3b = np.cross(t1b,t2b)
    
    t1n = n1
    t2n = np.cross(n1,n2)/np.linalg.norm(np.cross(n1,n2))
    t3n = np.cross(t1n,t2n)
    
    BT = np.column_stack((t1b,t2b,t3b))
    NT = np.column_stack((t1n,t2n,t3n))
    
    BN = np.matmul(BT,np.transpose(NT))
    
    #BN = 3/2*BN - 1/2*np.matmul(np.matmul(BN,np.transpose(BN)),BN)
    
    return BN

def q_method (b,n,w): # The matrices should be column stacked 
    #no_vecs = b.shape[1]
    no_vecs = w.size
    
    B = np.zeros((3,3))
    
    for i in range(0,no_vecs):
        vb = b[:,i]
        vn = n[:,i]
        
        vb = vb.reshape((3,1))
        vn = vn.reshape((3,1))
        
        B = B + w[i]*np.matmul(vb,np.transpose(vn))
        
    sigma = np.trace(B)
    S = B + np.transpose(B)
    Z = np.transpose(np.array([[B[1][2]-B[2][1],B[2][0]-B[0][2],B[0][1]-B[1][0]]]))

    row1 = np.hstack((sigma.reshape((1,1)),np.transpose(Z)))
    temp = S-sigma*np.eye(3)
    row2 = np.hstack((Z,temp))
    
    K = np.vstack((row1,row2))

    eigVal,eigVec = np.linalg.eig(K)
    
    ind_max = np.where(eigVal==max(eigVal))[0][0]

    beta_set = eigVec[:,ind_max]
    
    if beta_set[0]<0:
        beta_set = -1*beta_set
    
    print('Quaternion set used: \n',beta_set)
    return EP2DCM(beta_set)
        
def QUEST_method (b,n,w):
    no_vecs = w.size
    
    B = np.zeros((3,3))
    
    for i in range(0,no_vecs):
        vb = b[:,i]
        vn = n[:,i]
        
        vb = vb.reshape((3,1))
        vn = vn.reshape((3,1))
        
        B = B + w[i]*np.matmul(vb,np.transpose(vn))
        
    sigma = np.trace(B)
    S = B + np.transpose(B)
    Z = np.transpose(np.array([[B[1][2]-B[2][1],B[2][0]-B[0][2],B[0][1]-B[1][0]]]))

    row1 = np.hstack((sigma.reshape((1,1)),np.transpose(Z)))
    temp = S-sigma*np.eye(3)
    row2 = np.hstack((Z,temp))
    
    K = np.vstack((row1,row2))
    
    #abs_error = 1.0e-12
    
    #f = lambda s: np.linalg.det(K-s*np.eye(4))
    
    l0 = np.sum(w)
    
    print(l0,'\n')
    
    # Not doing any lamda optimization
    
    l = l0
    
    q = np.matmul(np.linalg.inv((l+sigma)*np.eye(3)-S),Z)
    print(q)
    beta = 1/(1+np.matmul(np.transpose(q),q))**0.5 * np.vstack((np.array([[1]]),q))
    beta = beta.reshape((4,))
    print('Quaternion set used: \n',beta)
    return EP2DCM(beta)