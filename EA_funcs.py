# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:38:02 2024

@author: sina
"""
# Euler angles functions 

from numpy import array,cos,sin

def Rot(axis,x):
    if axis == 1:
        Rot = array([[1,0,0],
                     [0,cos(x),sin(x)],
                     [0,-sin(x),cos(x)]])
    elif axis == 2:
        Rot = array([[cos(x),0,-sin(x)],
                     [0,1,0],
                     [sin(x),0,cos(x)]])
    else:
        Rot = array([[cos(x),sin(x),0],
                     [-sin(x),cos(x),0],
                     [0,0,1]])
    return Rot

def EA2DCM(order,an):
    dcm  = Rot(order[2],an[2])@Rot(order[1],an[1])@Rot(order[0],an[0])
    return dcm