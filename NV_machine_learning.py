#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: NV_machine_learning.py
Description: This script simulates a one qubit cluster sorter by means of a NV center on diamond.
Authors: Asier Mongelos and Miguel Lopez Varga
Creation date: 2025-05-20
"""





import numpy as np
from scipy.linalg import expm

##Operators##
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
Sx = np.array([[0,1,0],[1,0,1],[0,1,0]])
Sy = np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])

#We define the rotation operations over the N Vcenter
##Each function retunrs th eunitary operation representing the rotation over
#states 0 and 1 of the NV. The pulses are then considered to be in tune
# with the transition 0--1. 
#Functions tae the arguments:
    # alpha    --> Rotation angle in Radians
    # Omega    --> MW amplitude in MHz
    # B        --> magnetic field value in T (tesla)
    # detuning --> error in targetin gthe resonance line (default =0) in MHz.
#
def Rx(alpha, Omega, B, detuning=0):
    #Physical parameters
    D = 2870        #MHz, Zer-field splitting.
    gamma_e = 28024 #MHz/T electron gyromagnetic ratio
    
    H = D*np.matmul(Sz,Sz) + gamma_e*B*Sz + (Omega/2)*Sx -(D+detuning)*np.matmul(Sz,Sz)
    
    t = alpha/(np.pi*Omega)
    
    U = expm(1j*H*t)
    return U


def Ry(alpha, Omega, B, detuning=0):
    #Physical parameters
    D = 2870        #MHz, Zer-field splitting.
    gamma_e = 28024 #MHz/T electron gyromagnetic ratio
    
    H = D*np.matmul(Sz,Sz) + gamma_e*B*Sz + (Omega/2)*Sy -(D+detuning)*np.matmul(Sz,Sz)
    
    t = alpha/(np.pi*Omega)
    
    U = expm(1j*H*t)
    return U


