#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: NV_machine_learning.py
Description: This script simulates a one qubit cluster sorter by means of a NV center on diamond.
Authors: Asier Mongelos and Miguel Lopez Varga
Creation date: 2025-05-20
"""





import numpy as np
from scipy.linalg import expm, sqrtm

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



def ProjX(rho):
    ProjX = np.array([[1/2,1/2,0],[1/2,1/2,0],[0,0,0]])
    
    return np.trace(np.matmul(rho*ProjX))


def ProjY(rho):
    ProjY = np.array([[1/2,1j/2,0],[-1j/2,1/2,0],[0,0,0]])
    
    return np.trace(np.matmul(rho*ProjY))


def ProjZ(rho):
    return rho[1,1]




def BlochCoordinates(rho):
    x = ProjX(rho)
    y = ProjY(rho)
    z = ProjZ(rho)
    
    return np.array([x,y,z])



def Fidelity(rho, target, pure=True):
    
    if pure:
        f = np.trace(np.matmul(rho*target))
    else:
        f = np.matmul(sqrtm(target)*rho)
        f = sqrtm(np.matmul(f*sqrtm(target)))
        f = (np.trace(f))**2
    
    return f
    
    


