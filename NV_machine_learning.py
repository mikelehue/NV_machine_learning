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

# These are the projectors over the X, Y, and Z axes of the Bloch sphere.
# They are used to calculate the Bloch coordinates of a density matrix.
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

# Here we calculate the fidelity between two density matrices.
def Fidelity(rho, target, pure=True):
    
    if pure:
        f = np.trace(np.matmul(rho*target))
    else:
        f = np.matmul(sqrtm(target)*rho)
        f = sqrtm(np.matmul(f*sqrtm(target)))
        f = (np.trace(f))**2
    
    return f

# This function calculates the distance between two points in thhe coordinate plane
def Distance(x_1, x_2):
    return float(np.sqrt((float(x_1[0])-float(x_2[0]))**2+(float(x_1[1])-float(x_2[1]))**2+(float(x_1[2])-float(x_2[2]))**2))

def Centroide (x):
    centroid = []
    centroid= sum(x)
    centroid= centroid /len(x)

    return np.array(centroid)

# This function tests the quantum states against the labels and returns the maximum fidelities and their corresponding indices.
def Test(quantum_states, labels):
    fidelities = np.matmul(quantum_states,labels)
    fidelities = fidelities*np.conj(fidelities)
    first_label = fidelities[0]
    second_label = fidelities[1]
    max_fidelities = [np.max([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]
    arg_fidelities = [np.argmax([first_label[i], second_label[i]]) for i in range(len(fidelities[0]))]

    return max_fidelities, arg_fidelities

# This function computes the cost function for a batch of quantum states, coordinates, and labels.
def CostFunction(quantum_states, coordinates, labels,_lambda=0):
    # Compute prediction for each input in data batch
    loss = 0  # initialize loss
    fidelities, arg_fidelities = Test(quantum_states,labels)
    coordinates = np.array(coordinates)
    arg_fidelities = np.array(arg_fidelities)
    for i in range(len(coordinates)):
        f_i = fidelities[i]
        for j in range(i + 1, (len(coordinates))):
            f_j = fidelities[j]
            if arg_fidelities[i] == arg_fidelities[j]:
                delta_y = 1
            else:
                delta_y = 0
            loss = loss + delta_y * (Distance(coordinates[i], coordinates[j]) + _lambda*Distance(coordinates[i], Centroide(coordinates[np.where(arg_fidelities == arg_fidelities[i])]))) * ((1 - f_i) * (1 - f_j))
    
    return np.real(loss / len(coordinates))