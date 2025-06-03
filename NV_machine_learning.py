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

# This function calculates the centroid of a set of coordinates.
def Centroide (x):
    centroid = []
    centroid= sum(x)
    centroid= centroid /len(x)

    return np.array(centroid)

# This function converts Cartesian coordinates to coefficients for the quantum state.
def CartToCoef(points_x, points_y, points_z):
    phi_once = np.arctan2(points_y, points_x)
    theta_once = np.arccos(points_z)
    c1_once = np.cos(theta_once / 2)
    c2_once = np.exp(complex(0, 1) * phi_once) * np.sin(theta_once / 2)

    return np.array(c1_once, dtype=complex), np.array(c2_once, dtype=complex)

# This function generates points on a sphere using the Fibonacci spiral method.
def FibonacciSphere(samples):
    points_x = []
    points_y = []
    points_z = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points_x.append(x)
        points_y.append(y)
        points_z.append(z)
    return points_x, points_y, points_z

def StateLabels(number_labels):
    labels = []
    points_x_labels, points_y_labels, points_z_labels = FibonacciSphere(number_labels)
    for i in range(len(points_x_labels)):
        c1_round, c2_round = CartToCoef(points_x_labels[i], points_y_labels[i], points_z_labels[i])
        labels.append([[c1_round], [c2_round]])
    state_labels = np.array(labels, dtype=complex)
    return state_labels

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
def CostFunction(quantum_states, coordinates, labels,_lambda):
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

############### Simulation of the cluster sorter ################
############### Data for the simulation ################
# Number of labels
number_labels = 4
# Initialize the labels
labels = StateLabels(number_labels)
# Choose the number of times the whole set of gates is applied
number_iterations = 50
# Choose the step for calculate the gradient
st = 0.01
# Choose the value of the learning rate
lr = 0.006
# Choose the value of lambda for the cost function
_lambda = 1
# Choose the magnetic field value in Tesla
B = 0.1
# Choose the detuning value in MHz
detuning = 0
# Choose the amplitude of the microwave pulse in MHz
Omega = 10

###############################################################
# Initialize data set
np.random.seed(123)
# Set up cluster parameters
number_clusters = 4
points_per_cluster = 10
N_points = number_clusters * points_per_cluster
centers = [(-0.29*np.pi, 0*np.pi), (0*np.pi, 0.12*np.pi), (0.29*np.pi, 0*np.pi), (0*np.pi, -0.12*np.pi)]
width = 0.075
# Initialize arrays for coordinates
coordinates = []

# Generate points within clusters
for i in range(number_clusters):
    # Generate points within current cluster
    for j in range(points_per_cluster):
        # Generate point with Gaussian distribution
        point = np.random.normal(loc=centers[i], scale=width)
        coordinates.append([point[0], point[1], 0])
# Convert coordinates to numpy array
coordinates = np.array(coordinates)
# Initialize quantum states
quantum_states = np.zeros((N_points, 2, 1), dtype=complex)
# Convert coordinates to quantum states
# The idea is to start with a quantum state in the zero state of the bloch sphere
# That is the state |0> = [1, 0, 0] and the we have to apply the gates to rotate the state to the desired point in the bloch sphere
# The gates are the Rx and Ry functions defined above
# So we start with the state |0>
initial_state = np.array([[1], [0], [0]], dtype=complex)