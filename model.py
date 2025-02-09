# This python code contains the main functions for the computational model

# Copyright 2024 Heiko Hoffmann

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import numpy as np
import random
import sys


# Create a binary population code with N neurons and tuning width sigma, encoding the variable x
def create_code(x, sigma, N):
    xmin = -1.0
    xrange = 2.0
    sqrsigma2 = 2 * sigma**2
    C = 2 / (N * sigma * math.sqrt(2*math.pi)) # normalization constant

    # Calculate probabilities:
    code = np.zeros(N)
    for i in range(N):
        mean = xmin + i * xrange / (N - 1) # preferred value of neuron i
        dx = abs(x - mean)
        # Use periodic boundary condition
        if dx > 1: 
            dx = 2 - dx
        p = C*math.exp(- dx**2 / sqrsigma2)
        if random.random() < p:
            code[i] = 1
            
    return code

# Subtract two population-coded variables and return the resulting code
def compute(code1, code2):
    N = code1.shape[0]
    if code2.shape[0] != N:
        sys.exit("Error: Both codes have to be of the same size (it's possible to adapt this program to allow different sizes)")
        
    res = np.zeros(N)
    N2 = int(N/2)

    # Loop through all connections to compute the resulting code
    for i in range(N):
        if code1[i] == 1:
            x = i - N2
            for j in range(N):
                if code2[j] == 1:
                    y = j - N2
                    z = (x - y + N2) % N # using periodic boundary
                    res[z] = 1

    return res


# Add two population-coded variables and return the resulting code
def compute_add(code1, code2):
    N = code1.shape[0]
    if code2.shape[0] != N:
        sys.exit("Error: Both codes have to be of the same size (it's possible to adapt this program to allow different sizes)")
        
    res = np.zeros(N)
    N2 = int(N/2)

    # Loop through all connections to compute the resulting code
    for i in range(N):
        if code1[i] == 1:
            x = i - N2
            for j in range(N):
                if code2[j] == 1:
                    y = j - N2
                    z = (x + y + N2) % N # using periodic boundary
                    res[z] = 1

    return res


# Multiply two population-coded variables and return the resulting code
def compute_mult(code1, code2):
    N = code1.shape[0]
    if code2.shape[0] != N:
        sys.exit("Error: Both codes have to be of the same size (it's possible to adapt this program to allow different sizes)")
        
    res = np.zeros(N)
    N2 = int(N/2)

    # Loop through all connections to compute the resulting code
    for i in range(N):
        if code1[i] == 1:
            x = i - N2
            for j in range(N):
                if code2[j] == 1:
                    y = j - N2
                    z = ((x*y) // N2 + N2) % N # using periodic boundary
                    res[z] = 1

    return res


# Add three population codes and return the composition code
def compute3(code1, code2, code3):
    N = code1.shape[0]
    if code2.shape[0] != N or code3.shape[0] != N:
        sys.exit("Error: All three codes have to be of the same size (it's possible to adapt this program to allow different sizes)")
        
    res = np.zeros(N)
    N2 = int(N/2)

    # Loop through all connections to compute the composition code
    for i in range(N):
        if code1[i] == 1:
            x = i - N2
            for j in range(N):
                if code2[j] == 1:
                    y = j - N2
                    for k in range(N):
                        if code3[k] == 1:
                            z = k - N2
                            s = (x + y + z + N2) % N # using periodic boundary
                            res[s] = 1

    return res

