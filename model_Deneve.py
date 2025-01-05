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

# The neural network has been coded based on the description in the
# Supplementary Information of the article
# Deneve S, Latham PE, Pouget A. 2001. Efficient computation and cue
# integration with noisy population codes. Nat Neurosci. 4(8):826–831


import math
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork:
    def __init__(self, N=40):
        self.N = N
        self.input_units = N
        self.hidden_layer = (N, N)

        # Parameters (as mentioned in supplementary info of Deneve et al, 2001)
        self.Kw = 1
        self.sigma_w = 0.37
        self.S = 0.1
        self.mu = 0.002
        
        # Initialize weights
        self.init_weights()
        
        # Initialize activities
        self.A_lm = np.zeros(self.hidden_layer)
        self.R_rj = np.zeros(self.input_units)
        self.R_ej = np.zeros(self.input_units)
        self.R_aj = np.zeros(self.input_units)
        

    # Weight initialization (as described in supplementary info of Deneve et al, 2001)
    def init_weights(self):
        self.W_r = np.zeros((self.input_units, *self.hidden_layer))
        self.W_e = np.zeros((self.input_units, *self.hidden_layer))
        self.W_a = np.zeros((self.input_units, *self.hidden_layer))
        for j in range(self.input_units):
            for l in range(0, self.N):
                for m in range(0, self.N):
                    # Weights from the first base code to the intermediate layer
                    self.W_r[j, l, m] = self.Kw * np.exp(
                        (np.cos((2 * np.pi / self.N) * (j - l)) - 1) / self.sigma_w**2
                    )
                    # Weights from the second base code to the intermediate layer
                    self.W_e[j, l, m] = self.Kw * np.exp(
                        (np.cos((2 * np.pi / self.N) * (j - m)) - 1) / self.sigma_w**2
                    )
                    # Weights for the subtraction operation (the different sign between l and m
                    # takes care of the subtraction)
                    self.W_a[j, l, m] = self.Kw * np.exp(
                        (np.cos((2 * np.pi / self.N) * (self.N // 2 + j - l + m)) - 1) / self.sigma_w**2
                    )
 

    def plot3D(self, data):
        x = np.arange(data.shape[1])  # Number of columns
        y = np.arange(data.shape[0])  # Number of rows
        X, Y = np.meshgrid(x, y)
        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        ax.plot_surface(X, Y, data, cmap='viridis')
        plt.show()

    # Network dynamics (as described in supplementary info of Deneve et al, 2001)
    def evolve(self, iterations=3):
        L_lm = np.zeros(self.hidden_layer)
        for _ in range(iterations):
            # Update L_lm
            for l in range(self.hidden_layer[0]):
                for m in range(self.hidden_layer[1]):
                    L_lm[l, m] = (
                        np.sum(self.W_r[:, l, m] * self.R_rj) +
                        np.sum(self.W_e[:, l, m] * self.R_ej) +
                        np.sum(self.W_a[:, l, m] * self.R_aj)
                    )
                    
            # Update A_lm
            self.A_lm = L_lm**2 / (self.S + self.mu * np.sum(L_lm**2))
            
            # Update R_rj, R_ej, R_aj
            for j in range(self.input_units):
                self.R_rj[j] = (
                    np.sum(self.W_r[j, :, :] * self.A_lm)**2 /
                    (self.S + self.mu * np.sum(np.sum(self.W_r[j, :, :] * self.A_lm)**2))
                )
                self.R_ej[j] = (
                    np.sum(self.W_e[j, :, :] * self.A_lm)**2 /
                    (self.S + self.mu * np.sum(np.sum(self.W_e[j, :, :] * self.A_lm)**2))
                )
                self.R_aj[j] = (
                    np.sum(self.W_a[j, :, :] * self.A_lm)**2 /
                    (self.S + self.mu * np.sum(np.sum(self.W_a[j, :, :] * self.A_lm)**2))
                )

    def set_input(self, code1, code2):
        self.R_rj = code1.copy()
        self.R_ej = code2.copy()

    # Output function adapted to produce spiking (binary) activity
    def get_output(self):
        threshold = 5
        activity = self.R_aj - np.min(self.R_aj)
        result = np.where(activity < threshold, 0, 1)
        return result

    # Subtract two population-coded variables and return the resulting code
    def compute(self, code1, code2):
        self.set_input(code1, code2)
        self.R_aj = np.zeros(self.input_units)
        self.evolve(1)
        return self.get_output()
    

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



