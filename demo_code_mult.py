# This python code demonstrates the computational model for multiplying
# two population-encoded variables.

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


import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from scipy.integrate import quad
from tqdm import tqdm
import model


# Parameters for the simulation
N = 51 # Population code size
num_steps = 2000000 # Number of simulation time steps
histogram_steps = 1000 # Initial number of steps used for the histogram
a = 0.5 # Variable encoded in code A
a_sig = 0.25 # Tuning standard deviation for code A
b = -0.5 # Variable encoded in code B
b_sig = 0.25 # Tuning standard deviation for code B
firing_rate_bin = 500 # Bin size for computing firing rate


# Compute distribution of firing rates
def firing_rate(spike_train, bin_size):
    N = spike_train.shape[0]
    num_steps = spike_train.shape[1]
    firing_rates = []
    
    for i in range(N):
        spike_binary = spike_train[i,:]
        
        # Calculate the number of bins
        num_bins = len(spike_binary) // bin_size

        # Truncate the array to ensure it's evenly divisible by bin_size
        arr = spike_binary[:num_bins * bin_size]

        # Reshape the array into a 2D array with each row representing a bin
        split_arr = arr.reshape((num_bins, bin_size))

        # Sum each row (bin) along axis 1 to get an array of sums
        bin_sums = np.sum(split_arr, axis=1)

        firing_rates.append(bin_sums/bin_size)
        
    return firing_rates


# Define the integrand as a function of x and z
def integrand(x, z):
    Ca = 1.0 / (a_sig * math.sqrt(2*math.pi))
    Cb = 1.0 / (b_sig * math.sqrt(2*math.pi))
    return np.exp(- ((x - a)**2)/(2 * a_sig**2) - ((z / (x+1e-20) - b)**2)/(2 * b_sig**2) ) * Ca * Cb

# Define the integral as a function of z
def integral_function(z, x_min = -2, x_max = 2):
    result, _ = quad(integrand, x_min, x_max, args=(z,))
    return result

# Compute the theoretically expected activity distribution for a multiplication population code
def theoretical_code_mult():
    norm, _ = quad(integral_function, -5, 5)
    z_values = np.linspace(-1, 1, N)  # Range of z
    integral_values = np.array([integral_function(z) for z in z_values])/norm
    return integral_values

# Compute the theoretically expected activity distribution for a population code
def theoretical_code(x, mu, sigma):
    C = 1.0 / (sigma * math.sqrt(2*math.pi))
    dx = x - mu
    return np.exp(- dx*dx / (2 * sigma**2))*C
    

# Initialize population codes A, B, and C
A = np.zeros((N, num_steps))
B = np.zeros((N, num_steps))
C = np.zeros((N, num_steps))

# Loop through simulation time steps
for t in tqdm(range(num_steps)):
    A[:,t] = model.create_code(a,a_sig,N)
    B[:,t] = model.create_code(b,b_sig,N)
    C[:,t] = model.compute_mult(A[:,t], B[:,t])

# Compute total activation
A_act = np.sum(A)
B_act = np.sum(B)
C_act = np.sum(C)

print(f"A total activation: {A_act}")
print(f"B total activation: {B_act}")
print(f"C total activation: {C_act}")
    
# Compute histograms
x = np.arange(0, N)/((N-1)/2)-1
hA = np.sum(A[:,0:histogram_steps], axis=1)/histogram_steps
hB = np.sum(B[:,0:histogram_steps], axis=1)/histogram_steps
hC = np.sum(C[:,0:histogram_steps], axis=1)/histogram_steps

rf_A = firing_rate(A,firing_rate_bin)
rf_B = firing_rate(B,firing_rate_bin)
rf_C = firing_rate(C,firing_rate_bin)

theo_A = theoretical_code(x, a, a_sig)/N*2
theo_B = theoretical_code(x, b, b_sig)/N*2
theo_C = theoretical_code_mult()/N*2

# Create figure of spike trains
fig, axes = plt.subplots(3, 1, figsize=(6, 6))
plot_steps = 201 # Number of visualized simulation steps for the spike-train plots

# Plot each spike train in a separate subplot
axes[0].imshow(A[:,:plot_steps], cmap='gray_r', interpolation='nearest', aspect=0.8)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_title('Code A', fontsize=16)
axes[0].set_ylabel('Neuron ID', fontsize=14)
axes[0].tick_params(axis='both', which='both', labelsize=14)
axes[0].xaxis.set_major_locator(MultipleLocator(100))
axes[0].yaxis.set_major_locator(MultipleLocator(25))

axes[1].imshow(B[:,:plot_steps], cmap='gray_r', interpolation='nearest', aspect=0.8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Code B', fontsize=16)
axes[1].set_ylabel('Neuron ID', fontsize=14)
axes[1].tick_params(axis='both', which='both', labelsize=14)
axes[1].xaxis.set_major_locator(MultipleLocator(100))
axes[1].yaxis.set_major_locator(MultipleLocator(25))

axes[2].imshow(C[:,:plot_steps], cmap='gray_r', interpolation='nearest', aspect=0.8)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].set_title('Code A * B', fontsize=16)
axes[2].set_xlabel('Time Step', fontsize=14)
axes[2].set_ylabel('Neuron ID', fontsize=14)
axes[2].tick_params(axis='both', which='both', labelsize=14)
axes[2].xaxis.set_major_locator(MultipleLocator(100))
axes[2].yaxis.set_major_locator(MultipleLocator(25))

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('PC_main_sim_spike_trains_mult.eps', format='eps')

# Create figure of activity distributions
fig, ax = plt.subplots()

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#ax.set_title('Population Codes', fontsize=16)
ax.plot(x, hA, label='Code A', linewidth=2, color='0.8')
ax.plot(x, theo_A, linestyle='--', linewidth=2, color='0.8')
ax.plot(x, hB, label='Code B', linewidth=2, color='0.6')
ax.plot(x, theo_B, linestyle='--', linewidth=2, color='0.6')
ax.plot(x, hC, label='Code A * B', linewidth=2, color='0')
ax.plot(x, theo_C, linestyle='--', linewidth=2, color='0')
ax.legend(fontsize=14)
ax.tick_params(axis='both', which='both', labelsize=14)
ax.set_xlabel("Preferred Value [Arbitrary Unit]", fontsize=14)
ax.set_ylabel("Firing Rate [Spikes / Time Step]", fontsize=14)
plt.savefig('PC_main_sim_code_sample_mult.eps', format='eps')

# Choose two neurons for computing the correlation
n1_C = int((a*b)*N/2+N/2)
n2_C = n1_C + 1

# Compute Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(C[n1_C,:], C[n2_C,:])
print(f"Correlation coefficient for A * B for nearby preferred values: {correlation_coefficient}")

# Create figure showing firing rate distribution between the two neurons
fig, ax = plt.subplots()

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Neurons with nearby preferred values', fontsize=16)
ax.plot(rf_C[n1_C], rf_C[n2_C], '.', color='black', label = 'Code A * B')
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', which='both', labelsize=14)
ax.set_xlabel("Firing Rate of Neuron 22", fontsize=14)
ax.set_ylabel("Firing Rate of Neuron 23", fontsize=14)
ax.set_xlim(0, 0.11)
ax.set_ylim(0, 0.11)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))

# Fit an ellipse to the C data
data_points = np.transpose(np.vstack((rf_C[n1_C], rf_C[n2_C])))
cov_matrix = np.cov(data_points, rowvar=False)
mean = np.mean(data_points, axis=0)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Get the angle of rotation and scale factors
angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
width, height = 4 * np.sqrt(eigenvalues)

# Create and plot the ellipse
ellipseC = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='k', fc='None', lw=2)


plt.gca().add_patch(ellipseC)
plt.savefig('PC_main_sim_corr_fire_rates_near_mult.eps', format='eps')

# Choose another two neurons for the firing rate plot
n2_C = 40

# Create figure showing firing rate distribution between the two neurons
fig, ax = plt.subplots()

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Neurons with distant preferred values', fontsize=16)
ax.plot(rf_C[n1_C], rf_C[n2_C], '.', color='black', label = 'Code A * B')
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', which='both', labelsize=14)
ax.set_xlabel("Firing Rate of Neuron 22", fontsize=14)
ax.set_ylabel("Firing Rate of Neuron 40", fontsize=14)
ax.set_xlim(0, 0.11)
ax.set_ylim(0, 0.11)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.05))

# Fit an ellipse to the C data
data_points = np.transpose(np.vstack((rf_C[n1_C], rf_C[n2_C])))
cov_matrix = np.cov(data_points, rowvar=False)
mean = np.mean(data_points, axis=0)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Get the angle of rotation and scale factors
angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
width, height = 4 * np.sqrt(eigenvalues)

# Create and plot the ellipse
ellipseC = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='k', fc='None', lw=2)

plt.gca().add_patch(ellipseC)
plt.savefig('PC_main_sim_corr_fire_rates_far_mult.eps', format='eps')


plt.show()

