# This python code plot the correlation coefficients as function
# of the distance in preferred values for a set of population-code sizes.
# This code generates Fig. 3a based on the output data from the program
# correlation_experiment.py

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
import matplotlib.pyplot as plt
import sys

# Get the file name from the command line argument
def get_filename():
    default_filename = "correlation_n_dependence_sig0.25_10trials.txt"
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return default_filename

data_file = get_filename()
data = np.loadtxt(data_file)

# get unique N values
n_values = np.unique(data[:,0])

fig, ax = plt.subplots()

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create plot for each N value
for i in range(len(n_values)):
    N = n_values[i]
    N_2 = int(N/2)

    data_N = data[data[:, 0] == N]

    cc_mean = np.zeros(N_2)
    cc_std = np.zeros(N_2)
    dist = np.zeros(N_2)
    
    for d in range(1,N_2+1):
        data_d = data_N[data_N[:, 3] == d]
        cc = data_d[:,4]
        cc_mean[d-1] = np.mean(cc)
        cc_std[d-1] = np.std(cc)
        dist[d-1] = d*2/(N-1)

    label = "N = " + str(int(N))
    color = str(1 - (i+1)*0.2)
    ax.plot(dist, cc_mean, label=label, color=color)
    ax.errorbar(dist, cc_mean, yerr=cc_std, fmt='.', capsize=3, color=color)

ax.tick_params(axis='both', which='both', labelsize=14)
ax.set_xlabel("Distance Between Preferred Speeds",fontsize=14)
ax.set_ylabel("Spike Correlation (r)",fontsize=14)
plt.legend(fontsize=14)
plt.savefig('correlation_vs_distance.eps', format='eps')

plt.show()

