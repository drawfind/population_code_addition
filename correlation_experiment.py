# This python code simulates the model and computes the correlation coefficients as function of the distance in preferred values for a set of population-code sizes.

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
import random
import sys
from scipy.stats import pearsonr
from tqdm import tqdm
import model

# Parameters for the experiment:
num_trials = 10 # Number of simulation runs
num_steps = 1000000 # Number of time steps in each run
a = 0 # Encoded variable 
a_sig = 0.25 # Tuning standard deviation

n_values = [11, 25, 51, 101, 201] # List of population-code sizes

# Loop through simulation runs
for t in tqdm(range(num_trials),file=sys.stderr):
    # Loop through N values
    for N in n_values: 
        N_2 = int(N/2)
    
        C = np.zeros((N, num_steps))
        
        for i in range(num_steps):
            A = model.create_code(a, a_sig, N)
            B = model.create_code(a, a_sig, N)
            # Compute composite code A + B
            C[:,i] = model.compute(A, B)

        # Compute mean value
        mu = np.mean(C[N_2,:])

        # Compute variance
        var = np.var(C[N_2,:]-mu)

        # Compute correlation
        # Loop through distance in preferred values
        for i in range(N_2):
            correlation_coefficient, p_value = pearsonr(C[N_2,:], C[N_2+1+i,:])
            # Print result in terminal
            print(f"{N} {mu} {var} {i+1} {correlation_coefficient}")

