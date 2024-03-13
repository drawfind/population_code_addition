# Model for computing with population-encoded variables

The code in this repository accompanies the article "Model for computing with population-encoded variables explains neural correlations," submitted to Current Biology.

## Requirements

This code requires Python 3 and the modules numpy, scipy, matplotlib, and tqdm. 

The code has been tested with Python 3.10.9 running on an Apple MacBook Air with M1 chip, 8 GB memory, and macOS Ventura 13.5.2.

## Installation

In a terminal, execute the following commands
```bash
git clone https://github.com/drawfind/population_code_addition
cd population_code_addition
```

## Demonstration

To run the simulation, type
```bash
python demo_code_addition.py
```

The typical run time is 1 minute on an Apple M1.

This demo will generate and display four figures:

Figure 1: Spike trains of the population codes A, B, and A + B.

Figure 2: Firing rate distributions over neurons' preferred values for each of the three population codes.

Figure 3: Firing rates between two neurons with nearby preferred values.

Figure 4: Firing rates between two neurons with distant preferred values.

Combined, these four figures create a figure as Figure 2 in the main manuscript.

## Experiments

To run the experiment that computes the correlation coefficients as function of the distance in preferred values, type
```bash
python correlation_experiment.py     
```
This program generates data as was used for Figure 3a in the main manuscript.

To run the experiment with ternary functions, type
```bash
python correlation_experiment_add3.py     
```
This program generates data as was used for Figure 6 in the main manuscript.




