# Model for computing with population-encoded variables

The code in this repository accompanies the article "Model for computing with population-encoded variables explains neural correlations"

## Requirements

This code requires Python 3 and the modules numpy, scipy, matplotlib, and tqdm. 

The code has been tested with Python 3.10.9 running on an Apple MacBook Air with M1 chip, 8 GB memory, and macOS Ventura 13.5.2.

## Installation

In a terminal, execute the following commands
```bash
git clone https://github.com/drawfind/population_code_addition
cd population_code_addition
```

## Demonstrations

To run a simulation of the computational model, type
```bash
python demo_code_subtraction.py
```

The typical run time is 1 minute on an Apple M1.

This demo will generate and display four figures:

Figure 1: Spike trains of the population codes v_O, v_P, and v_O - v_P.

Figure 2: Firing rate distributions over neurons' preferred values for each of the three population codes.

Figure 3: Firing rates between two neurons with nearby preferred values.

Figure 4: Firing rates between two neurons with distant preferred values.

Combined, these four figures create a figure as Figure 2 in the main manuscript.

To simulate the model for  multiplication instead of subtraction, run
```bash
python demo_code_mult.py
```

This demo will generate the same type of figures as above. The firing-rate distribution shows a plot as in Figure 6a of the main manuscript.

To simulate the Deneve model for subtraction, run
```bash
python demo_code_subtraction_Deneve.py
```

## Experiments

To run the experiment that computes the correlation coefficients as function of the distance in preferred values, type
```bash
python correlation_experiment.py > data.txt  
```

To plot the data, run

```bash
python plot_correlation_vs_distance.py data.txt
```

This program generates a figure as Figure 3a in the main manuscript.




