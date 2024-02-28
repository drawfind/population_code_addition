# Model for computing with population-encoded variables

The code in this repository accompanies the article "Model for computing with population-encoded variables explains noise correlations."

This code requires Python 3 and the modules numpy, scipy, matplotlib, and tqdm. 

It has been tested with Python 3.10.9 running on an Apple MacBook Air with M1 chip, 8 GB memory, and macOS Ventura 13.5.2.

To install the code, under the green Code button, choose Download ZIP and unzip the file (if it doesn't unzip automatically).

In a terminal, change into the directory created by unzipping the downloaded file.

To run the simulation, type
```bash
python demo_code_addition.py
```

The typical run time is 1 minute on an Apple M1.

The code will generate and display 4 figures.

Figure 1: Spike trains of the population codes A, B, and A + B.

Figure 2: Firing rate distributions over neurons' preferred values for each of the three population codes.

Figure 3: Firing rates between two neurons with nearby preferred values.

Figure 4: Firing rates between two neurons with distant preferred values.