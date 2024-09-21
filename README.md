# On Safety in Safe Bayesian Optimization
This repository contains Python implementations of RealBetaSafeOpt, LoSBO, and LosGP-UCB.

## Installation
Clone the repository. Then, into an environment with e.g., Python 3.8.5, you can install all needed packages with:

```pip install -r requirements.txt```

## Reproducing the experiments with grid implementation of SafeOpt, RealBetaSafeOpt and LoSBO
To reproduce the experiments with synthetic target functions of the paper, you can now run the Python scripts corresponding to the different experiments from the command line.

There are different experiments. The functions take 4 arguments. The first is the experiment configuration, the second is the function number between 0 and 100, the third is the number of runs, and the fourth is the option to plot the results. 

Example: 

```python experiment_scripts/grid_experiment.py 2 46 10000 False```

## Reproducing the experiments with LosGP-UCB

Example: 

The experiment for LoSGP-UCB uses config files. It takes the arguments of the path to the config file and the random seed for the experiment. 

```python experiment_scripts/losgpucb_experiment.py config/experiment_se_camel.yaml 1```




