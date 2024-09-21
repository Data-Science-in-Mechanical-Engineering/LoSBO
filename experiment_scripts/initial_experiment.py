""" 
Module to run initial experiment that demonstratets that demonstrates that beta=2 does not contain the target function in the uncertainty set for the ONB functions in many cases.
"""
import os
import fire
import yaml
import pandas as pd
import numpy as np
import torch
import gpytorch
from src.experiments.testfunctions import observe_data 
from src.gps.models import ExactGPSEModel


def experiment_0(file, beta=2):
    """
    This is the first experiment of the Paper. We generate 100 random RKHS functions on [0, 1] with an SE kernel and the ONB approach.
    We choose RKHS norm bound 10. We generate 10000 data sets by uniformly sampling 100 inputs from [0, 1], evaluating the RKHS function on these inputs. 
    We add i.i.d. normal noise with 0.01. We apply GP regression with zero mean prior and the SE kernel with the same length scale as covariance function.
    Args:
        file (str): path to yaml file containing function information
        beta (float): uncertainty parameter
    Returns:
        contains_all_dict (dict): dictionary containing whether the target function from the RKHS is completely contained in the uncertainty set for each of the 10000 runs
    """

    function_info = yaml.load(open(file, encoding="utf-8"), Loader=yaml.FullLoader)
    contains_all_dict = {}
    for i in range(10000):
        # generate 10000 data sets by uniformly sampling 100 inputs from [0, 1], evaluating the RKHS function on these inputs,
        # and then adding i.i.d. normal noise with 0.01
        x = torch.rand(100, 1)
        y = observe_data(x, function_info)
        # apply GP regression with zero mean prior and the SE kernel with the same length scale as covariance function
        # no Hyperparameter Tuning
        gp_model = ExactGPSEModel(
            x,
            y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=gpytorch.priors.NormalPrior(function_info["gamma"]/np.sqrt(2), 1),
            outputscale_constraint=None,
            outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0,
        )
        gp_model.eval()
        test_x = torch.linspace(0, 1, 1000).unsqueeze(1)
        y_data = observe_data(test_x, function_info)
        f_preds = gp_model(test_x)
        f_mean = f_preds.mean
        f_var = f_preds.variance
        
        # check if points are contained in uncertainty set
        lower_bound = f_mean - beta*f_var.sqrt()
        upper_bound = f_mean + beta*f_var.sqrt()
        contained = (y_data >= lower_bound) & (y_data <= upper_bound)
        contains_all_dict[f"iter_{i}"] = contained.all().item()  
    return contains_all_dict
     
     
def run_experiment_0_for_specific_function(onb_function_no=20):
    '''
    Run the experiment for multiple seeds so that it is faster
    '''
    # create results folder if it does not exist
    results_path = os.path.join("results", f"experiment_0")
    if os.path.isdir(results_path) is False:
            os.mkdir(results_path)
    # select config file
    file=f"src/function_configs/onb_functions/onb_function_{onb_function_no}.yaml"
    
    # run experiment with config
    contains_all_dict = experiment_0(file, beta=2)
    
    # save results in dataframe
    df = pd.DataFrame.from_dict(contains_all_dict, orient='index', columns=['contains_all'])
    df.to_csv(f"{results_path}/results_{onb_function_no}.csv")

if __name__ == '__main__':
    fire.Fire(run_experiment_0_for_specific_function)