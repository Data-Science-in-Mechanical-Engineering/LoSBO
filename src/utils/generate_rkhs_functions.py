'''
Module with python functions to generate RKHS test functions

These scripts are used to generate the test functions, the test functions in this repository are generated on a windows pc, this leads to different results than the ubuntu docker conttainer. 
'''
import os
import math
import numpy as np
import scipy.linalg as la
import pandas as pd
import yaml
import pickle

from sklearn.gaussian_process.kernels import RBF, Matern

from src.utils.sample_rkhs import PreRKHSfunctionGenerator, PreRKHSfunction, se_bfunc_1d


#function to check if path exists, other wise create it
def check_if_path_exists(path):
    """
    check if a path exists, otherwise create it
    
    Parameters:
        path: path to check
    """
    if not os.path.exists(path):
        os.makedirs(path)

def generate_100_onb_se_test_functions(base_config_path):
    '''
    function that generate 100 RKHS test functions with SE kernel and ONB sampling
    Remark: sometimes there are numerical issues so that for a certain seed the function cannot be generated, these seeds are skipped
    
    Parameters:
        path: path where the function information should be saved
    '''
    counter = 0
    seed = 0
    while counter < 100:
        try:
            function_config = yaml.load(open(base_config_path, encoding="utf-8"), Loader=yaml.FullLoader)
            check_if_path_exists(f"{function_config['path']}")
            function_config["function_number"] = counter
            function_config["random_seed"] = seed 
            function_config = sample_rkhs_se_onb_function(function_config)
            yaml.dump(function_config, open(f"{function_config['path']}/function_{counter}.yaml", "w"))
            seed += 1
            counter += 1
        except ValueError:
            seed += 1
            continue
    
    
def sample_rkhs_se_onb_function(function_config):
    '''
    evaluation of the RKHS function with SE kernel and ONB sampling
    
    Parameters:
        function_info: dictionary containing the function information
    
    Returns:
        function_info: dictionary containing the function information and the analysis of the function
    
    '''
    # analysis of the function on a grid
    x = np.linspace(0, 1, 1001)
    
    np.random.seed(function_config["random_seed"]) # important as otherwise ranomdness is not reproducible, when evaluating the function
    n_bfuncs = np.random.randint(low=function_config["n_bfuncs_min"], high=20, size=1).item()
    indices_bfuncs = np.random.choice(np.arange(0, 20+1), size=n_bfuncs, replace=False)
    ys_bfuncs = np.zeros([len(x), n_bfuncs])
    for i_bf in range(n_bfuncs):
        ys_bfuncs[:, i_bf] = se_bfunc_1d(x.flatten(), indices_bfuncs[i_bf], gamma=function_config["gamma"])
    coeffs = np.random.normal(size=n_bfuncs)
    coeffs = coeffs/la.norm(coeffs, ord=2)*function_config["rkhs_norm"]
    y = ys_bfuncs @ coeffs
    if np.isnan(y).any():
        raise ValueError('Seed not usable')
    # save coefficients in pickle file
    check_if_path_exists(f"{function_config['path']}/coeffs")
    dict_coeffs = {"n_bfuncs": n_bfuncs, "coeffs": coeffs, "indices_bfuncs": indices_bfuncs}
    dict_coeffs = pickle.dump(dict_coeffs, open(f"{function_config['path']}/coeffs/coeffs_function_{function_config['function_number']}.pkl", "wb"))
    # analysis of the function
    analysis_dict = function_analysis(x, y)
    function_config.update(analysis_dict)
    return function_config

def generate_100_pre_rkhs_functions(base_config_path):
    '''
    function that generate 100 RKHS test functions with pre_rkhs sampling
    
    Parameters:
        base_config_path: path to the base config file
    '''
    for seed in range(0, 100):
            function_config = yaml.load(open(base_config_path, encoding="utf-8"), Loader=yaml.FullLoader)
            check_if_path_exists(f"{function_config['path']}")
            function_config["function_number"] = seed
            function_config["random_seed"] = seed 
            function_config = sample_pre_rkhs(function_config)
            yaml.dump(function_config, open(f"{function_config['path']}/function_{seed}.yaml", "w"))

def sample_pre_rkhs(function_config):
    '''
    sample a pre rkhs function
    
    Parameters:
        function_config: dictionary containing the function information
    
    Returns:
        function_config: dictionary containing the function information and the analysis of the function
    '''
    
    np.random.seed(function_config["random_seed"])
    
    # analysis of the function on a grid
    x = np.linspace(0, 1, 1001)
    
    length_scale = function_config["gamma"]/math.sqrt(2)
    
    base_point_generator = lambda n_base_points: np.random.normal(size=(n_base_points, 1))
    if function_config["kernel"] == "SE":
        kernel = RBF(length_scale=length_scale)
    elif function_config["kernel"] == "Matern":
        kernel = Matern(length_scale=length_scale, nu=function_config["nu"])
    else:
        raise NotImplementedError

    pre_rkhs_function_generator = PreRKHSfunctionGenerator(kernel, base_point_generator)
    pre_rkhs_function = pre_rkhs_function_generator(rkhs_norm=function_config["rkhs_norm"], n_base_points=function_config["n_basepoints"])
    dict_coeffs = {"base_points": pre_rkhs_function._base_points, "coeffs": pre_rkhs_function._coeffs}
    check_if_path_exists(f"{function_config['path']}/coeffs")
    pickle.dump(dict_coeffs, open(f"{function_config['path']}/coeffs/coeffs_function_{function_config['random_seed']}.pkl", "wb"))
    y = pre_rkhs_function(x)
    analysis_dict = function_analysis(x, y)
    function_config.update(analysis_dict)
    return function_config
    

############################################################################################################
# Function analysis
############################################################################################################
def function_analysis(x,y,safety_threshold_height=0.2):
    '''
    get information about the generated rkhs function
    
    Parameters:
        x: input values
        y: sampled, noisefree function values
        safety_threshold_height: safety threshhold height
    '''
    info_dict = dict()
    info_dict["mean"] = y.mean().item()
    info_dict["std"] = y.std().item()
    info_dict["optimum"] = y.max().item()
    info_dict["safety_threshold"] = (info_dict["mean"] - info_dict["std"]*safety_threshold_height)
    
    # create data frame for further analysis
    df = pd.DataFrame({"x":x, "y":y.flatten()})
    info_dict["lipschitz_constant"] = determine_lipschitz_constant(df)
    info_dict["reachable_safe_set"] = find_optimal_interval(df, info_dict["safety_threshold"]).tolist()
    return info_dict

def determine_lipschitz_constant(df):
    """
    determine maximum gradient of function
    
    Parameters:
        df: dataframe containing a grid of noisefree function values
    
    Returns:
        maximum gradient of function
    """
    df['gradient'] = df['y'].diff()/df['x'].diff()
    return float(df['gradient'].abs().max())

def find_optimal_interval(df, safety_threshold):
    '''
    find the set of safe function values containing the optimum 
    
    Parameters:
        df: dataframe containing the function values
        safety_threshold: safety threshhold
    
    Returns:
        reachable_safe_set: set of safe function values containing the optimum
    '''
    # Step 1: Find the index of the row with the maximum 'y' value
    max_y_idx = df['y'].idxmax()

    # Step 2: Create a boolean series where 'y' > 's'
    above_threshold = df['y'] > safety_threshold

    # Step 3: Group by intervals separated by 'y' dropping below 's'
    # 'cumsum' increments the group number every time 'y' goes below 's'
    groups = (~above_threshold).cumsum()

    # Step 4: Find the group containing the max 'y' value
    optimal_group = groups[max_y_idx]

    # Filter the DataFrame to this group only
    optimal_interval_df = df[(groups == optimal_group) & above_threshold]
    reachable_safe_set = np.array([optimal_interval_df ['x'].values[0], optimal_interval_df ['x'].values[-1]])

    return reachable_safe_set

# main function
if __name__ == '__main__':
    path = "config/function_config/base_pre_rkhs_se.yaml"
    generate_100_pre_rkhs_functions(path)
    path = "config/function_config/base_pre_rkhs_matern32.yaml"
    generate_100_pre_rkhs_functions(path)
    path = "config/function_config/base_onb_rkhs_se.yaml"
    generate_100_onb_se_test_functions(path)