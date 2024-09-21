import math
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from utils.transform_utils import n_sphere_to_cartesian


def seed_everything(seed=0):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
    - seed (int): Seed value. Default is 0.

    Note:
    This function sets seeds for the Python standard library, NumPy, and PyTorch.
    Additionally, it sets the environment variable for PL_GLOBAL_SEED.
    """
    
    # Set seed for the Python standard library's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for PyTorch Lightning's global seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)

def read_config(config_path='config.yaml'):
    """
    Reads a YAML configuration file.

    Parameters:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def save_results(config, function_config, losbo_config, train_X, train_Y, time, regret, random=False, seed=0):
    """
    Saves experimental results in a structured format.

    Parameters:
        config (dict): General experiment configuration.
        function_config (dict): Configuration for the function under test.
        losbo_config (dict): Configuration for the LOSBO settings.
        train_X (torch.Tensor): Training inputs.
        train_Y (torch.Tensor): Training outputs.
        time (float): Time taken for the experiment.
        regret (torch.Tensor): Recorded regret over iterations.
        random (bool, optional): Specifies if random results are being saved. Defaults to False.
        seed (int, optional): Seed value for reproducibility. Defaults to 0.
    """
    results_path = config['results_path']
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    results = {'Y': train_Y.detach().numpy()}
    for i in range(function_config['domain_size']):
        results[f'X{i+1}'] = train_X[:, i].detach().numpy()

    regret = torch.cat((torch.ones(losbo_config['initial_safe_points']-1).unsqueeze(1), regret), 0).squeeze(-1).detach().numpy()
    results['regret'] = regret

    iteration = torch.cat((torch.zeros(losbo_config['initial_safe_points']-1).unsqueeze(1), torch.linspace(0, losbo_config['max_iterations'], losbo_config['max_iterations']+1).unsqueeze(1)), 0).squeeze(-1).detach().numpy()
    results['iteration'] = iteration

    results_df = pd.DataFrame.from_dict(results)
    file_name = 'results_random.csv' if random else f'seed_{seed}_results.csv'
    results_df.to_csv(os.path.join(results_path, file_name), index=False)

    with open(os.path.join(results_path, "log.txt"), "w") as file:
        file.write(f"Time: {time} seconds")

def setup_experiment(path, seed=42, set_lengthscale=False):
    """
    Configures and initializes the experiment.

    Parameters:
        path (str): Path to the main configuration file.
        seed (int, optional): Seed for random number generation. Defaults to 42.
        set_lengthscale (bool, optional): If True, adjusts kernel lengthscale. Defaults to False.

    Returns:
        tuple: Tuple containing LOSBO, function, kernel, and general configurations.
    """
    config = read_config(path)
    losbo_config = read_config(os.path.join("config", "losbo_config", f"{config['losbo_config']}.yaml"))
    kernel_config = read_config(os.path.join("config", "kernel_config", f"{config['kernel_config']}.yaml"))
    function_config = read_config(os.path.join("config", "function_config", f"{config['function_config']}.yaml"))

    if set_lengthscale:
        kernel_config["lengthscale"]["prior_mean"] = 1/function_config["lipschitz_constant"]

    config["seed"] = int(seed)
    seed_everything(config["seed"])

    if config["experiment_name"] != "pre_rkhs":
        config["results_path"] = f"results/{config['experiment_name']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{function_config['type']}_{losbo_config['acqf']}_{losbo_config['beta']['type']}_seed_{config['seed']}"

        if not os.path.exists(config["results_path"]):
            os.makedirs(config["results_path"])
        with open(os.path.join(config["results_path"], "config.yaml"), "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            yaml.dump(losbo_config, file, default_flow_style=False)
            yaml.dump(kernel_config, file, default_flow_style=False)
            yaml.dump(function_config, file, default_flow_style=False)

    return losbo_config, function_config, kernel_config, config

def determine_lipschitz_constant(observe_data, bounds, num_points=100):
    """
    Calculates the Lipschitz constant for a function by estimating the maximum gradient over a grid.

    Parameters:
        observe_data (callable): The function whose Lipschitz constant is to be determined.
        bounds (list): List of bounds for each dimension of the input space.
        num_points (int, optional): Number of points in each dimension of the grid. Defaults to 100.

    Returns:
        float: Estimated Lipschitz constant.
    """
    x1_range = torch.linspace(bounds[0], bounds[1], num_points)
    x2_range = torch.linspace(bounds[2], bounds[3], num_points)
    grid_x1, grid_x2 = torch.meshgrid(x1_range, x2_range)
    grid = torch.stack((grid_x1.flatten(), grid_x2.flatten()), dim=1)

    max_gradient = 0
    for point in grid:
        point = point.requires_grad_(True)
        y = observe_data(point.unsqueeze(0))
        y.backward()
        gradient_magnitude = point.grad.norm().item()
        max_gradient = max(max_gradient, gradient_magnitude)

    return max_gradient
    

######################
# Plotting functions #
######################

def plot_2d_function_with_colorbar(function, bounds, num_points=1000, fig=None, ax=None, axis_x="x1", axis_y="x2"):
    """
    Plots a 2D function with a colorbar.

    Parameters:
        function (callable): Function to plot.
        bounds (list): Bounds of the input space [xmin, xmax, ymin, ymax].
        num_points (int): Number of points to plot.
    """
    # Create a new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Create a grid of points
    x_values = np.linspace(bounds[0], bounds[1], num_points)
    y_values = np.linspace(bounds[2], bounds[3], num_points)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Convert to torch tensor for function evaluation
    torch_grid = torch.tensor(grid_points, dtype=torch.float32)
    zz = function(torch_grid).detach().numpy()
    
    # Plot the function
    contour_set = ax.contourf(xx, yy, zz.reshape(xx.shape))
    ax.set_xlabel(axis_x)
    ax.set_ylabel(axis_y)

    # Add a colorbar
    fig.colorbar(contour_set, ax=ax)

    return fig, ax

def plot_2d_gp_with_colorbar(gp, bounds, num_points=100, fig=None, ax=None,axis_x="x1", axis_y="x2"):
    """
    Plots a 2D function with a colorbar.

    Parameters:
        function (callable): Function to plot.
        bounds (list): Bounds of the input space [xmin, xmax, ymin, ymax].
        num_points (int): Number of points to plot.
    """
    # Create a new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Create a grid of points
    x_values = np.linspace(bounds[0], bounds[1], num_points)
    y_values = np.linspace(bounds[2], bounds[3], num_points)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Convert to torch tensor for function evaluation
    torch_grid = torch.tensor(grid_points, dtype=torch.float32)
    f_preds = gp(torch_grid)
    zz = f_preds.mean.detach().numpy()
    #zz = function(torch_grid).mean.detach().numpy()
    
    # Plot the function
    contour_set = ax.contourf(xx, yy, zz.reshape(xx.shape))

    # Add a colorbar
    fig.colorbar(contour_set, ax=ax)

    return fig, ax

def plot_circle(center, radius, fig=None, ax=None):
    """
    Plots a circle.

    Parameters:
        center (torch.Tensor): Center of the circle.
        radius (float): Radius of the circle.
    """
    phi = 2 * math.pi * torch.linspace(0, 1, 100)
    x = radius * torch.cos(phi) + center[0]
    y = radius * torch.sin(phi) + center[1]
    ax.plot(x, y, color='black')
    #ax.plot(x, y, color='black')
    return fig, ax

def plot_2d_ucb_grad_with_colorbar(ucb, bounds, num_points=100, fig=None, ax=None, axis_x="x1", axis_y="x2"):
    """
    Plots a 2D function with a colorbar.

    Parameters:
        ucb (callable): UCB acquisition function.
        bounds (list): Bounds of the input space [xmin, xmax, ymin, ymax].
        num_points (int): Number of points to plot.
    """
    # Create a new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Create a grid of points
    x_values = np.linspace(bounds[0], bounds[1], num_points)
    y_values = np.linspace(bounds[2], bounds[3], num_points)
    xx, yy = np.meshgrid(x_values, y_values)
    
    # Convert to torch tensor for function evaluation
    torch_grid = torch.tensor(np.column_stack((xx.flatten(), yy.flatten())), dtype=torch.float32)
    input_points = torch_grid.requires_grad_()

    # Evaluate the UCB acquisition function at the input points
    ucb_values = ucb(input_points)
    print(ucb_values)
    ucb_values.sum().backward()
    gradient = input_points.grad
    print(gradient.abs().min())
    mag = gradient.norm(dim=-1)
    
    # Reshape the gradient to match the shape of xx and yy
    zz = mag.reshape(xx.shape).detach().numpy()
    
    # Plot the function
    contour_set = ax.contourf(xx, yy, zz)

    # Add a colorbar
    fig.colorbar(contour_set, ax=ax)

    return fig, ax

################################################################ 
##################### safe initial points ######################
################################################################

def get_initial_safe_points(observe_data, config, function_config, losbo_config, kernel_config):
    """
    Determines initial safe points based on the function configuration.

    Parameters:
        observe_data (callable): Function to observe data points.
        config (dict): General configuration dictionary.
        function_config (dict): Function-specific configuration.
        losbo_config (dict): LOSBO-related configuration.
        kernel_config (dict): Kernel configuration.

    Returns:
        torch.Tensor: Tensor of initial safe points.
    """
    if function_config["type"] == "gaussian_10D":
        train_X = generate_level_set_gaussian_10D(function_config["safety_threshold"]+0.2, losbo_config["initial_safe_points"])
    elif "rkhs" in function_config["type"]:
        #config['results_path'] = f"experiments/results/{config['experiment_name']}/{function_config['type']}_{losbo_config['acqf']}_{function_config['function_number']}"
        train_X = generate_random_safe_seed_set(observe_data, function_config["safety_threshold"]+losbo_config["E"], seed=config["seed"], intervall=function_config["reachable_safe_set"], sample_points=1, function_info=function_config, function_config=function_config)
    else:
        train_X = generate_random_safe_initial_points(lambda x: observe_data(x, function_config), function_config["domain_bounds"], function_config["safety_threshold"]+0.2, losbo_config["initial_safe_points"])
    
    if train_X.ndim == 1:
        train_X = torch.Tensor(train_X).unsqueeze(0)
    return train_X

def generate_random_safe_initial_points(func, bounds, safety_threshold, num_points):
    """
    Generates a tensor of random safe initial points within given bounds.

    Parameters:
        func (callable): The function to evaluate safety.
        bounds (list of tuples): List of min and max values for each dimension.
        safety_threshold (float): Threshold above which a point is considered safe.
        num_points (int): Number of safe points to generate.

    Returns:
        torch.Tensor: A tensor containing safe points.
    """
    valid_points = []
    while len(valid_points) < num_points:
        x = torch.tensor([torch.FloatTensor(1).uniform_(a, b) for a, b in bounds])
        func_value = func(x)
        if func_value > safety_threshold:
            valid_points.append(x)
    return torch.stack(valid_points)

def generate_random_safe_seed_set(observe_data, safety_threshhold, seed=0, intervall=[0,1], sample_points=2, function_info=None, function_config=None):
    """
    Generates a random seed set that is safe within specified interval and threshold.

    Parameters:
        observe_data (callable): Function to observe data points.
        safety_threshhold (float): Safety threshold to determine if a point is safe.
        seed (int): Seed for reproducibility of random samples.
        intervall (list): Interval within which points are considered.
        sample_points (int): Number of safe points to sample.
        function_info (dict): Additional function-specific info.
        function_config (dict): Function configuration details.

    Returns:
        numpy.ndarray: Array of safe seed points.
    """
    x = torch.linspace(0, 1, 10000).unsqueeze(1)
    y = observe_data(x, function_config)
    df = pd.DataFrame({'x': x.squeeze(-1).numpy(), 'y': y.squeeze(-1).numpy()})
    df = df[df['y'] > (safety_threshhold + 2 * function_info["noise_lvl"])]
    df = df[(df['x'] > intervall[0]) & (df['x'] < intervall[1])]
    df = df.sample(n=sample_points, random_state=seed)
    return df['x'].values

def generate_level_set_gaussian_10D(level, points):
    """
    Generates points on the level set of a Gaussian in 10D space.

    Parameters:
        level (float): Level set to generate points on.
        points (int): Number of points to generate.

    Returns:
        torch.Tensor: Points on the Gaussian level set in 10D.
    """
    radius = math.sqrt(-math.log(level) / 4)
    center = torch.zeros(10)
    polar_coordinates = torch.rand(points, 10).unsqueeze(1)
    polar_coordinates[:, :, 0] = 1
    return n_sphere_to_cartesian(polar_coordinates, radius, center)