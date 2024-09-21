import torch
from botorch.generation import gen_candidates_torch
from botorch.optim import optimize_acqf
from joblib import Parallel, delayed

from src.algorithms.acquisition_functions import (Mean, MeanSphere,
                                                  UpperConfidenceBound,
                                                  UpperConfidenceBoundSphere)
from utils.transform_utils import n_sphere_to_cartesian


class SafeBO:
    """
    Base class for Safe Bayesian Optimization.

    Attributes:
        gp_model (ExactGPSEModel): Gaussian Process (GP) model.
        bounds (list): Bounds of the input space, e.g., [[0, 1], [0, 1]].
        domain_size (int): Dimension of the input domain.
        seed_set (torch.Tensor): Initial set of safe points.
        safe_set (torch.Tensor): Current set of safe points.
        safety_threshold (float): Safety threshold.
        lipschitz_constant (float): Lipschitz constant of the function.
        E (float): Noise bound.
        beta (float): Exploration-exploitation trade-off parameter.
        X (torch.Tensor): Training inputs.
        Y (torch.Tensor): Training targets.
    """
    def __init__(self, config, gp):
        self.gp_model = gp  # GP model
        self.bounds = config["bounds"] # bounds of the input space D for example:  [[0, 1], [0, 1]]
        if isinstance(self.bounds[0], int):
            self.domain_size = 1
        else:
            self.domain_size = len(self.bounds)
        self.seed_set = self.gp_model.train_inputs[0]   # S_0
        self.safe_set = self.gp_model.train_inputs[0]   # S_t
        self.safety_threshold = config["safety_threshhold"]   # h
        self.lipschitz_constant = config["lipschitz_constant"] # L
        self.E = config["E"] # noise bound
        self.beta = config["beta"] # beta
        self.X = self.gp_model.train_inputs[0]
        self.Y = self.gp_model.train_targets
        if len(self.Y.shape) == 0:
            self.Y = self.Y.unsqueeze(0)


class LosGPUCB(SafeBO):
    """
    Implements the Los-GP-UCB algorithm for Safe Bayesian Optimization with Gaussian Processes.

    Inherits:
        SafeBO: Base class for safe Bayesian Optimization.

    Methods:
        add_new_point: Adds a new point to the training data.
        update_circles: Updates the safe region based on the latest observations.
        optimize: Optimizes the acquisition function to select the next query point.
        calculate_regret: Calculates the regret of the current solution against the optimum.
        process: Processes each candidate during parallel optimization.
    """
    
    def __init__(self, config, gp): 
        super().__init__(config, gp)
        self.beta = 2
        self.restarts = 10
        self.raw_samples = 10
        self.update_circles()
    
    def add_new_point(self, x, y):
        """
        Adds a new observation to the model's training data.

        Parameters:
            x (torch.Tensor): The input point.
            y (torch.Tensor): The observed output.
        """
        if self.domain_size == 1:
            x = torch.tensor([x])
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        self.X = torch.cat((self.X, x.unsqueeze(0)), 0)
        self.Y = torch.cat((self.Y, y.squeeze(0)), 0)
        
    def update_circles(self):
        """
        Updates the radius and center of the safe region based on the latest observations.
        """
        self.radius = (self.Y - self.E - self.safety_threshold) / self.lipschitz_constant
        self.center = self.X
    
    def optimize(self):
        self.update_circles()
        # Parallelize the computation of the acquisition function
        candidates = Parallel(n_jobs=10)(delayed(self.process)(i, "UCB") for i in range(len(self.X)))
    
        candidates = torch.cat(candidates).squeeze(0)
        
        # Initialize a filter with all True values
        valid_candidates = torch.ones(candidates.size(0), dtype=torch.bool)

        if self.domain_size == 1:
            lower_bound, upper_bound = self.bounds
            valid_candidates &= (candidates >= lower_bound) & (candidates <= upper_bound)
        else:
            # Update the filter for each dimension
            for dim in range(self.domain_size):
                lower_bound, upper_bound = self.bounds[dim]
                valid_candidates &= (candidates[:, dim] >= lower_bound) & (candidates[:, dim] <= upper_bound)
        
        acqf = UpperConfidenceBound(self.gp_model, beta=2)
        
        # Apply the filter to make sure result is inside bounds
        candidates = candidates[valid_candidates]
        opt = acqf(candidates).argmax(dim=0)
        x_next = candidates[opt]
        return x_next
    
    def calculate_regret(self, function_config, observe_data):
        """
        Calculates the regret of the current solution against the known optimum.

        Parameters:
            function_config (dict): Configuration of the function being optimized.
            observe_data (callable): Function to observe data points.

        Returns:
            torch.Tensor: The regret of the current best solution.
        """
        self.update_circles()
        # Parallelize the computation of the acquisition function
        candidates = Parallel(n_jobs=10)(delayed(self.process)(i, "Mean") for i in range(len(self.X)))
        candidates = torch.cat(candidates).squeeze(0)
        acqf = Mean(self.gp_model)
        opt = acqf(candidates).argmax(dim=0)
        candidate = candidates[opt]
        current_estimated_opt = observe_data(candidate)
        optimum = function_config["optimum"]
        regret = optimum - current_estimated_opt
        if len(regret.shape) == 1:
            regret = regret.unsqueeze(0)
        return regret
    
    def process(self, i, type):
        """
        Processes each candidate point during the parallel optimization step.

        Parameters:
            i (int): Index of the candidate point.
            type (str): Type of acquisition function to apply ("UCB" or "Mean").

        Returns:
            torch.Tensor: The processed candidate point.
        """            
        radius = self.radius[i]
        center = self.center[i].squeeze(0)
        if type == "UCB":
            acqfs = UpperConfidenceBoundSphere(self.gp_model, beta=self.beta, radius=radius, center=center)
        elif type == "Mean":
            acqfs = MeanSphere(self.gp_model, radius=radius, center=center)
        if self.domain_size == 1:
            if center - self.bounds[0] < radius:
                bounds = torch.stack([(bounds[0] - center)/radius, torch.ones(1)])
            elif self.bounds[1] - center < radius:
                bounds = torch.stack([-torch.ones(1), (bounds[1]-center)/radius])
            else:
                bounds = torch.stack([-torch.ones(1), torch.ones(1)])
        else:
            bounds = torch.stack([torch.zeros(len(self.bounds)), torch.ones(len(self.bounds))])
        candidate, acq_value = optimize_acqf(
            acqfs,
            bounds=bounds,
            q=1,
            num_restarts=self.restarts,
            raw_samples=self.raw_samples,
            return_best_only=False,
            gen_candidates=gen_candidates_torch,
        )

        candidate = n_sphere_to_cartesian(candidate, radius, center).squeeze(1)
        return candidate