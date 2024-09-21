import numpy as np
from scipy.optimize import minimize


# Define the function
def func(x):
    return -np.exp(-4*np.linalg.norm(x)**2)

# Define the Lipschitz function
def lipschitz_func(x, y):
    return abs(func(x) - func(y)) / np.linalg.norm(x - y)

# Define the objective function to minimize (-Lipschitz function)
def objective(x):
    return -func(x)

# Define the bounds for the optimization ([-1, 1] for each dimension)
bounds = [(-1, 1)] * 10  # Assuming 10 dimensions

# Perform optimization to find the maximum Lipschitz constant
result = minimize(objective, np.zeros(10), bounds=bounds)

# The maximum Lipschitz constant is the negative of the optimized objective value
max_lipschitz_constant = -result.fun

print("Approximate Maximum Lipschitz Constant:", max_lipschitz_constant)