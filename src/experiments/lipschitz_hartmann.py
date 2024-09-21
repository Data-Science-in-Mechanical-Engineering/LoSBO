import torch
import math
import src.experiments.testfunctions as testfunctions

# Define your PyTorch function here
def hartmann_6D(X): 
    '''
    2D Optimization problem
    Rosenbrock function: domain [0, 1]^6
    '''
    if X.dim() == 1:
        X = X.unsqueeze(0)
    
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([[10, 3, 17, 3.50, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14]])

    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])

    # Initialize the outer and inner variables
    outer = 0

    # Calculate the outer summation
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = X[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner += Aij * (xj - Pij) ** 2
        new = alpha[ii] * torch.exp(-inner)
        outer += new

    # Calculate the final result
    y = -outer/(-3.32237)
    return y

def camel_function(X):
    """
    Computes the Camel function.

    Parameters:
        X (torch.Tensor): A 2D tensor where each row is a point in the function's domain.

    Returns:
        torch.Tensor: The function values at each row of X.
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)
    xx = X[:,0]
    yy = X[:,1]
    y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
    return torch.maximum(-y, torch.Tensor([-2.5])).unsqueeze(-1)

def gaussian_function(X):
    """
    Computes the Gaussian function.

    Parameters:
        X (torch.Tensor): A 2D tensor where each row is a point in the function's domain.

    Returns:
        torch.Tensor: The function values at each row of X.
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return torch.exp(-4*X.norm(dim=-1, keepdim=True)**2).squeeze(-1)

def func(x):
    #return gaussian_function(x)
    return hartmann_6D(x)



def determine_lipschitz_constant(observe_data, bounds, num_points=100):
    x1_range = torch.linspace(bounds[0], bounds[1], num_points)  # X1 range
    x2_range = torch.linspace(bounds[2], bounds[3],  num_points)   # X2 range
    grid_x1, grid_x2 = torch.meshgrid(x1_range, x2_range)
    grid = torch.stack((grid_x1.flatten(), grid_x2.flatten()), dim=1)

    # Calculate the gradient and find the maximum gradient magnitude
    max_gradient = 0
    for point in grid:
        point = point.requires_grad_(True)
        y = observe_data(point.unsqueeze(0))
        y.backward()
        gradient_magnitude = point.grad.norm().item()
        max_gradient = max(max_gradient, gradient_magnitude)

    return max_gradient*math.sqrt(2)

def lipschitz_grid(func, fdim, grid_size=0.1, lower_bound=0, upper_bound=1):
    # Generate grid points
    grid_points = torch.arange(lower_bound, upper_bound + grid_size, grid_size)

    # Generate meshgrid for 6D space
    grid = torch.meshgrid(*[grid_points]*fdim)

    # Convert grid points to tensor and reshape for function evaluation
    grid_tensor = torch.stack(grid, dim=-1).reshape(-1, fdim)
    print(grid_tensor)

    # Evaluate the function on the grid
    function_values = func(grid_tensor)

    # Reshape function values to match the grid shape
    function_values = function_values.reshape((len(grid_points),) * fdim)

    # Compute gradients using finite differences
    gradient = torch.zeros_like(grid_tensor)
    for dim in range(6):
        # Shift grid points in the positive direction
        grid_tensor_pos_shift = grid_tensor.clone()
        grid_tensor_pos_shift[:, dim] += grid_size
        function_values_pos_shift = func(grid_tensor_pos_shift)
        
        # Shift grid points in the negative direction
        grid_tensor_neg_shift = grid_tensor.clone()
        #grid_tensor_neg_shift[:, dim] -= grid_size
        function_values_neg_shift = func(grid_tensor)
        
        # Compute gradient using finite differences
        gradient[:, dim] = (function_values_pos_shift - function_values_neg_shift) / (grid_size)

    # Calculate the gradient magnitude
    gradient_magnitude = torch.norm(gradient, dim=-1)

    # Find the maximum gradient magnitude
    max_gradient_magnitude = torch.max(gradient_magnitude)
    return max_gradient_magnitude*math.sqrt(fdim)

print(lipschitz_grid(hartmann_6D, 6, grid_size=0.1, lower_bound=0, upper_bound=1))
print(lipschitz_grid(gaussian_function, 10, grid_size=0.2, lower_bound=0, upper_bound=1))


max_gradient = determine_lipschitz_constant(testfunctions.camel_function, [-2, 2, -1, 1], num_points=50)
print(max_gradient)