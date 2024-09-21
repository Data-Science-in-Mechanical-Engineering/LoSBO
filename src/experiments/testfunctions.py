import math
import pickle
import numpy as np
import torch
from sklearn.gaussian_process.kernels import RBF, Matern
from src.utils.experiment_utils import read_config
from src.utils.sample_rkhs import PreRKHSfunctionGenerator, PreRKHSfunction, se_bfunc_1d

def branin_mod(x):   
    """ 
    2D Optimization problem
    Braning function: domain [-5, 10] x [0, 15]
    like Paulson 2022 "Efficient Multi-Step Lookahead Bayesian Optimization
    with Local Search Constraints" 
    further modified to a maximazation problem
    """
    print("branin")
    try:
        x1 = x[:,0]
        x2 = x[:,1]
    except:
        x1 = x[0]
        x2 = x[1]
    a = 1
    b = 5.1/(4*torch.pi**2)
    c = 5/torch.pi
    r = 6
    s = 10
    t = 1/(8*torch.pi) 
    f1 = a*(x2 - b*x1**2 + c*x1 - r)**2
    f2 = s*(1-t)*torch.cos(x1)+s
    l1 = 5*torch.exp(-5*((x1+3.14)**2+(x2-12.27)**2))
    l2 = 5*torch.exp(-5*((x1+3.14)**2+(x2-2.275)**2))
    y = f1 + f2 + l1 + l2
    return ((300-y)/300).unsqueeze(-1)

def rosenbrock_2d(X, a = 1, b = 100): 
    '''
    2D Optimization problem
    Rosenbrock function: domain [-2, 2] x [-1,3]
    '''
    try:
        x1 = X[:,0]
        x2 = X[:,1]
    except:
        x1 = X[0]
        x2 = X[1]
    f = (300 - ((a-x1)**2 + b*(x2-x1**2)**2))/300
    return f.unsqueeze(-1)


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
    return y.unsqueeze(-1)

def rosenbrock(X):
    """
    Computes the Rosenbrock function.

    Parameters:
        X (torch.Tensor): A 2D tensor where each row is a point in the function's domain.

    Returns:
        torch.Tensor: The function values at each row of X.
    """
    print("rosenbrock")

    if X.dim() == 1:
        X = X.unsqueeze(0)
  
    idx = torch.arange(X.size(1) - 1)

    len = X.size(1)
    Y = (0.5*10**len - torch.sum(100 * (X[:, idx + 1] - X[:, idx] ** 2) ** 2 + (X[:, idx] - 1) ** 2, dim=1))/5000
    return Y

def sphere_function(X): 
    Y = 1 - (X - 0.5).norm(dim=-1, keepdim=True)
    return Y

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

def camel_function_embedded(X):
    if X.dim() == 1:
        X = X.unsqueeze(0)
    xx = X[:,0]
    yy = X[:,1]
    y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
    print(y)
    return -y.unsqueeze(-1)

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
    return torch.exp(-4*X.norm(dim=-1, keepdim=True)**2)

def sample_pre_rkhs(xs, rkhs_norm, gamma=0.2, n_basepoints=10, seed=0, kernel="SE", nu=1.5, dim=2):
    '''
    sample a pre rkhs function
    
    Args:
    xs: points to evaluate the function
    rkhs_norm: norm of the RKHS
    gamma: lengthscale of the kernel
    n_basepoints: number of base points
    seed: random seed
    kernel: kernel type
    nu: parameter of the Matern kernel
    dim: dimension of the input space
    '''
    np.random.seed(seed)
    length_scale = gamma/math.sqrt(2)
    base_point_generator = lambda n_base_points: np.random.normal(size=(n_base_points, dim))
    if kernel == "SE":
        kernel = RBF(length_scale=length_scale)
    elif kernel == "Matern":
        kernel = Matern(length_scale=length_scale, nu=nu)
    else:
        raise NotImplementedError

    pre_rkhs_function_generator = PreRKHSfunctionGenerator(kernel, base_point_generator)
    pre_rkhs_function = pre_rkhs_function_generator(rkhs_norm=rkhs_norm, n_base_points=n_basepoints)
    y = pre_rkhs_function(xs)
    if isinstance(y, float):
        y = np.array([y])
    return y[:,None]

def sample_fixed_pre_rkhs(xs, function_config):
    '''
    sample a pre rkhs function based on fixed coefficients
    
    Args:
    xs: points to evaluate the function
    function_config: dictionary with the function configuration
    
    Returns:
    y: function values at points xs
    '''
    
    path = function_config["path"]
    dict_coeffs = pickle.load(open(f"{path}/coeffs/coeffs_function_{function_config['function_number']}.pkl", "rb"))
    base_points = dict_coeffs["base_points"]
    coeffs = dict_coeffs["coeffs"]
    if function_config["kernel"] == "SE":
        kernel = RBF(length_scale=function_config["gamma"]/math.sqrt(2))
    elif function_config["kernel"] == "Matern":
        kernel = Matern(length_scale=function_config["gamma"]/math.sqrt(2), nu=function_config["nu"])
    else:
        raise NotImplementedError
    pre_rkhs_function = PreRKHSfunction(kernel, base_points, coeffs)
    xs = xs.detach().numpy()
    y = pre_rkhs_function(xs)
    if isinstance(y, float):
        y = np.array([y])
    return y[:,None]

def sample_fixed_rkhs_se_onb(xs, function_config):
    gamma = function_config["gamma"]
    path = function_config["path"]
    dict_coeffs = pickle.load(open(f"{path}/coeffs/coeffs_function_{function_config['function_number']}.pkl", "rb"))
    n_bfuncs = dict_coeffs["n_bfuncs"]
    indices_bfuncs = dict_coeffs["indices_bfuncs"]
    ys_bfuncs = np.zeros([len(xs), n_bfuncs])
    for i_bf in range(n_bfuncs):
        ys_bfuncs[:, i_bf] = se_bfunc_1d(xs.flatten(), indices_bfuncs[i_bf], gamma=gamma)
    coeffs = dict_coeffs["coeffs"]
    ys = ys_bfuncs @ coeffs
    return ys[:,None]


def observe_data(x, function_config, noise_on=True):
    #print(function_config["type"])
    if "branin_mod" in function_config["type"]:
        Y = branin_mod(x)
    elif  "sphere" in function_config["type"]:
        Y = sphere_function(x)
    elif "rosenbrock2d" in function_config["type"]:
        Y = rosenbrock_2d(x)
    elif "camel_2D" in function_config["type"]:
        Y = camel_function(x)
    elif "camel_10D" in function_config["type"]:
        Y = camel_function_embedded(x)
    elif "hartmann_6D" in function_config["type"]:
        Y = hartmann_6D(x)
    elif "gaussian_10D" in function_config["type"]:
        Y = gaussian_function(x)
    elif "pre_rkhs" in function_config["type"]:
            y = sample_fixed_pre_rkhs(x, function_config=function_config)
            Y = torch.tensor(y, dtype=torch.float).squeeze(-1)
    elif "rkhs_onb_se" in function_config["type"]:
        y = sample_fixed_rkhs_se_onb(x, function_config=function_config)
        Y = torch.tensor(y, dtype=torch.float).squeeze(-1)
    else:
        raise NotImplementedError
    if noise_on:
        noise = (torch.rand_like(Y) - 0.5) * function_config["noise_lvl"] * 2
    else:
        noise = torch.rand_like(Y) * 0
    return Y + noise
