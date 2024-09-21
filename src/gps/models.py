import torch
import gpytorch
import botorch
gpytorch.settings.debug._state = False

#####################################################################################################
## Exact GP models
#####################################################################################################

class ExactGPSEModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPSEModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean
            
    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



#####################################################################################################
## Exact GP Matern 52 Model Class
#####################################################################################################
    
class ExactGPMatern32Model(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPMatern32Model, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu = 1.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean
        if noise_hyperprior is not None:
            likelihood.noise = noise_hyperprior.mean
            likelihood.noise_covar.raw_noise.requires_grad_(False)
            
    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class ExactGPMatern52Model(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPMatern52Model, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu = 2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean
        if noise_hyperprior is not None:
            likelihood.noise = noise_hyperprior.mean
            likelihood.noise_covar.raw_noise.requires_grad_(False)
            
    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)