"""Module providing analytic acquisition functions for single-outcome models."""

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from src.utils.transform_utils import n_sphere_to_cartesian

class UpperConfidenceBoundSphere(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model,
        beta,
        center,
        radius,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize
        self.center = center
        self.radius = radius

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        if len(X.shape) == 1:
            X = self.center + self.radius * X
        else:
            X = n_sphere_to_cartesian(X, self.radius, self.center)
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta * sigma
        return ucb

class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model,
        beta=2,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        return mean + self.beta*sigma

class MaximumVarianceSphere(AnalyticAcquisitionFunction):
    r"""Single-outcome Maximum Variance (MV).

    Analytic lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `MV(x) = sigma(x)`, where `sigma` is the standard deviation.
    """
    def __init__(
        self,
        model,
        center,
        radius,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize
        self.center = center
        self.radius = radius

    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        X = n_sphere_to_cartesian(X, self.radius, self.center)
        _ , sigma = self._mean_and_sigma(X)
        return sigma

class MaximumVariance(AnalyticAcquisitionFunction):
    r"""Single-outcome Maximum Variance (MV).

    Analytic lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `MV(x) = sigma(x)`, where `sigma` is the standard deviation.
    """

    def __init__(
        self,
        model,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        _ , sigma = self._mean_and_sigma(X)
        return sigma

class Mean(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).
    mu(x)
    """

    def __init__(
        self,
        model,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, _ = self._mean_and_sigma(X)
        return mean

class MeanSphere(AnalyticAcquisitionFunction):
    r"""Single-outcome Mean.
    Analytic lower confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `Mean(x) = mu(x) `, where `mu` is the posterior mean.
    """

    def __init__(
        self,
        model,
        center,
        radius,
        posterior_transform = None,
        maximize = True,
    ) :
        r"""Single-outcome Mean.
        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize
        self.center = center
        self.radius = radius

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""Evaluate the Upper Confidence Bound on the candidate set X.
        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        X = n_sphere_to_cartesian(X, self.radius, self.center)
        mean, _ = self._mean_and_sigma(X)
        return mean
