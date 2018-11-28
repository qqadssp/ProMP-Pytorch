import numpy as np
import math
import torch
from meta_policy_search.policies.distributions.base import Distribution

class DiagonalGaussian(Distribution):
    """
    General methods for a diagonal gaussian distribution of this size 
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence of two multivariate 
        Gaussian distribution with diagonal covariance matrices

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as Tensor

        Returns:
            (Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """
        old_means = old_dist_info_vars["mean"] if torch.is_tensor(old_dist_info_vars['mean']) else torch.tensor(old_dist_info_vars['mean']).float()
        old_log_stds = old_dist_info_vars["log_std"] if torch.is_tensor(old_dist_info_vars['log_std']) else torch.tensor(old_dist_info_vars['log_std']).float()
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]

        old_std = torch.exp(old_log_stds)
        new_std = torch.exp(new_log_stds)

        numerator = (old_means - new_means) ** 2 + old_std ** 2 - new_std ** 2
        denominator = 2 * new_std ** 2 + 1e-8
        return torch.sum(
            numerator / denominator + new_log_stds - old_log_stds, dim=-1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices

       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """
        old_means = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]

        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        numerator = np.square(old_means - new_means) + \
                    np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as Tensor

        Returns:
            (Tensor): likelihood ratio
        """
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return torch.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        Symbolic log likelihood log p(x) of the distribution

        Args:
            x_var (Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as Tensor

        Returns:
             (numpy array): log likelihood
        """
        x_var = torch.tensor(x_var).float()
        means = dist_info_vars["mean"] if torch.is_tensor(dist_info_vars["mean"]) else torch.tensor(dist_info_vars["mean"]).float()
        log_stds = dist_info_vars["log_std"] if torch.is_tensor(dist_info_vars["log_std"]) else torch.tensor(dist_info_vars["log_std"]).float()

        zs = (x_var - means) / torch.exp(log_stds)
        return - torch.sum(log_stds, dim=-1) - 0.5 * torch.sum(zs ** 2, dim=-1) - 0.5 * self.dim * math.log(2. * math.pi)

    def log_likelihood(self, xs, dist_info):
        """
        Compute the log likelihood log p(x) of the distribution

        Args:
           x_var (numpy array): variable where to evaluate the log likelihood
           dist_info_vars (dict) : dict of distribution parameters as numpy array

        Returns:
            (numpy array): log likelihood
        """
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def entropy_sym(self, dist_info_vars):
        """
        Symbolic entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as Tensor

        Returns:
            (Tensor): entropy
        """
        log_stds = dist_info_vars["log_std"]
        return torch.reduce_sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), reduction_indices=-1)

    def entropy(self, dist_info):
        """
        Compute the entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as numpy array

        Returns:
          (numpy array): entropy
        """
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def sample(self, dist_info):
        """
        Draws a sample from the distribution

        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array

        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim,)), ("log_std", (self.dim,))]
