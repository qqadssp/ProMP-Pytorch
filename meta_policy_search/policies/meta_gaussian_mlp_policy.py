from meta_policy_search.policies.base import MetaPolicy
from meta_policy_search.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np
import torch
from meta_policy_search.policies.networks.mlp import forward_mlp


class MetaGaussianMLPPolicy(GaussianMLPPolicy, MetaPolicy):
    def __init__(self, meta_batch_size,  *args, **kwargs):
        self.quick_init(locals()) # store init arguments for serialization
        self.meta_batch_size = meta_batch_size

        self.pre_update_action_var = None
        self.pre_update_mean_var = None
        self.pre_update_log_std_var = None

        self.post_update_action_var = None
        self.post_update_mean_var = None
        self.post_update_log_std_var = None

        super(MetaGaussianMLPPolicy, self).__init__(*args, **kwargs)


    def build_graph(self):
        """
        Builds computational graph for policy. This is deprecated in Pytorch version, just keep this function. 
        """
        # Create pre-update policy by calling build_graph of the super class
        super(MetaGaussianMLPPolicy, self).build_graph()

        self.policy_params_keys = list(self.policy_params.keys())

    def get_action(self, observation, task=0):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.repeat(np.expand_dims(np.expand_dims(observation, axis=0), axis=0), self.meta_batch_size, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[task][0], dict(mean=agent_infos[task][0]['mean'], log_std=agent_infos[task][0]['log_std'])
        return action, agent_infos

    def get_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        Returns:
            (tuple) : A tuple containing a list of numpy arrays of action, and a list of list of dicts of agent infos
        """
        assert len(observations) == self.meta_batch_size
        if self._pre_update_mode:
            actions, agent_infos = self._get_pre_update_actions(observations)
        else:
            actions, agent_infos = self._get_post_update_actions(observations)


        assert len(actions) == self.meta_batch_size
        return actions, agent_infos

    def _get_pre_update_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

            this function is same as _get_post_update_action(), because self.policise_params_vals has already been generated
            in MetaPolicy.switch_to_pre_update() in base.py, and it will be updated after MAMLAlgo._adapt() in meta_aglos/base.py.
            just keep this function.
        """
        assert self.policies_params_vals is not None
        obs_stack = np.concatenate(observations, axis=0)
        obs_var_per_task = np.split(obs_stack, self.meta_batch_size, axis=0)

        means, actions, log_stds = [], [], []
        for idx in range(self.meta_batch_size):
            dist = self.distribution_info_sym(obs_var = obs_var_per_task[idx], params = self.policies_params_vals[idx])
            mean = (dist['mean'].data * 1.0).numpy()
            log_std = (dist['log_std'].data * 1.0).numpy()
            action = mean + np.random.normal(size=mean.shape) * np.exp(log_std)

            means.append(mean)
            log_stds.append(log_std)
            actions.append(action)

        agent_infos = [[dict(mean = means[idx][i], log_std = log_std[i]) for i in range((means[idx]).shape[0])] for idx in range(self.meta_batch_size)]
        return actions, agent_infos

    def _get_post_update_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        """
        assert self.policies_params_vals is not None
        obs_stack = np.concatenate(observations, axis=0)
        obs_var_per_task = np.split(obs_stack, self.meta_batch_size, axis=0)

        means, actions, log_stds = [], [], []
        for idx in range(self.meta_batch_size):
            dist = self.distribution_info_sym(obs_var = obs_var_per_task[idx], params = self.policies_params_vals[idx])

            mean = (dist['mean'].data * 1.0).numpy()
            log_std = (dist['log_std'].data * 1.0).numpy()
            action = mean + np.random.normal(size=mean.shape) * np.exp(log_std)

            means.append(mean)
            log_stds.append(log_std)
            actions.append(action)

        agent_infos = [[dict(mean = means[idx][i], log_std = log_std[i]) for i in range((means[idx]).shape[0])] for idx in range(self.meta_batch_size)]
        return actions, agent_infos

