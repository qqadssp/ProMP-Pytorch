from meta_policy_search import utils
from meta_policy_search.policies.base import Policy
import torch
from collections import OrderedDict
import numpy as np


class MetaAlgo(object):
    """
    Base class for algorithms

    Args:
        policy (Policy) : policy object
    """

    def __init__(self, policy):
        assert isinstance(policy, Policy)
        self.policy = policy
        self._optimization_keys = None

    def build_graph(self):
        """
        Creates meta-learning computation graph

        Pseudocode::

            for task in meta_batch_size:
                make_vars
                init_dist_info_sym
            for step in num_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_dist_info_sym
            set objectives for optimizer
        """
        raise NotImplementedError

    def make_vars(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        raise NotImplementedError

    def _adapt_sym(self, surr_obj, params_var):
        """
        Creates the symbolic representation of the policy after one gradient step towards the surr_obj

        Args:
            surr_obj (op) : tensorflow op for task specific (inner) objective
            params_var (dict) : dict of placeholders for current policy params

        Returns:
            (dict):  dict of Tensors for adapted policy params
        """
        raise NotImplementedError

    def _adapt(self, samples):
        """
        Performs MAML inner step for each task and stores resulting gradients # (in the policy?)

        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task

        Returns:
            None
        """
        raise NotImplementedError

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        raise NotImplementedError



class MAMLAlgo(MetaAlgo):
    """
    Provides some implementations shared between all MAML algorithms
    
    Args:
        policy (Policy): policy object
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(self, policy, inner_lr=0.1, meta_batch_size=20, num_inner_grad_steps=1, trainable_inner_step_size=False):
        super(MAMLAlgo, self).__init__(policy)

        assert type(num_inner_grad_steps) and num_inner_grad_steps >= 0
        assert type(meta_batch_size) == int

        self.inner_lr = float(inner_lr)
        self.meta_batch_size = meta_batch_size
        self.num_inner_grad_steps = num_inner_grad_steps
        self.trainable_inner_step_size = trainable_inner_step_size #TODO: make sure this actually works

        self.adapt_input_ph_dict = None
        self.adapted_policies_params = None
        self.step_sizes = None

    def _make_input_placeholders(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task, 
            and for convenience, a list containing all placeholders created
        """
        logger.log('The function <_make_input_placehoders> is deprecated in pytorch version of this repo')
        return 0

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        raise NotImplementedError

    def _build_inner_adaption(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        logger.log('The function <_build_inner_adaption> is deprecated in pytorch version of this repo')
        return 0

    def _adapt_sym(self, surr_obj, params_var):
        """
        Creates the symbolic representation of the policy after one gradient step towards the surr_obj

        Args:
            surr_obj (op) : tensorflow op for task specific (inner) objective
            params_var (dict) : dict of Tensors for current policy params

        Returns:
            (dict):  dict of Tensors for adapted policy params
        """
        # TODO: Fix this if we want to learn the learning rate (it isn't supported right now).

        # gradient descent
        adapted_policy_params_dict = OrderedDict()
        g = torch.autograd.grad(surr_obj, [params_var[key] for key in params_var.keys()], retain_graph=True)
        for idx, key in enumerate(params_var.keys()):
            adapted_policy_params_dict[key] = params_var[key] - self.step_sizes[key].data * g[idx]

        return adapted_policy_params_dict

    def _adapt(self, samples):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(samples) == self.meta_batch_size
        assert [sample_dict.keys() for sample_dict in samples]

        # prepare feed dict
        input_dict = self._extract_input_dict(samples, self._optimization_keys, prefix='adapt')

        adapted_policies_params = []
        for i in range(self.meta_batch_size):
            obs = input_dict['%s_task%i_%s' % ('adapt', i, self._optimization_keys[0])]
            action = input_dict['%s_task%i_%s' % ('adapt', i, self._optimization_keys[1])]
            adv = input_dict['%s_task%i_%s' % ('adapt', i, self._optimization_keys[2])]
            dist_info_old = dict(mean = input_dict['%s_task%i_%s/%s' % ('adapt', i, self._optimization_keys[3], 'mean')],
                                 log_std = input_dict['%s_task%i_%s/%s' % ('adapt', i, self._optimization_keys[3], 'log_std')])

            distribution_info_new = self.policy.distribution_info_sym(obs, params=self.policy.policies_params_vals[i]) # dict(mean = mean, log_std = log_std), requires_grad=True
            # inner surrogate objective
            surr_obj_adapt = self._adapt_objective_sym(action, adv, dist_info_old, distribution_info_new)

            # get operation for adapted (post-update) policy
            adapted_policy_param = self._adapt_sym(surr_obj_adapt, self.policy.policies_params_vals[i])
            adapted_policies_params.append(adapted_policy_param)

        # store the new parameter values in the policy
        self.policy.update_task_parameters(adapted_policies_params)


    def _extract_input_dict(self, samples_data_meta_batch, keys, prefix=''):
        """
        Re-arranges a list of dicts containing the processed sample data into a OrderedDict that can be matched
        with a placeholder dict for creating a feed dict

        Args:
            samples_data_meta_batch (list) : list of dicts containing the processed data corresponding to each meta-task
            keys (list) : a list of keys that should exist in each dict and whose values shall be extracted
            prefix (str): prefix to prepend the keys in the resulting OrderedDict

        Returns:
            OrderedDict containing the data from all_samples_data. The data keys follow the naming convention:
                '<prefix>_task<task_number>_<key_name>'
        """
        assert len(samples_data_meta_batch) == self.meta_batch_size

        input_dict = OrderedDict()

        for meta_task in range(self.meta_batch_size):
            extracted_data = utils.extract(
                samples_data_meta_batch[meta_task], *keys
            )

            # iterate over the desired data instances and corresponding keys
            for j, (data, key) in enumerate(zip(extracted_data, keys)):
                if isinstance(data, dict):
                    # if the data instance is a dict -> iterate over the items of this dict
                    for k, d in data.items():
                        assert isinstance(d, np.ndarray)
                        input_dict['%s_task%i_%s/%s' % (prefix, meta_task, key, k)] = d

                elif isinstance(data, np.ndarray):
                    input_dict['%s_task%i_%s'%(prefix, meta_task, key)] = data
                else:
                    raise NotImplementedError
        return input_dict

    def _extract_input_dict_meta_op(self, all_samples_data, keys):
        """
        Creates the input dict for all the samples data required to perform the meta-update

        Args:
            all_samples_data (list):list (len = num_inner_grad_steps + 1) of lists (len = meta_batch_size) containing
                                    dicts that hold processed samples data
            keys (list): a list of keys (str) that should exist in each dict and whose values shall be extracted

        Returns:

        """
        assert len(all_samples_data) == self.num_inner_grad_steps + 1

        meta_op_input_dict = OrderedDict()
        for step_id, samples_data in enumerate(all_samples_data):  # these are the gradient steps
            dict_input_dict_step = self._extract_input_dict(samples_data, keys, prefix='step%i'%step_id)
            meta_op_input_dict.update(dict_input_dict_step)

        return meta_op_input_dict

    def _create_step_size_vars(self):
        # Step sizes
        step_sizes = OrderedDict()
        for key, param in self.policy.policy_params.items():
            step_sizes[key] = torch.nn.Parameter(torch.ones(param.shape) * self.inner_lr, requires_grad = True)
        return step_sizes
