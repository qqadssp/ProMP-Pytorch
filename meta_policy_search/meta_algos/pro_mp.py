from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer

import torch
import numpy as np
from collections import OrderedDict

class ProMP(MAMLAlgo):
    """
    ProMP Algorithm

    Args:
        policy (Policy): policy object
        name (str):  variable scope, not used in pytorch version
        learning_rate (float): learning rate for optimizing the meta-objective
        num_ppo_steps (int): number of ProMP steps (without re-sampling)
        num_minibatches (int): number of minibatches for computing the ppo gradient steps
        clip_eps (float): PPO clip range
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for annealing clip_eps. If anneal_factor < 1, clip_eps <- anneal_factor * clip_eps at each iteration
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable

    """
    def __init__(
            self,
            *args,
            name="ppo_maml",
            learning_rate=1e-3,
            num_ppo_steps=5,
            num_minibatches=1,
            clip_eps=0.2,
            target_inner_step=0.01,
            init_inner_kl_penalty=1e-2,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            **kwargs
            ):
        super(ProMP, self).__init__(*args, **kwargs)

        self.optimizer = MAMLPPOOptimizer(target=self.policy.get_param_for_optim(), learning_rate=learning_rate, max_epochs=num_ppo_steps, num_minibatches=num_minibatches)
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = init_inner_kl_penalty * np.ones(self.num_inner_grad_steps)
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps
 
        self.step_sizes = self._create_step_size_vars()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):

        likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                   dist_info_old_sym, dist_info_new_sym)
        surr_obj_adapt = -torch.mean(likelihood_ratio_adapt * torch.tensor(adv_sym).float())
        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph
        """
        logger.log('The function <build_graph> is deprecated in pytorch version of this repo')
        return 0

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        meta_op_input_dict = self._extract_input_dict_meta_op(all_samples_data, self._optimization_keys)

        # add kl_coeffs / clip_eps to meta_op_input_dict
        meta_op_input_dict['inner_kl_coeff'] = self.inner_kl_coeff

        meta_op_input_dict['clip_eps'] = self.clip_eps

        if log: logger.log("Optimizing")

        loss_before = self.optimizer.optimize(self.compute_stats, meta_op_input_dict)

        if log: logger.log("Computing statistics")
        loss_after, inner_kls, outer_kl = self.optimizer.compute_stats(self.compute_stats, meta_op_input_dict)

        if self.adaptive_inner_kl_penalty:
            if log: logger.log("Updating inner KL loss coefficients")
            self.inner_kl_coeff = self.adapt_kl_coeff(self.inner_kl_coeff, inner_kls.data.numpy(), self.target_inner_step)


        if log:
            logger.logkv('LossBefore', loss_before.data.numpy())
            logger.logkv('LossAfter', loss_after.data.numpy())
            logger.logkv('KLInner', np.mean(inner_kls.data.numpy()))
            logger.logkv('KLCoeffInner', np.mean(self.inner_kl_coeff))

    def compute_stats(self, input_dict):
        '''computation graph from sample data to surr_objs
        '''
        distribution_info_vars, current_policy_params = [], []
        all_surr_objs, all_inner_kls = [], []

        for i in range(self.meta_batch_size):
            obs = input_dict['step%i_task%i_%s' % (0, i, self._optimization_keys[0])]
            action = [input_dict['step%i_task%i_%s' % (0, i, self._optimization_keys[1])] for i in range(self.meta_batch_size)]
            adv = [input_dict['step%i_task%i_%s' % (0, i, self._optimization_keys[2])] for i in range(self.meta_batch_size)]
            dist_info_old = [dict(mean = input_dict['step%i_task%i_%s/%s' % (0, i, self._optimization_keys[3], 'mean')],
                                  log_std = input_dict['step%i_task%i_%s/%s' % (0, i, self._optimization_keys[3], 'log_std')]) for i in range(self.meta_batch_size)]

            current_policy_params.append(self.policy.policy_params) 
            dist_info_sym = self.policy.distribution_info_sym(obs, params=self.policy.policy_params)
            distribution_info_vars.append(dist_info_sym)  # step 0

        """ Inner updates"""
        for step_id in range(1, self.num_inner_grad_steps+1):
            surr_objs, kls, adapted_policy_params = [], [], []

            # inner adaptation step for each task

            for i in range(self.meta_batch_size):

                surr_loss = self._adapt_objective_sym(action[i], adv[i], dist_info_old[i], distribution_info_vars[i])
                kl_loss = torch.mean(self.policy.distribution.kl_sym(dist_info_old[i], distribution_info_vars[i]))

                adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

                adapted_policy_params.append(adapted_params_var)
                kls.append(kl_loss)
                surr_objs.append(surr_loss)

            all_surr_objs.append(surr_objs)
            all_inner_kls.append(kls)

            # dist_info_vars_for_next_step
            obs = [input_dict['step%i_task%i_%s' % (step_id, i, self._optimization_keys[0])] for i in range(self.meta_batch_size)]
            action = [input_dict['step%i_task%i_%s' % (step_id, i, self._optimization_keys[1])] for i in range(self.meta_batch_size)]
            adv = [input_dict['step%i_task%i_%s' % (step_id, i, self._optimization_keys[2])] for i in range(self.meta_batch_size)]
            dist_info_old = [dict(mean = input_dict['step%i_task%i_%s/%s' % (step_id, i, self._optimization_keys[3], 'mean')],
                                  log_std = input_dict['step%i_task%i_%s/%s' % (step_id, i, self._optimization_keys[3], 'log_std')]) for i in range(self.meta_batch_size)]

            current_policy_params = adapted_policy_params
            distribution_info_vars = [self.policy.distribution_info_sym(obs[i], params=current_policy_params[i])
                                          for i in range(self.meta_batch_size)]

        # per step: compute mean of kls over tasks
        mean_inner_kl_per_step = torch.stack([torch.mean(torch.stack(inner_kls)) for inner_kls in all_inner_kls])

        """ Outer objective """
        surr_objs, outer_kls = [], []
        inner_kl_coeff = input_dict['inner_kl_coeff']
        clip_eps = input_dict['clip_eps']

        for i in range(self.meta_batch_size):
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action[i], dist_info_old[i], distribution_info_vars[i])
            outer_kl = torch.mean(self.policy.distribution.kl_sym(dist_info_old[i], distribution_info_vars[i]))

            # clipped likelihood ratio
            clipped_obj = torch.min(likelihood_ratio * torch.tensor(adv[i]).float(), torch.clamp(likelihood_ratio, 1. - clip_eps, 1. + clip_eps) * torch.tensor(adv[i]).float())
            surr_obj = - torch.mean(clipped_obj)

            surr_objs.append(surr_obj)
            outer_kls.append(outer_kl)
        mean_outer_kl = torch.mean(torch.stack(outer_kls))
        inner_kl_penalty = torch.mean(torch.tensor(inner_kl_coeff).float() * mean_inner_kl_per_step)

        meta_objective = torch.mean(torch.stack(surr_objs, 0)) + inner_kl_penalty
        return meta_objective, mean_inner_kl_per_step, mean_outer_kl

    def adapt_kl_coeff(self, kl_coeff, kl_values, kl_target):
        if hasattr(kl_values, '__iter__'):
            assert len(kl_coeff) == len(kl_values)
            return np.array([_adapt_kl_coeff(kl_coeff[i], kl, kl_target) for i, kl in enumerate(kl_values)])
        else:
            return _adapt_kl_coeff(kl_coeff, kl_values, kl_target)

def _adapt_kl_coeff(kl_coeff, kl, kl_target):
    if kl < kl_target / 1.5:
        kl_coeff /= 2

    elif kl > kl_target * 1.5:
        kl_coeff *= 2
    return kl_coeff
