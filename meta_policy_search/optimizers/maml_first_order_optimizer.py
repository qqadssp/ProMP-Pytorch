from meta_policy_search.utils import logger
from meta_policy_search.optimizers.base import Optimizer
import torch

class MAMLFirstOrderOptimizer(Optimizer):
    """
    Optimizer for first order methods (SGD, Adam)

    Args:
        torch_optimizer_cls (torch.optim.optimizer): desired tensorflow optimzier for training
        torch_optimizer_args (dict or None): arguments for the optimizer
        learning_rate (float): learning rate
        max_epochs: number of maximum epochs for training
        tolerance (float): tolerance for early stopping. If the loss fucntion decreases less than the specified tolerance
        after an epoch, then the training stops.
        num_minibatches (int): number of mini-batches for performing the gradient step. The mini-batch size is
        batch size//num_minibatches.
        verbose (bool): Whether to log or not the optimization process

    """

    def __init__(
            self,
            torch_optimizer_cls=torch.optim.Adam,
            torch_optimizer_args=None,
            target = None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            num_minibatches=1,
            verbose=False
            ):

        self._target = target
        if torch_optimizer_args is None:
            torch_optimizer_args = dict()
        torch_optimizer_args['lr'] = learning_rate

        self._torch_optimizer = torch_optimizer_cls(self._target, **torch_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._num_minibatches = num_minibatches # Unused
        self._verbose = verbose
        self._all_inputs = None
        self._train_op = None
        self._loss = None
        self._input_ph_dict = None
        
    def build_graph(self, loss, target, input_ph_dict):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        logger.log('The function <build_graph> is deprecated in pytorch version of this repo')
        return 0

    def loss(self, compute_stats, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        loss, _, _ = compute_stats(input_val_dict)
        return loss

    def optimize(self, compute_stats, input_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float) loss before optimization

        """

        # Overload self._batch size
        # dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)
        # Todo: reimplement minibatches

        loss_before_opt = None
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % epoch)

            loss, _, _ = compute_stats(input_dict)
            loss.backward(retain_graph=True)
            self._torch_optimizer.step()
            self._torch_optimizer.zero_grad()

            if not loss_before_opt: loss_before_opt = loss

            # if self._verbose:
            #     logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            #
            # if abs(last_loss - new_loss) < self._tolerance:
            #     break
            # last_loss = new_loss
        return loss_before_opt


class MAMLPPOOptimizer(MAMLFirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(MAMLPPOOptimizer, self).__init__(*args, **kwargs)
        self._inner_kl = None
        self._outer_kl = None

    def build_graph(self, loss, target, input_ph_dict, inner_kl=None, outer_kl=None):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (Tensor) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
            inner_kl (list): list with the inner kl loss for each task
            outer_kl (list): list with the outer kl loss for each task
        """
        logger.log('The function <build_graph> is deprecated in pytorch version of this repo')
        return 0

    def compute_stats(self, compute_stats, input_val_dict):
        """
        Computes the value the loss, the outer KL and the inner KL-divergence between the current policy and the
        provided dist_info_data

        Args:
           inputs (list): inputs needed to compute the inner KL
           extra_inputs (list): additional inputs needed to compute the inner KL

        Returns:
           (float): value of the loss
           (ndarray): inner kls - numpy array of shape (num_inner_grad_steps,)
           (float): outer_kl
        """
        loss, inner_kl, outer_kl = compute_stats(input_val_dict)
        return loss, inner_kl, outer_kl



