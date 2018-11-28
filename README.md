# ProMP-Pytorch

This is a implementation of ProMP in Pytorch, modified from the [origin ProMP](https://github.com/jonasrothfuss/ProMP). Now the ProMP algorithm can work and I am still working in others(include MAML and e-MAML). Here is the [paper](https://arxiv.org/abs/1810.06784).  

Some notes:  

1. In origin implementation, 'phs' means placeholders. It takes me half a day to realize this.  

2. As gradient descent is adopted, the weight and bias of the n-th inner adaptive step is a intermediate result because it is obtained from the gradient of the loss w.r.t the weight and bias of (n-1)-th adaptive step, which are also a intermediate result except n=1, and the gradient used in inner adaptive step are a part of compution graph of the outer optimization and can be backward. This is the most interesting thing. We can backward a gradient?!  

## Requirement

python 3.6  
pytorch 0.4  

## Train

Download this repo and run pro-mp_run_mujoco.py  

    git clone git@github.com:qqadssp/ProMP-Pytorch
    cd ProMP-Pytorch
    python3 pro-mp_run_mujoco.py

## Unfinished files

    /run_scripts/e-maml_run_mujoco.py
    /run_scripts/maml_run_mujoco.py
    /run_scripts/pro-mp_run_point_mass.py

    /meta_algos/trpo_maml.py
    /meta_algos/vpg_dice_maml.py
    /meta_algos/vpg_maml.py
    /meta_algos/dice_maml.py

    /optimizers/conjugate_gradient_optimizer.py
