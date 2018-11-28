import torch
from collections import OrderedDict

def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=torch.nn.init.xavier_uniform_,
               b_init=(torch.nn.init.constant_, 1.),
               reuse=False
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    module = OrderedDict()
    module[name + '_linear0'] = torch.nn.Linear(input_dim, hidden_sizes[0])
    module[name + '_nonlinear0'] = hidden_nonlinearity()
    for idx in range(len(hidden_sizes) - 1):
        module[name + '_linear'+str(idx+1)] = torch.nn.Linear(hidden_sizes[idx], hidden_sizes[idx+1])
        module[name + '_nonlinear'+str(idx+1)] = hidden_nonlinearity()
    module[name + '_output'] = torch.nn.Linear(hidden_sizes[-1], output_dim)
    if output_nonlinearity != None:
        model[name + '_outputnonlinear'] = output_nonlinearity()
    model = torch.nn.Sequential(module)

    for m in model:
        if isinstance(m, torch.nn.Linear):
            w_init(m.weight.data)
            b_init[0](m.bias.data, b_init[1])

    return input_var, model


def forward_mlp(output_dim,
                hidden_sizes,
                hidden_nonlinearity,
                output_nonlinearity,
                input_var,
                mlp_params,
                ):
    """
    Creates the forward pass of an mlp given the input vars and the mlp params. Assumes that the params are passed in
    order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        mlp_params (OrderedDict): OrderedDict of the params of the neural network. 

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """
    x = torch.tensor(input_var).float()
    idx = 0
    bias_added = False
    sizes = tuple(hidden_sizes) + (output_dim,)

    for name, param in mlp_params.items():
        assert str(idx) in name or (idx == len(hidden_sizes) and "output" in name)
        if "weight" in name:
            assert param.shape == (sizes[idx], x.shape[-1])
            x = torch.matmul(x, param.t())
        elif "bias" in name:
            assert param.shape == (sizes[idx],)
            x = torch.add(x, param)
            bias_added = True
        else:
            raise NameError

        if bias_added:
            if "output" in name:
                if output_nonlinearity is not None:
                    x = output_nonlinearity()(x)
            else:
                x = hidden_nonlinearity()(x)
            idx += 1
            bias_added = False
    output_var = x
    return input_var, output_var # Todo why return input_var?

