from typing import Callable, List, Optional
from torch import nn

def feedforward(input_size: int, output_size: int, hidden_sizes: Optional[List[int]] = None, activation_fn: Callable = nn.ReLU) -> nn.Sequential:
    """Generate a feed-forward network that contains a given number of hidden layers.

    Parameters
    ----------
    input_size : int
        The input size of thr first layer.
    output_size : int
        The output size of the last layer.
    hidden_sizes : Optional[List[int]], optional
        The sizes of the hidden layers. If None, no hidden layers are added. (Default: None).
    activation_fn : Callable, optional
        The activation function of the hidden layers (not added to the output layer). (Default: nn.ReLU).

    Returns
    -------
    nn.Sequential
        The feed-forward network.
    """
    if hidden_sizes is None: hidden_sizes = []
    layers = []
    size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(size, hidden_size))
        layers.append(activation_fn())
        size = hidden_size
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)