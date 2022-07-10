from typing import Callable, List, Optional
from torch import nn

def snake(height: int, width: int):
    reverse_if = lambda cond, it: reversed(it) if cond else it
    for y in range(height):
        first = True
        for x in reverse_if(y&1!=0, range(width)):
            yield (y, x, first)
            first = False

def feedforward(input_size: int, output_size: int, hidden_sizes: Optional[List[int]] = None, activation_fn: Callable = nn.ReLU) -> nn.Sequential:
    if hidden_sizes is None: hidden_sizes = []
    layers = []
    size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(size, hidden_size))
        layers.append(activation_fn())
        size = hidden_size
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)