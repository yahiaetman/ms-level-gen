from typing import Any, Tuple
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, tile_count: int, condition_size: int, config: Any = None) -> None:
        super().__init__()
        self.name = ""
        self.tile_count = tile_count
        self.condition_size = condition_size
    
    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, conditions: torch.Tensor) -> torch.Tensor:
        pass

class MSGenerator(Generator):
    def generate(self, conditions: torch.Tensor, size: Tuple[int, int], config: Any = None) -> torch.Tensor:
        pass