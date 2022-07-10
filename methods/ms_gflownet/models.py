from __future__ import annotations
from dataclasses import dataclass
from common.config_tools import config
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import warnings
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from methods.generator import MSGenerator
from methods.utils import feedforward, snake

class GFlowMSGenerator(MSGenerator):
    def loss(self, token, log_z0: torch.Tensor, log_rewards: torch.Tensor) -> torch.Tensor:
        pass

@dataclass
class CellSetup:
    cell: Type[nn.Module]
    zero: Callable[[int, int, torch.device], Any]
    output: Callable[[Any], torch.Tensor]

def get_cell_setup(name: str) -> CellSetup:
    return {
        "gru": CellSetup(
            nn.GRUCell, 
            ( lambda batch_size, hidden_size, device: torch.zeros((batch_size, hidden_size), device=device) ),
            ( lambda h: h )
        ),
        "lstm": CellSetup(
            nn.LSTMCell,
            ( lambda batch_size, hidden_size, device: (torch.zeros((batch_size, hidden_size), device=device),)*2 ),
            ( lambda hc: hc[0] )
        )
    }[name.lower()]

class SnakeMSCEGen(GFlowMSGenerator):
    @config
    @dataclass
    class Config:
        cell_type: str = "GRU"
        layer_count: int = 2
        layer_size: int = 128
        embedding_size: int = 32

        @property
        def model_constructor(self):
            return lambda tile_count, condition_size: SnakeMSCEGen(tile_count, condition_size, self)
    
    @dataclass
    class Token:
        size: Tuple[int, int]
        forward_log_prob: torch.Tensor

    def __init__(self, tile_count: int, condition_size: int, config: Optional[SnakeMSCEGen] = None) -> None:
        super().__init__(tile_count, condition_size)
        self.config = config or SnakeMSCEGen.Config()
        self.name = "GFLOW_SNAKE_MSCE" + self.config.cell_type.upper() + "_GEN"
        
        self.conditional_embedding = feedforward(
            condition_size + 2, 
            self.config.embedding_size, 
            [self.config.embedding_size // 2], 
            lambda: nn.LeakyReLU(0.1, inplace=True)
        )

        self.cell_setup = get_cell_setup(self.config.cell_type)

        cells = []
        for index in range(self.config.layer_count):
            input_size = self.config.embedding_size + tile_count + 1
            if index != 0: input_size += self.config.layer_size
            cells.append(self.cell_setup.cell(input_size, self.config.layer_size))
        self.cells = nn.ModuleList(cells)
        
        self.action_module = feedforward(
            self.config.embedding_size + tile_count + 1 + self.config.layer_count * self.config.layer_size,
            self.tile_count, [32], lambda: nn.LeakyReLU(0.01, inplace=True)
        )
        last_layer: nn.Linear = self.action_module[-1]
        last_layer.weight.data.zero_()
        last_layer.bias.data.zero_()
    
    def embed(self, conditions: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        size_tensor = torch.tensor([list(size)], dtype=conditions.dtype, device=conditions.device).tile([conditions.shape[0], 1])
        embedding = self.conditional_embedding(torch.cat([conditions, size_tensor], dim=1))
        return embedding
    
    def zero(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.zeros((batch_size, self.tile_count), device=device)
        return x, [self.cell_setup.zero(batch_size, self.config.layer_size, device)]*self.config.layer_count

    def step(self, x: torch.Tensor, hidden: List[torch.Tensor]) -> torch.Tensor:
        next_hidden = []
        outputs = []
        last = None
        for cell, ph in zip(self.cells, hidden):
            inputs = x if last is None else torch.cat((x, last), dim=-1)
            h = cell(inputs, ph)
            next_hidden.append(h)
            last = self.cell_setup.output(h)
            outputs.append(last)
        return self.action_module(torch.cat((x, *outputs), dim=-1)), next_hidden
    
    def forward(self, conditions: torch.Tensor, targets_or_size: Union[torch.Tensor, Tuple[int, int]], generate_token: bool = False):
        batch_size = conditions.shape[0]
        device = conditions.device
        if isinstance(targets_or_size, torch.Tensor):
            sample = False
            targets = targets_or_size
            size = targets.shape[1:]
        else:
            sample = True
            size = targets_or_size
            targets = torch.empty((batch_size, *size), dtype=torch.int64, device=device)
        start, hidden = self.zero(batch_size, device)
        ce = self.embed(conditions, size)
        if generate_token: forward_log_prob = 0
        for y, x, first in snake(*size):
            inputs = torch.cat((torch.full((batch_size,1), float(first), device=device), start, ce), dim=1)
            logits, hidden = self.step(inputs, hidden)
            dist = Categorical(logits=logits)
            if sample:
                tiles = dist.sample()
                targets[:,y,x] = tiles
            else:
                tiles = targets[:,y,x]
            if generate_token: forward_log_prob += dist.log_prob(tiles)
            start = F.one_hot(tiles, num_classes=self.tile_count).float()
        if generate_token:
            return targets, SnakeMSCEGen.Token(size, forward_log_prob)
        else: 
            return targets
    
    def generate(self, conditions: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        with torch.no_grad():
            return self(conditions, size)
    
    def loss(self, token: SnakeMSCEGen.Token, log_z0: torch.Tensor, log_rewards: torch.Tensor):
        h, w = token.size
        return F.mse_loss(token.forward_log_prob + log_z0, log_rewards) / (h * w)







def get_msgen_by_name(name: str) -> Type[GFlowMSGenerator]:
    return {
        "snakemscegen": SnakeMSCEGen,
    }[name.lower()]

        