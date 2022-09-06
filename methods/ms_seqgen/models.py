from __future__ import annotations
from dataclasses import dataclass
from common.config_tools import config
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from methods.generator import MSGenerator
from methods.utils import snake
from methods.common_modules import feedforward

class SeqMSGenerator(MSGenerator):
    """The base class for all the Auto-regressive Sequence-based Multi-size Level Generators.
    """
    pass

@dataclass
class CellSetup:
    """A class containing all the information to create and use a recurrent cell.
    It includes:
    - "cell":   The cell type.
    - "zero":   A function to create an initial state of the recurrent cell.
    - "output": A function that extracts the output from the cell state.
    """
    cell: Type[nn.Module]
    zero: Callable[[int, int, torch.device], Any]
    output: Callable[[Any], torch.Tensor]

def get_cell_setup(name: str) -> CellSetup:
    """Given a cell name, returns the setup needed to use the cell

    Parameters
    ----------
    name : str
        The cell name.

    Returns
    -------
    CellSetup
        The cell setup.
    """
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

class SnakeMSCEGen(SeqMSGenerator):
    """An Auto-regressive Sequence-based generator that traverses the level in a snake-like pattern.
    """

    @config
    @dataclass
    class Config:
        """The config of "SnakeMSCEGen".

        It contains:
        - "cell_type":      the name of the recurrent cell.
        - "layer_count":    the number of recurrent layers.
        - "layer_size":     the hidden state size of each recurrent cell.
        - "embedding_size": the size of the conditional embedding.
        """
        cell_type: str = "GRU"
        layer_count: int = 2
        layer_size: int = 128
        embedding_size: int = 32

        @property
        def model_constructor(self):
            """Returns a function to construct a snake GFlowNet generator.
            """
            return lambda tile_count, condition_size: SnakeMSCEGen(tile_count, condition_size, self)

    def __init__(self, tile_count: int, condition_size: int, config: Optional[SnakeMSCEGen] = None) -> None:
        """The generator's initializer.

        Parameters
        ----------
        tile_count : int
            The size of the game's tileset.
        condition_size : int
            The size of the condition (control) vector.
        config : Optional[SnakeMSCEGen], optional
            The generator's configuration. If None, the default configuration will be used. (Default: None)
        """
        super().__init__(tile_count, condition_size)
        self.config = config or SnakeMSCEGen.Config()
        self.name = "SEQGEN_SNAKE_MSCE" + self.config.cell_type.upper() + "_GEN" # The name is used to create an experiment name.
        
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
        """Create a conditional embedding from the conditions (controls) and the level size.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls).
        size : Tuple[int, int]
            The level size.

        Returns
        -------
        torch.Tensor
            The conditional embedding.
        """
        size_tensor = torch.tensor([list(size)], dtype=conditions.dtype, device=conditions.device).tile([conditions.shape[0], 1])
        embedding = self.conditional_embedding(torch.cat([conditions, size_tensor], dim=1))
        return embedding
    
    def zero(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Create the start symbol and the initial hidden state of the recurrent part of the model.

        Parameters
        ----------
        batch_size : int
            The initial state batch size.
        device : torch.device
            The device on which the initial state should reside.

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            The start symbol and the initial hidden state of the recurrent part of the model.
        """
        x = torch.zeros((batch_size, self.tile_count), device=device)
        return x, [self.cell_setup.zero(batch_size, self.config.layer_size, device)]*self.config.layer_count

    def step(self, x: torch.Tensor, hidden: List[torch.Tensor]) -> torch.Tensor:
        """Compute a step through the recurrent module.

        Parameters
        ----------
        x : torch.Tensor
            The last generated tile (or the start symbol of nothing was generated yet) in addition to
            the condition embedding and a boolean that marks the start of every row.
        hidden : List[torch.Tensor]
            The hidden state.

        Returns
        -------
        torch.Tensor
            The action logits to pick the next tile.
        """
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
    
    def forward(self, conditions: torch.Tensor, targets_or_size: Union[torch.Tensor, Tuple[int, int]]):
        """Run a forward pass through the model for a whole level.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls) to generate (or used to generate) the level.
        targets_or_size : Union[torch.Tensor, Tuple[int, int]]
            A level to use for training, or a level size to generate.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The generated level and the loss.
        """
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
        loss = 0
        last_y = None
        for y, x in snake(*size):
            first = y != last_y
            last_y = y
            inputs = torch.cat((torch.full((batch_size,1), float(first), device=device), start, ce), dim=1)
            logits, hidden = self.step(inputs, hidden)
            if sample:
                tiles = Categorical(logits=logits).sample()
                targets[:,y,x] = tiles
            else:
                tiles = targets[:,y,x]
                loss += F.cross_entropy(logits, tiles)
            start = F.one_hot(tiles, num_classes=self.tile_count).float()
        if sample:
            return targets
        else: 
            return targets, loss / (size[0] * size[1])
    
    def generate(self, conditions: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Generate levels.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls) to generate the level.
        size : Tuple[int, int]
            The requested level size.

        Returns
        -------
        torch.Tensor
            The generated levels.
        """
        with torch.no_grad():
            return self(conditions, size)


        