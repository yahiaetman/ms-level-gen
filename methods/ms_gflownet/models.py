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
from methods.utils import snake
from methods.common_modules import feedforward

class GFlowMSGenerator(MSGenerator):
    """The base class for all the GFlowNet Multi-size Level Generators.
    """
    def loss(self, token, log_z0: torch.Tensor, log_rewards: torch.Tensor) -> torch.Tensor:
        """Compute the loss function.

        Parameters
        ----------
        token : Any
            An object created by the generator during generation to store all the data needed to compute the loss.
        log_z0 : torch.Tensor
            The log of the source flow (Z0).
        log_rewards : torch.Tensor
            The log of the level rewards. The tensor size should be equal the number of levels generated alongside the token. 

        Returns
        -------
        torch.Tensor
            The loss.
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

class SnakeMSCEGen(GFlowMSGenerator):
    """A GFlowNet generator that traverses the level in a snake-like pattern.
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
        - "row_anchor":     whether to include an boolean input that will be
                            true at the start of every row.
        """
        cell_type: str = "GRU"
        layer_count: int = 2
        layer_size: int = 128
        embedding_size: int = 32
        row_anchor: bool = True

        @property
        def model_constructor(self):
            """Returns a function to construct a snake GFlowNet generator.
            """
            return lambda tile_count, condition_size: SnakeMSCEGen(tile_count, condition_size, self)
    
    @dataclass
    class Token:
        """A Token returned by the generator while generating levels to be used for computing the loss.
        """
        size: Tuple[int, int]           # The level size (height, width).
        forward_log_prob: torch.Tensor  # The forward log probability of the actions used to generate the level.

    def __init__(self, tile_count: int, condition_size: int, config: Optional[SnakeMSCEGen.Config] = None) -> None:
        """The generator's initializer.

        Parameters
        ----------
        tile_count : int
            The size of the game's tileset.
        condition_size : int
            The size of the condition (control) vector.
        config : Optional[SnakeMSCEGen.Config], optional
            The generator's configuration. If None, the default configuration will be used. (Default: None)
        """
        super().__init__(tile_count, condition_size)
        self.config = config or SnakeMSCEGen.Config()
        self.name = "GFLOW_SNAKE_MSCE" + self.config.cell_type.upper() + "_GEN" # The name is used to create an experiment name.
        
        self.conditional_embedding = feedforward(
            condition_size + 2, 
            self.config.embedding_size, 
            [self.config.embedding_size // 2], 
            lambda: nn.LeakyReLU(0.1, inplace=True)
        )

        self.cell_setup = get_cell_setup(self.config.cell_type)

        cells = []
        for index in range(self.config.layer_count):
            input_size = self.config.embedding_size + tile_count + int(self.config.row_anchor)
            if index != 0: input_size += self.config.layer_size
            cells.append(self.cell_setup.cell(input_size, self.config.layer_size))
        self.cells = nn.ModuleList(cells)
        
        self.action_module = feedforward(
            self.config.embedding_size + tile_count + int(self.config.row_anchor) + self.config.layer_count * self.config.layer_size,
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
    
    def forward(self, conditions: torch.Tensor, targets_or_size: Union[torch.Tensor, Tuple[int, int]], generate_token: bool = False):
        """Run a forward pass through the model for a whole level.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls) to generate (or used to generate) the level.
        targets_or_size : Union[torch.Tensor, Tuple[int, int]]
            A level to use for training, or a level size to generate.
        generate_token : bool, optional
            Should the function return a token used for loss computation? (Default: False)

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, SnakeMSCEGen.Token]]
            The generated level (and optionally, a token).
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
        if generate_token: forward_log_prob = 0
        last_y = None
        for y, x in snake(*size):
            first = y != last_y
            last_y = y
            inputs = [start, ce]
            if self.config.row_anchor:
                anchor = torch.full((batch_size, 1), float(first), device=device)
                inputs.insert(0, anchor)
            inputs = torch.cat(inputs, dim=1)
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
    
    def loss(self, token: SnakeMSCEGen.Token, log_z0: torch.Tensor, log_rewards: torch.Tensor):
        h, w = token.size
        return F.mse_loss(token.forward_log_prob + log_z0, log_rewards) / (h * w)

###########################
##### GRU-CROSS Model #####
###########################

class SnakeMSCEGRUX(GFlowMSGenerator):
    """A GRU-CROSS GFlowNet generator that traverses the level in a snake-like pattern and adds hidden connections along the other dimension.
    """

    @config
    @dataclass
    class Config:
        """The config of "SnakeMSCEGRUX".

        It contains:
        - "layer_count":    the number of recurrent layers.
        - "layer_size":     the hidden state size of each recurrent cell.
        - "embedding_size": the size of the conditional embedding.
        - "row_anchor":     whether to include an boolean input that will be
                            true at the start of every row.
        """
        layer_count: int = 2
        layer_size: int = 64
        embedding_size: int = 32
        row_anchor: bool = True

        @property
        def model_constructor(self):
            """Returns a function to construct a snake GFlowNet generator.
            """
            return lambda tile_count, condition_size: SnakeMSCEGRUX(tile_count, condition_size, self)
    
    @dataclass
    class Token:
        """A Token returned by the generator while generating levels to be used for computing the loss.
        """
        size: Tuple[int, int]           # The level size (height, width).
        forward_log_prob: torch.Tensor  # The forward log probability of the actions used to generate the level.

    def __init__(self, tile_count: int, condition_size: int, config: Optional[SnakeMSCEGRUX.Config] = None) -> None:
        """The generator's initializer.

        Parameters
        ----------
        tile_count : int
            The size of the game's tileset.
        condition_size : int
            The size of the condition (control) vector.
        config : Optional[SnakeMSCEGRUX.Config], optional
            The generator's configuration. If None, the default configuration will be used. (Default: None)
        """
        super().__init__(tile_count, condition_size)
        self.config = config or SnakeMSCEGen.Config()
        self.name = "GFLOW_SNAKE_MSCE_GRUX_GEN" # The name is used to create an experiment name.
        
        self.conditional_embedding = feedforward(
            condition_size + 2, 
            self.config.embedding_size, 
            [self.config.embedding_size // 2], 
            lambda: nn.LeakyReLU(0.1, inplace=True)
        )

        cells = []
        for index in range(self.config.layer_count):
            input_size = self.config.embedding_size + tile_count + int(self.config.row_anchor)
            if index != 0: input_size += self.config.layer_size * 2
            cells.append(nn.GRUCell(input_size, self.config.layer_size * 2))
        self.cells = nn.ModuleList(cells)
        
        self.action_module = feedforward(
            self.config.embedding_size + tile_count + int(self.config.row_anchor) + self.config.layer_count * self.config.layer_size * 2,
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
        return x, [torch.zeros((batch_size, self.config.layer_size), device=device)]*self.config.layer_count

    def step(self, x: torch.Tensor, hidden_y: List[torch.Tensor], hidden_x: List[torch.Tensor]) -> torch.Tensor:
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
        next_hidden_y, next_hidden_x = [], []
        outputs = []
        last = None
        h_chunk_sizes = [self.config.layer_size]*2
        for cell, phy, phx in zip(self.cells, hidden_y, hidden_x):
            inputs = x if last is None else torch.cat((x, last), dim=-1)
            ph = torch.cat((phy, phx), dim=-1)
            h = cell(inputs, ph)
            hy, hx = torch.split(h, h_chunk_sizes, dim=-1)
            next_hidden_y.append(hy)
            next_hidden_x.append(hx)
            last = h
            outputs.append(last)
        return self.action_module(torch.cat((x, *outputs), dim=-1)), next_hidden_y, next_hidden_x
    
    def forward(self, conditions: torch.Tensor, targets_or_size: Union[torch.Tensor, Tuple[int, int]], generate_token: bool = False):
        """Run a forward pass through the model for a whole level.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls) to generate (or used to generate) the level.
        targets_or_size : Union[torch.Tensor, Tuple[int, int]]
            A level to use for training, or a level size to generate.
        generate_token : bool, optional
            Should the function return a token used for loss computation? (Default: False)

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, SnakeMSCEGen.Token]]
            The generated level (and optionally, a token).
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
        start, hidden_x = self.zero(batch_size, device)
        hidden_y_list = [hidden_x]*size[1]
        ce = self.embed(conditions, size)
        if generate_token: forward_log_prob = 0
        last_y = None
        for y, x in snake(*size):
            first = y != last_y
            last_y = y
            inputs = [start, ce]
            if self.config.row_anchor:
                anchor = torch.full((batch_size, 1), float(first), device=device)
                inputs.insert(0, anchor)
            inputs = torch.cat(inputs, dim=1)
            logits, hidden_y_list[x], hidden_x = self.step(inputs, hidden_y_list[x], hidden_x)
            dist = Categorical(logits=logits)
            if sample:
                tiles = dist.sample()
                targets[:,y,x] = tiles
            else:
                tiles = targets[:,y,x]
            if generate_token: forward_log_prob += dist.log_prob(tiles)
            start = F.one_hot(tiles, num_classes=self.tile_count).float()
        if generate_token:
            return targets, SnakeMSCEGRUX.Token(size, forward_log_prob)
        else: 
            return targets
    
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
    
    def loss(self, token: SnakeMSCEGRUX.Token, log_z0: torch.Tensor, log_rewards: torch.Tensor):
        h, w = token.size
        return F.mse_loss(token.forward_log_prob + log_z0, log_rewards) / (h * w)
