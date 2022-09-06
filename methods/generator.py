from typing import Any, Tuple
import torch
from torch import nn


class Generator(nn.Module):
    """The base class for all the generators (Single Level Size only).
    """

    def __init__(self, tile_count: int, condition_size: int, config: Any = None) -> None:
        """Initialize the generator class

        Parameters
        ----------
        tile_count : int
            The tileset size.
        condition_size : int
            The size of the condition (control) vector.
        config : Any, optional
            The generator configuration, by default None
        """
        super().__init__()
        self.name = ""
        self.tile_count = tile_count
        self.condition_size = condition_size
    
    @property
    def device(self):
        """A utility property to simplify getting the device on which the generator resides.

        Returns
        -------
        nn.Device
            The device on which the generator resides.
        """
        return next(self.parameters()).device

    def generate(self, conditions: torch.Tensor) -> torch.Tensor:
        """Generate levels for the given conditions (controls).

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls). (Shape: Batch_Size x Condition_Size).

        Returns
        -------
        torch.Tensor
            The generated levels. (Shape: Batch_Size x Tileset_Size x Height x Width).
        """
        pass

class MSGenerator(Generator):
    """Generate levels for the given conditions (controls).
    """

    def generate(self, conditions: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Generate levels for the given conditions (controls) and size.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls). (Shape: Batch_Size x Condition_Size).
        size : Tuple[int, int]
            The level size (Height x Width).

        Returns
        -------
        torch.Tensor
            The generated levels. (Shape: Batch_Size x Tileset_Size x Height x Width).
        """
        pass