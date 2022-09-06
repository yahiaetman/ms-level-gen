from __future__ import annotations
from dataclasses import dataclass
from common.config_tools import config
from typing import Optional
import torch
from torch import optim

from .models import SeqMSGenerator

class SeqMSOptimizer:
    """An optimizer class to train an Auto-regressive sequence-based Multi-size level generator.
    """

    @config
    @dataclass
    class Config:
        """The optimizer configuration.

        It contains:
        - "lr":      the learning rate of the generator.
        """
        lr: float = 1e-3

    def __init__(self, netG: SeqMSGenerator, config: Optional[SeqMSOptimizer.Config] = None) -> None:
        """The optimizer's initializer.

        Parameters
        ----------
        netG : SeqMSGenerator
            The network to optimize.
        config : Optional[SeqMSOptimizer.Config], optional
            The optimizer's configuration. If None, the default values will be used. (Default: None)
        """
        self.config = config or SeqMSOptimizer.Config()
        self.name = "MSAR" # A name for the optimizer to use in the experiment name
        self.device = netG.device
        self.netG = netG
        self.optG = optim.RMSprop(netG.parameters(), self.config.lr)
        
    def train(self):
        """Set the networks to training mode.
        """
        self.netG.train()
    
    def step(self, conditions: torch.Tensor, targets: torch.Tensor) -> float:
        """Apply a training step

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions corresponding to the given levels.
        targets : torch.Tensor
            The level to train on.

        Returns
        -------
        float
            The loss value.
        """
        _, loss = self.netG(conditions, targets)
        self.optG.zero_grad()
        loss.backward()
        self.optG.step()
        return loss.item()

    def save_checkpoint(self, path: str):
        """Save the optimizer and its content to a checkpoint file.

        Parameters
        ----------
        path : str
            The path to the checkpoint file to save.
        """
        torch.save({
            "netG": self.netG.state_dict(),
            "optG": self.optG.state_dict(),
        }, path)
    
    
    def load_checkpoint(self, path: str):
        """Load the optimizer and its content from a checkpoint file.

        Parameters
        ----------
        path : str
            The path to the checkpoint file.
        """
        data = torch.load(path)
        self.netG.load_state_dict(data["netG"])
        self.optG.load_state_dict(data["optG"])