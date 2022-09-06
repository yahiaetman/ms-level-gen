from __future__ import annotations
from dataclasses import dataclass
from common.config_tools import config
from typing import Any, List, Optional, Tuple
import warnings
import torch
from torch import nn, optim

from .models import GFlowMSGenerator

class LogZ0Net(nn.Module):
    """A network to estimate the source flow for the GFlowNet.
    """

    def __init__(self, condition_size: int, hidden_size: int = 32):
        """The network initializer.

        Parameters
        ----------
        condition_size : int
            The size of the condition (control) vector.
        hidden_size : int, optional
            The size of the hidden layer, by default 32
        """
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.LeakyReLU(inplace=True)
        )
        self.log_z0 = nn.Linear(hidden_size, 1)
        self.log_z0.weight.data.zero_()
        self.log_z0.bias.data.zero_()
    
    def forward(self, condition: torch.Tensor):
        return self.log_z0(self.hidden(condition))[:,0]

class GFlowMSTBOptimizer:
    """An optimizer class to train a GFlowNet Multi-size level generator.
    """

    @config
    @dataclass
    class Config:
        """The optimizer configuration.

        It contains:
        - "log_z0_hidden":  the hidden size of the source flow estimation network.
        - "log_z0_lr":      the learning rate of the source flow estimation network.
        - "policy_lr":      the learning rate of the generator.
        """
        log_z0_hidden: int = 32
        log_z0_lr: float = 1e-2
        policy_lr: float = 1e-3
    
    @dataclass
    class Token:
        """A token to store any information needed to optimize the networks after a forward pass.

        It contains:
        - "conditions": the conditions used to generate the levels.
        - "size": the generated levels' size.
        - "gen_token": the generator's token.
        """
        conditions: torch.tensor
        size: Tuple[int, int]
        gen_token: Any

    def __init__(self, netG: GFlowMSGenerator, sizes: List[Tuple[int, int]], config: Optional[GFlowMSTBOptimizer.Config] = None) -> None:
        """The optimizer's initializer.

        Parameters
        ----------
        netG : GFlowMSGenerator
            The network to optimize.
        sizes : List[Tuple[int, int]]
            The level sizes (used to create the source flow estimation networks).
        config : Optional[GFlowMSTBOptimizer.Config], optional
            The optimizer's configuration. If None, the default values will be used. (Default: None)
        """
        self.config = config or GFlowMSTBOptimizer.Config()
        self.name = "MSTB" # A name for the optimizer to use in the experiment name
        self.device = netG.device
        self.netG = netG
        self.optG = optim.RMSprop(netG.parameters(), self.config.policy_lr)
        self.netsLogZ0 = {size: LogZ0Net(netG.condition_size, self.config.log_z0_hidden).to(self.device) for size in sizes}
        self.optsLogZ0 = {size: optim.RMSprop(netLogZ0.parameters(), self.config.log_z0_lr) for size, netLogZ0 in self.netsLogZ0.items()}
        self.token = None
    
    def train(self):
        """Set the networks to training mode.
        """
        self.netG.train()
        for netLogZ0 in self.netsLogZ0.values():
            netLogZ0.train()

    def ask(self, conditions: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Ask the optimizer generate levels for the given conditions (controls) and sizes, then wait for our feedback on the level's quality.

        Parameters
        ----------
        conditions : torch.Tensor
            The conditions (controls) to generate the levels for.
        size : Tuple[int, int]
            The requested level sizes.

        Returns
        -------
        torch.Tensor
            The generated levels.
        """
        assert size in self.netsLogZ0, "the requested size must be one of the sizes given to the trainers initializer."
        levels, gen_token = self.netG(conditions, size, generate_token=True)
        self.token = GFlowMSTBOptimizer.Token(conditions, size, gen_token)
        return levels 
    
    def tell(self, log_rewards: torch.Tensor) -> float:
        """Update the networks based on the given feedback (log reward). The "ask" function must be called before this.
        The number of rewards must be equal the number of levels generated by the last "ask" call.

        Parameters
        ----------
        log_rewards : torch.Tensor
            The log reward for each level.

        Returns
        -------
        Optional[Tuple[float, float]]
            The trajectory balance loss and log source flow.
            It would returrn None if an 'ask' was not called before this function. 
        """
        if self.token is None:
            warnings.warn("You must call ask before calling tell.")
            return
        assert self.token.conditions.shape[0] == log_rewards.shape[0], "The number of rewards must match the number of level generated by 'ask'."
        size = self.token.size
        netLogZ0 = self.netsLogZ0[size]
        optLogZ0 = self.optsLogZ0[size]
        log_z0 = netLogZ0(self.token.conditions)
        loss = self.netG.loss(self.token.gen_token, log_z0, log_rewards)
        self.optG.zero_grad()
        optLogZ0.zero_grad()
        loss.backward()
        optLogZ0.step()
        self.optG.step()
        self.token = None
        return loss.item(), log_z0.mean().item()
    
    def replay(self, conditions: torch.Tensor, targets: torch.Tensor, log_rewards: torch.Tensor) -> float:
        """Train the networks using experience replay.

        Parameters
        ----------
        conditions : torch.Tensor
            The level properties corresponding to the conditions (controls).
        targets : torch.Tensor
            The levels to learn.
        log_rewards : torch.Tensor
            The levels' log reward.

        Returns
        -------
        Tuple[float, float]
            The trajectory balance loss and log source flow.
        """
        _, h, w = targets.shape
        size = (h, w)
        netLogZ0 = self.netsLogZ0[size]
        optLogZ0 = self.optsLogZ0[size]
        _, gen_token = self.netG.forward(conditions, targets, generate_token=True)
        log_z0 = netLogZ0(conditions)
        loss = self.netG.loss(gen_token, log_z0, log_rewards)
        self.optG.zero_grad()
        optLogZ0.zero_grad()
        loss.backward()
        optLogZ0.step()
        self.optG.step()
        return loss.item(), log_z0.mean().item()
    
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
            "netLogZ0": {size: netLogZ0.state_dict() for size, netLogZ0 in self.netsLogZ0.items()},
            "optLogZ0": {size: optLogZ0.state_dict() for size, optLogZ0 in self.optsLogZ0.items()},
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
        netLogZ0_states = data["netLogZ0"]
        optLogZ0_states = data["optLogZ0"]
        for size, netLogZ0 in self.netsLogZ0.items():
            if size in netLogZ0_states:
                netLogZ0.load_state_dict(netLogZ0_states[size])
            else:
                warnings.warn(f"netLogZ0 for the size {size} is not available in the checkpoint.")
            if size in optLogZ0_states:
                self.optsLogZ0[size].load_state_dict(optLogZ0_states[size])
            else:
                warnings.warn(f"optLogZ0 for the size {size} is not available in the checkpoint.")