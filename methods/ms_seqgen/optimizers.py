from __future__ import annotations
from dataclasses import dataclass
from common.config_tools import config
from typing import Optional
import torch
from torch import optim

from .models import SeqMSGenerator

class SeqMSOptimizer:
    @config
    @dataclass
    class Config:
        lr: float = 1e-3

    def __init__(self, netG: SeqMSGenerator, config: Optional[SeqMSOptimizer.Config] = None) -> None:
        self.config = config or SeqMSOptimizer.Config()
        self.name = "MSAR"
        self.device = netG.device
        self.netG = netG
        self.optG = optim.RMSprop(netG.parameters(), self.config.lr)
        
    def train(self):
        self.netG.train()
    
    def step(self, conditions: torch.Tensor, targets: torch.Tensor) -> float:
        _, loss = self.netG(conditions, targets)
        self.optG.zero_grad()
        loss.backward()
        self.optG.step()
        return loss.item()

    def save_checkpoint(self, path: str):
        torch.save({
            "netG": self.netG.state_dict(),
            "optG": self.optG.state_dict(),
        }, path)
    
    
    def load_checkpoint(self, path: str):
        data = torch.load(path)
        self.netG.load_state_dict(data["netG"])
        self.optG.load_state_dict(data["optG"])