from __future__ import annotations
from dataclasses import dataclass
import json
import random, os, datetime, pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

import torch
from torch.utils import tensorboard
import tqdm
import yaml
from common.config_tools import config

from games import GameConfig, create_game
from common.heatmap import Heatmaps
from .dataset import Dataset
from methods.ms_conditions import ConditionModel, get_condition_model_by_name
from .models import GFlowMSGenerator, get_msgen_by_name
from .optimizers import GFlowMSTBOptimizer

class Trainer:
    @config
    @dataclass
    class Config:
        game_config: GameConfig
        conditions: List[str]
        sizes: List[Tuple[int, int]]
        dataset_config: Dataset.Config
        condition_model_config: Any
        generator_config: Any
        optimizer_config: GFlowMSTBOptimizer.Config
        training_steps: int
        batch_size: int
        checkpoint_period: int
        sample_render_period: int = 10
        heatmap_config: Optional[Heatmaps.Config] = None
        heatmap_render_period: int = 100
        save_path: Optional[str] = None
        name_suffix: Optional[str] = None

        def create_trainer(self):
            return Trainer(self)
    
    def __init__(self, config: Trainer.Config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.game = create_game(config.game_config)
        self.dataset = Dataset(self.game, config.conditions, config.sizes, config.dataset_config)
        self.condition_model: ConditionModel = config.condition_model_config.model_constructor(self.game, config.conditions)
        self.netG: GFlowMSGenerator = config.generator_config.model_constructor(len(self.game.tiles), len(config.conditions)).to(self.device)
        self.optG = GFlowMSTBOptimizer(self.netG, config.sizes, config.optimizer_config)
        self.config = config

        self.heatmap = None
        if config.heatmap_config is not None:
            self.heatmap = Heatmaps(config.heatmap_config)
        
        self.name = f"{self.game.name}_{self.dataset.name}_{self.condition_model.name}_{self.netG.name}"
        if config.name_suffix is not None:
            self.name += f"_{config.name_suffix}"

        save_path = self.config.save_path or "./runs/%TIME_%NAME"
        save_path = save_path.replace("%TIME", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        save_path = save_path.replace("%NAME", self.name)
        self.config.save_path = save_path

        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        yaml.dump(self.config, open(os.path.join(save_path, "config.yml"), 'w'))
        self.checkpoint_path = os.path.join(save_path, "checkpoints")
        pathlib.Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.log_path = os.path.join(save_path, "log")
        
        self.elapsed_time = 0
        self.starting_step = 1
    
    def resume(self):
        checkpoint_info_path = os.path.join(self.checkpoint_path, "checkpoint.yml")
        if os.path.exists(checkpoint_info_path):
            checkpoint_state = yaml.unsafe_load(open(checkpoint_info_path, 'r'))
            self.elapsed_time = checkpoint_state["elapsed_seconds"]
            step = checkpoint_state["step"]
            self.starting_step = step + 1
            
            self.optG.load_checkpoint(os.path.join(self.checkpoint_path, f"model_{step}.pt"))
            infos = self.dataset.load_checkpoint(os.path.join(self.checkpoint_path, f"dataset_{step}"))
            if self.heatmap is not None:
                for size, info in infos.items():
                    self.heatmap.update(size, info)
        else:
            warnings.warn("There is no checkpoint. The training will start from scratch.")
        
        self.train()
        
    def train(self):
        batch_size = self.config.batch_size

        self.writer = tensorboard.writer.SummaryWriter(log_dir=self.log_path, purge_step=self.starting_step)
        print(f"\033[1mTensorboard Command: \033[4m\033[92mtensorboard --logdir {self.log_path}\033[0m")

        self.optG.train()

        start_time = time.time() - self.elapsed_time

        pbar = tqdm.tqdm(total=self.config.training_steps, initial=self.starting_step-1, desc="Start..", dynamic_ncols=True)
        for step in range(self.starting_step, self.config.training_steps+1):
            for size_index, size in enumerate(self.dataset.sizes):
                pbar.set_description(f"{size}: Query Conditions...")
                query_conditions = self.condition_model.sample(size, batch_size)
                
                pbar.set_description(f"{size}: Ask For Levels.....")
                levels = self.optG.ask(query_conditions.to(self.device), size)

                if step%self.config.sample_render_period == 0:
                    self.writer.add_images(f"Sample/Levels_{size}", self.game.render(levels.cpu().numpy()), step)
                
                pbar.set_description(f"{size}: Update Dataset.....")
                info, added_mask, log_rewards, stats  = self.dataset.analyze_and_update(size, levels.tolist(), query_conditions)

                if self.heatmap is not None:
                    self.heatmap.update(size, info)
                
                pbar.set_description(f"{size}: Update Cond-Model..")
                self.condition_model.update(size, [item for item, is_added in zip(info, added_mask) if is_added])
                
                losses, log_z0s = {}, {}

                if size_index == 0 or len(self.dataset.clusters[size]) >= self.dataset.config.cluster_threshold.get(size, 0):
                    pbar.set_description(f"{size}: Tell Optimizer.....")
                    loss, log_z0 = self.optG.tell(log_rewards.to(self.device))
                    losses["on-policy"] = loss
                    log_z0s["on-policy"] = log_z0

                pbar.set_description(f"{size}: Sample Replay Batch")
                replay_batch = self.dataset.sample(size, batch_size)
                if replay_batch is not None:
                    replay_levels, replay_conditions, replay_log_rewards = replay_batch
                    pbar.set_description(f"{size}: Train on Replay....")
                    replay_loss, replay_log_z0 = self.optG.replay(
                        replay_conditions.to(self.device), 
                        replay_levels.to(self.device), 
                        replay_log_rewards.to(self.device))
                    losses["replay"] = replay_loss
                    log_z0s["replay"] = replay_log_z0
            
                self.writer.add_scalars(f"Training/Loss_{size}", losses, step)
                self.writer.add_scalars(f"Training/LogZ0_{size}", log_z0s, step)
                self.writer.add_scalars(f"Generation/Quality_{size}", stats, step)

            self.writer.add_scalars(f"Dataset/Size", {str(size):len(items) for size, items in self.dataset.items.items()}, step)
            self.writer.add_scalars(f"Dataset/Clusters", {str(size):len(clusters) for size, clusters in self.dataset.clusters.items()}, step)

            if self.heatmap is not None and step % self.config.heatmap_render_period == 0:
                for size in self.dataset.sizes:
                    fig = self.heatmap.render(size)
                    self.writer.add_figure(f"Heatmaps/ER_{size}", fig, step)
            
            if step % self.config.checkpoint_period == 0:
                pbar.set_description(f"Saving A Checkpoint...")
                self.optG.save_checkpoint(os.path.join(self.checkpoint_path, f"model_{step}.pt"))
                self.dataset.save_checkpoint(os.path.join(self.checkpoint_path, f"dataset_{step}"))
                yaml.dump({
                    "elapsed_seconds": time.time() - start_time,
                    "step": step
                }, open(os.path.join(self.checkpoint_path, "checkpoint.yml"), 'w'))
            
            pbar.update()