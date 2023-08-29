from __future__ import annotations
from dataclasses import dataclass, field
import os, datetime, pathlib
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from torch.utils import tensorboard
import tqdm
import yaml
import numpy as np
from common.config_tools import config

from games import GameConfig
from common.heatmap import Heatmaps
from .dataset import Dataset
from methods.ms_conditions import ConditionModel
from .models import GFlowMSGenerator
from .optimizers import GFlowMSTBOptimizer

class Trainer:
    """Train a GFlowNet Multi-size generator.
    It orchestrates the whole training process and contains the training loop.
    """
    
    @config
    @dataclass
    class Config:
        """The trainer configuration.

        It contains:
        - "game_config":            the config used to create the game.
        - "conditions":             the names of the level properties to control.
        - "sizes":                  the training sizes.
        - "dataset_config":         the config of the dataset (experience replay buffer).
        - "condition_model_config": the config of the condition model used to sample conditions during training.
        - "generator_config":       the config used to create the generator.
        - "optimizer_config":       the config used to create the optimizer.
        - "training_steps":         the number of training steps (each step goes through every training size).
        - "batch_size":             the training batch size.
        - "checkpoint_period":      the number of steps before new checkpoints of the optimizer and dataset are saved.
        - "seed_count":             the number of sizes (from the start of the "sizes" list) that are seeds.
        - "stop":                   the config for each size to stop smaller sizes from being trained on after a certain condition is satisfied.
        - "sample_render_period":   the number of steps before the generated level batch is rendered and sent to tensorboard.
        - "heatmap_config":         the heatmap configuration.
        - "heatmap_render_period":  the number of steps before a new heatmap set is rendered and sent to tensorboard.
        - "save_path":              the path to which all the checkpoints and tensorboard logs are saved. It could contain:
                                        - "%TIME": It will be replaced by a timestamp consisting of the current date and time with the precision of seconds.
                                        - "%NAME": It will be replaced by the experiment name.
        - "name_suffix":            a suffix to add to the experiment name.
        """
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
        seed_count: int = 1
        stop: Dict[Tuple[int, int], Dict] = field(default_factory=lambda:{})
        sample_render_period: int = 10
        heatmap_config: Optional[Heatmaps.Config] = None
        heatmap_render_period: int = 100
        save_path: Optional[str] = None
        name_suffix: Optional[str] = None

        def create_trainer(self):
            """Create a training from this config.
            """
            return Trainer(self)
    
    def __init__(self, config: Trainer.Config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.game = config.game_config.create()
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
        """Resume a training process.
        """
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
        """Start training from the current state (from scratch or as a resumption if it is called from 'resume').
        """
        batch_size = self.config.batch_size

        self.writer = tensorboard.writer.SummaryWriter(log_dir=self.log_path, purge_step=self.starting_step)
        # print the tensorboard command to track the experiment.
        print(f"\033[1mTensorboard Command: \033[4m\033[92mtensorboard --logdir {self.log_path}\033[0m")

        self.optG.train()

        start_time = time.time() - self.elapsed_time # We subtract the elapsed time to handle training resumptions.

        stopped = [False]*len(self.dataset.sizes) # Store whether each size has been stopped from training or not
        stop_functions = [(lambda *_: None)]*len(self.dataset.sizes)    # Stores a function for each size that checks for the
                                                                        # stopping condition, then stops the targeted sizes. 
        for size_index, size in enumerate(self.dataset.sizes):
            stop_config = self.config.stop.get(size)
            if stop_config is None: continue
            
            condition_str = stop_config.get("condition", "len(trainer.dataset.items[size]) != 0") 
            condition = eval(f"lambda trainer, step, size: {condition_str}")
            target_count = stop_config.get("target_count", size_index) # by default, the size stops all the sizes below it.
            
            def stop_function(step: int):                
                if condition(self, step, size):
                    stop_functions[size_index] = lambda *_: None
                    for index in range(target_count):
                        stopped[index] = True
            
            stop_functions[size_index] = stop_function

        pbar = tqdm.tqdm(total=self.config.training_steps, initial=self.starting_step-1, desc="Start..", dynamic_ncols=True)
        for step in range(self.starting_step, self.config.training_steps+1):

            for size_index, size in enumerate(self.dataset.sizes):

                if stopped[size_index]: continue

                pbar.set_description(f"{size}: Query Conditions...")
                query_conditions = self.condition_model.sample(size, batch_size)
                
                pbar.set_description(f"{size}: Ask For Levels.....")
                levels = self.optG.ask(query_conditions.to(self.device), size)

                pbar.set_description(f"{size}: Update Dataset.....")
                info, added_mask, log_rewards, stats  = self.dataset.analyze_and_update(size, levels.tolist(), query_conditions)

                if step%self.config.sample_render_period == 0:
                    padding_colors = np.array([([0,127,0] if level_info["solvable"] else [127,0,0]) for level_info in info], dtype=np.uint8)
                    self.writer.add_images(f"Sample/Levels_{size}", self.game.render(levels.cpu().numpy(), padding_color=padding_colors), step)

                if self.heatmap is not None:
                    self.heatmap.update(size, info)
                
                pbar.set_description(f"{size}: Update Cond-Model..")
                self.condition_model.update(size, [item for item, is_added in zip(info, added_mask) if is_added])

                stop_functions[size_index](step)
                
                losses, log_z0s = {}, {}

                # Other than the seed size (the first size), all the other sizes are not trained until a certain
                # number of cluster for them already exists.
                if size_index < self.config.seed_count or self.dataset.distributions[size].leaf_count >= self.dataset.config.cluster_threshold.get(size, 1):
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
            self.writer.add_scalars(f"Dataset/Clusters", {str(size):distribution.leaf_count for size, distribution in self.dataset.distributions.items()}, step)

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