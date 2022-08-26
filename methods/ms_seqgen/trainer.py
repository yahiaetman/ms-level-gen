from __future__ import annotations
from dataclasses import dataclass
import os, datetime, pathlib
import time
from typing import Any, List, Optional, Tuple
import warnings

import torch
from torch.utils import tensorboard
import tqdm
import yaml
from common.config_tools import config

from games import GameConfig, create_game
from common.heatmap import Heatmaps
from .dataset import Dataset
from methods.ms_conditions import ConditionModel
from .models import SeqMSGenerator
from .optimizers import SeqMSOptimizer

class Trainer:
    @config
    @dataclass
    class Config:
        game_config: GameConfig
        conditions: List[str]
        sizes: List[Tuple[int, int]]
        dataset_seed_path: str
        dataset_postprocessing: List[str]
        dataset_config: Dataset.Config
        condition_model_config: Any
        generator_config: Any
        optimizer_config: SeqMSOptimizer.Config
        training_steps: int
        batch_size: int
        bootstrapping_period: int
        bootstrapping_batch_size: int
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
        self.netG: SeqMSGenerator = config.generator_config.model_constructor(len(self.game.tiles), len(config.conditions)).to(self.device)
        self.optG = SeqMSOptimizer(self.netG, config.optimizer_config)
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
        
        self.__start_trainning_loop()
        
    def train(self):
        levels, _ = self.game.load_dataset(self.config.dataset_seed_path)
        
        level_groups = {size:[] for size in self.config.sizes}
        for level in levels:
            size = len(level), len(level[0])
            if (group := level_groups.get(size)) is not None:
                group.append(level)
        
        postprocessors = self.game.dataset_postprocessors
        postprocessors = [postprocessors[name] for name in self.config.dataset_postprocessing]

        for size, group in level_groups.items():
            if len(group) == 0: continue
            for postprocessor in postprocessors:
                group = postprocessor(group)
            self.dataset.analyze_and_update(size, group)

        self.__start_trainning_loop()

    def __start_trainning_loop(self):
        batch_size = self.config.batch_size

        self.writer = tensorboard.writer.SummaryWriter(log_dir=self.log_path, purge_step=self.starting_step)
        print(f"\033[1mTensorboard Command: \033[4m\033[92mtensorboard --logdir {self.log_path}\033[0m")

        self.optG.train()

        start_time = time.time() - self.elapsed_time

        pbar = tqdm.tqdm(total=self.config.training_steps, initial=self.starting_step-1, desc="Start..", dynamic_ncols=True)
        for step in range(self.starting_step, self.config.training_steps+1):
            augment_dataset = (step % self.config.bootstrapping_period) == 0
            for size in self.dataset.sizes:

                pbar.set_description(f"{size}: Sample Batch.....")
                batch = self.dataset.sample(size, batch_size)
                if batch is not None:
                    levels, conditions = batch
                    pbar.set_description(f"{size}: Train............")
                    loss = self.optG.step(conditions.to(self.device), levels.to(self.device))
                    self.writer.add_scalar(f"Training/Loss_{size}", loss, step)
                
                if augment_dataset:
                    pbar.set_description(f"{size}: Query Conditions.")
                    query_conditions = self.condition_model.sample(size, batch_size)
                    
                    pbar.set_description(f"{size}: Query Levels.....")
                    levels = self.netG.generate(query_conditions.to(self.device), size)

                    if step%self.config.sample_render_period == 0:
                        self.writer.add_images(f"Sample/Levels_{size}", self.game.render(levels.cpu().numpy()), step)
                    
                    pbar.set_description(f"{size}: Update Dataset...")
                    info, added_mask, stats  = self.dataset.analyze_and_update(size, levels.tolist(), query_conditions)

                    if self.heatmap is not None:
                        self.heatmap.update(size, info)
                
                    pbar.set_description(f"{size}: Update Cond-Model")
                    self.condition_model.update(size, [item for item, is_added in zip(info, added_mask) if is_added])

                    self.writer.add_scalars(f"Generation/Quality_{size}", stats, step)

            if augment_dataset:
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