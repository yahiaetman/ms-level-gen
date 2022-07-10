from __future__ import annotations
from dataclasses import dataclass, field
import math, random, json
from typing import Any, Dict, List, Optional, Tuple

import torch
from common.config_tools import config

from games import Game
from games.game import Level

@dataclass
class DatasetItem:
    variants: torch.Tensor
    conditions: torch.Tensor
    property_log_reward: float
    cluster_key: Any
    info: dict

class Dataset:
    @config
    @dataclass
    class Config:
        data_augmentation: bool = False
        diversity_sampling: bool = False
        diversity_reward: bool = False
        property_reward: Optional[str] = None
        cluster_key: Optional[str] = None
        cluster_threshold: Dict[Tuple[int, int]] = field(default_factory=lambda:{})

    items: Dict[Tuple[int, int], List[DatasetItem]]
    clusters: Dict[Tuple[int, int], Dict[Any, List[int]]]

    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: Dataset.Config) -> None:
        self.config = config

        name = ""
        if self.config.diversity_sampling: name += "DIVS_"
        if self.config.diversity_reward: name += "DIVR_"
        if self.config.diversity_reward: name += "PREW_"
        if self.config.data_augmentation: name += "AUG_" 
        self.name = name + "DATASET"

        self.game = game
        self.conditions = conditions
        self.sizes = sizes

        self.condition_utility = game.condition_utility
        
        self.condition_utility.round = torch.round
        self.condition_utility.clamp = torch.clamp
        self.condition_utility.min = torch.minimum
        self.condition_utility.max = torch.maximum
        self.condition_utility.const = torch.tensor

        self.tolerence = {
            size: torch.tensor([self.condition_utility.get_tolerence(name, size) for name in conditions])
            for size in sizes
        }
        
        self.items = {size:[] for size in sizes}
        self.seen = {size:set() for size in sizes}
        
        self.property_reward = (lambda *_: 0) if self.config.property_reward is None else eval(f"lambda item, size: {self.config.property_reward}")
        self.cluster_key = (lambda *_: 0) if self.config.cluster_key is None else eval(f"lambda item, size: {self.config.cluster_key}")
        self.clusters = {size:{} for size in sizes}
        self.cluster_rewards = {size:{} for size in sizes}
        self.pioneer_rewards = {size:0 for size in sizes}
    
    def __update_cluster_rewards(self, size: Tuple[int, int]):
        clusters = self.clusters[size]
        cluster_log_sizes = {key:math.log(len(cluster)) for key, cluster in clusters.items()}
        if cluster_log_sizes:
            max_cluster_log_size = max(cluster_log_sizes.values())
            self.cluster_rewards[size] = {key: max_cluster_log_size - cluster_log_size for key, cluster_log_size in cluster_log_sizes.items()}
            self.pioneer_rewards[size] = max_cluster_log_size

    def anaalyze_and_update(self, size: Tuple[int, int], new_levels: List[Level], query_conditions: Optional[torch.Tensor] = None):
        new_info = self.game.analyze(new_levels)
        added, log_rewards, stats = self.update(size, new_info, query_conditions)
        return new_info, added, log_rewards, stats

    def update(self, size: Tuple[int, int], new_info: List[dict], query_conditions: Optional[torch.Tensor] = None):
        log_rewards = torch.empty((len(new_info),), dtype=torch.float)

        items = self.items[size]
        seen = self.seen[size]
        clusters = self.clusters[size]
        cluster_rewards = self.cluster_rewards[size]
        pioneer_reward = self.pioneer_rewards[size]
        tolerance = self.tolerence[size]
        compilable_count, solvable_count, new_count = 0, 0, 0
        bad_level_log_reward = - size[0] * size[1] * math.log(len(self.game.tiles))
        added = [False]*len(new_info)
        for index, info in enumerate(new_info):
            if not info["compilable"]:
                log_rewards[index] = bad_level_log_reward
                continue
            compilable_count += 1
            if not info["solvable"]:
                log_rewards[index] = bad_level_log_reward
                continue
            solvable_count += 1

            level = info["level"]
            if self.config.data_augmentation:
                variants = [self.game.augment_level(level, aug_index) for aug_index in range(self.game.possible_augmentation_count)]
            else:
                variants = [level]
            
            cluster_key = self.cluster_key(info, size)
            conditions = torch.tensor([info[name] for name in self.conditions])
            property_log_reward = self.property_reward(info, size)
            
            log_reward = property_log_reward
            if query_conditions is not None and torch.any(torch.abs(conditions - query_conditions[index]) > tolerance):
                log_reward += bad_level_log_reward
            if self.config.diversity_reward:
                log_reward += cluster_rewards.get(cluster_key, pioneer_reward)
            log_rewards[index] = log_reward
            
            tile_keys = [tuple(tuple(row) for row in level) for level in variants]
            if any(tile_key in seen for tile_key in tile_keys): continue
            for tile_key in tile_keys: seen.add(tile_key)
            
            if cluster_key in clusters:
                clusters[cluster_key].append(len(items))
            else:
                clusters[cluster_key] = [len(items)]
            
            item = DatasetItem(
                torch.tensor(variants),
                conditions,
                property_log_reward,
                cluster_key,
                info
            )
            items.append(item)
            
            new_count += 1
            added[index] = True
        
        self.__update_cluster_rewards(size)

        return added, log_rewards, {
            "compilable": compilable_count,
            "solvable": solvable_count,
            "new": new_count
        }
    
    def save_checkpoint(self, checkpoint_prefix: str):
        for size in self.sizes:
            h, w = size
            path = f"{checkpoint_prefix}_{h}x{w}.json"
            items = self.items[size]
            info = [item.info for item in items]
            json.dump(info, open(path, 'w'))
    
    def load_checkpoint(self, checkpoint_prefix: str):
        all_info = {}
        for size in self.sizes:
            h, w = size
            path = f"{checkpoint_prefix}_{h}x{w}.json"
            info = json.load(open(path, 'r'))
            self.update(size, info)
            all_info[size] = info
        return all_info

    def sample(self, size: Tuple[int, int], batch_size: int):
        items = self.items[size]
        if len(items) == 0: return None
        clusters = list(self.clusters[size].values())
        if len(clusters) < self.config.cluster_threshold.get(size, 0): return None
        cluster_rewards = self.cluster_rewards[size]

        levels = torch.zeros((batch_size, *size), dtype=torch.int64)
        conditions = torch.empty((batch_size, len(self.conditions)), dtype=torch.float)
        log_rewards = torch.empty((batch_size,), dtype=torch.float)
        for index in range(batch_size):
            if self.config.diversity_sampling:
                cluster = random.choice(clusters)
                item = items[random.choice(cluster)] 
            else:
                item = random.choice(item)
            levels[index] = random.choice(item.variants)
            conditions[index] = item.conditions
            log_reward = item.property_log_reward
            if self.config.diversity_reward:
                log_reward += cluster_rewards[item.cluster_key]
            log_rewards[index] = log_reward
        return levels, conditions, log_rewards