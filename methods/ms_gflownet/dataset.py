from __future__ import annotations
from dataclasses import dataclass, field
import math, random, json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from common.config_tools import config
from methods.diversity import NodeConfig, DiverseDistribution

from games import Game
from games.game import Level

@dataclass
class DatasetItem:
    """An item to be stored in the dataset.

    It contains:
    - "variants":               all the augmentations of the level.
    - "conditions":             the level properties corresponding to each condition (control).
    - "property_log_reward":    the log reward of the level's property reward.
    - "info":                   all the other info about the level generated by the game's analyze function.
    """
    variants: torch.Tensor
    conditions: torch.Tensor
    property_log_reward: float
    info: dict

class Dataset:
    """The dataset class is used to store levels and their information.
    It is used for multiple purposes:
    - Act as an Experience Replay Buffer that can be updated regularly and sampled for training.
    - Analyze and compute rewards for newly generated levels.
    - Save to and Load from a file.  
    """
    @config
    @dataclass
    class Config:
        """The dataset configuration.

        It contains:
        - "data_augmentation":      whether data augmentation is turned on or not.
        - "diversity_sampling":     whether diversity sampling is turned on or not.
        - "diversity_reward":       whether diversity reward is used or not.
        - "property_reward":        the level's property reward (log-space) as a function of the level 'item' and it's size 'size'.
                                    If none, it will be 0.
        - "cluster_key":            the level's cluster key as a function of the level 'item' and it's size 'size'. If none, it will be 0.
        - "cluster_threshold":      a threshold on the number of clusters for each size. If this threshold is not satisfied, no levels will
                                    be sampled for experience replay from this size. This is a safety mechanism to ensure that the generator's
                                    range does not collapse to a single cluster during training. It is optional and turning it off did not
                                    seem to cause any problems. On the contrary, setting it to a high value causes delays in the training
                                    process. We still use it in the config files just to be safe but never set it to more than 2. The default
                                    value will use the value 1 (which is the minimum) for every size.
        """
        data_augmentation: bool = False
        diversity_sampling: bool = False
        diversity_reward: bool = False
        property_reward: Optional[str] = None
        cluster_key: Union[str, List[NodeConfig], None] = None
        cluster_threshold: Dict[Tuple[int, int], int] = field(default_factory=lambda:{})

    items: Dict[Tuple[int, int], List[DatasetItem]]
    distributions: Dict[Tuple[int, int], DiverseDistribution]

    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: Dataset.Config) -> None:
        """The dataset initializer

        Parameters
        ----------
        game : Game
            The game for which the levels are generated. It is used for level analysis.
        conditions : List[str]
            The names of the level properties used to control the generator.
        sizes : List[Tuple[int, int]]
            The sizes used during training.
        config : Dataset.Config
            The dataset configuration.
        """
        self.config = config

        name = ""
        if self.config.diversity_sampling: name += "DIVS_"
        if self.config.diversity_reward: name += "DIVR_"
        if self.config.property_reward is not None: name += "PREW_"
        if self.config.data_augmentation: name += "AUG_" 
        self.name = name + "DATASET" # A name for the dataset used to create the experiment path.

        self.game = game
        self.conditions = conditions
        self.sizes = sizes

        self.condition_utility = game.condition_utility
        self.condition_utility.for_torch()

        self.tolerence = {
            size: torch.tensor([self.condition_utility.get_tolerence(name, size) for name in conditions])
            for size in sizes
        }
        
        self.items = {size:[] for size in sizes}    # The dataset for each size is a list since the order is important.
                                                    # We store the item index only in the clusters.
        self.seen = {size:set() for size in sizes}  # A set for each size to quickly test for levels that are already in the dataset.
        
        self.property_reward = (lambda *_: 0) if self.config.property_reward is None else eval(f"lambda item, size: {self.config.property_reward}")

        self.distributions = {size:DiverseDistribution(self.config.cluster_key, {'size': size}) for size in sizes}
        

    def analyze_and_update(self, size: Tuple[int, int], new_levels: List[Level], query_conditions: Optional[torch.Tensor] = None):
        """Analyzes a list of levels, updates the dataset and returns the level rewards.

        Parameters
        ----------
        size : Tuple[int, int]
            The levels' size.
        new_levels : List[Level]
            A list of the new levels.
        query_conditions : Optional[torch.Tensor], optional
            The conditions used to generate the given levels. It can be None, if the levels were not created
            by a generator. (Default: None)

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[bool], torch.Tensor, Dict[str, Any]]
            The result of the level analysis, a boolean list marking which levels were added to the dataset,
            the level log rewards and some statistics about the dataset update process. 
        """
        new_info = self.game.analyze(new_levels)
        added, log_rewards, stats = self.update(size, new_info, query_conditions)
        return new_info, added, log_rewards, stats

    def update(self, size: Tuple[int, int], new_info: List[dict], query_conditions: Optional[torch.Tensor] = None):
        """Updates the dataset and returns the level rewards.

        Parameters
        ----------
        size : Tuple[int, int]
            The levels' size.
        new_info : List[dict]
            A list of the information about the new levels.
        query_conditions : Optional[torch.Tensor], optional
            The conditions used to generate the given levels. It can be None, if the levels were not created
            by a generator. (Default: None)

        Returns
        -------
        Tuple[List[bool], torch.Tensor, Dict[str, Any]]
            A boolean list marking which levels were added to the dataset, the level log rewards and some 
            statistics about the dataset update process. 
        """

        log_rewards = torch.empty((len(new_info),), dtype=torch.float)

        items = self.items[size]
        seen = self.seen[size]
        distribution = self.distributions[size]
        reward_offset = -distribution.minimum_log_prop
        tolerance = self.tolerence[size]

        # used to store some statistics about the dataset update process.
        compilable_count, solvable_count, new_count = 0, 0, 0
        # The reward for levels that do not satisfy the conditions (or are just unplayable).
        bad_level_log_reward = - size[0] * size[1] * math.log(len(self.game.tiles))
        # A boolean list to mark which levels are added to the dataset
        added = [False]*len(new_info)

        for index, info in enumerate(new_info):
            if not info.get("compilable", True):
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
            
            diversity_reward = reward_offset + distribution.log_prop(info)
            conditions = torch.tensor([info[name] for name in self.conditions])
            property_log_reward = self.property_reward(info, size)
            
            log_reward = property_log_reward
            if query_conditions is not None and torch.any(torch.abs(conditions - query_conditions[index]) > tolerance):
                log_reward += bad_level_log_reward
            if self.config.diversity_reward:
                log_reward += diversity_reward
            log_rewards[index] = log_reward
            
            # Check if the new level is already in the dataset 
            tile_keys = [tuple(tuple(row) for row in level) for level in variants]
            if any(tile_key in seen for tile_key in tile_keys): continue
            for tile_key in tile_keys: seen.add(tile_key)
            
            # Add level to the diverse distribution
            distribution.add(info, len(items))
            
            item = DatasetItem(
                torch.tensor(variants),
                conditions,
                property_log_reward,
                info
            )
            items.append(item)
            
            new_count += 1
            added[index] = True
        
        return added, log_rewards, {
            "compilable": compilable_count,
            "solvable": solvable_count,
            "new": new_count
        }
    
    def save_checkpoint(self, checkpoint_prefix: str):
        """Save a checkpoint of the dataset to a file.

        Parameters
        ----------
        checkpoint_prefix : str
            The path to the checkpoint to save.
        """
        for size in self.sizes:
            h, w = size
            path = f"{checkpoint_prefix}_{h}x{w}.json"
            items = self.items[size]
            info = [item.info for item in items]
            json.dump(info, open(path, 'w'))
    
    def load_checkpoint(self, checkpoint_prefix: str):
        """Load a checkpoint of the dataset from a file.

        Parameters
        ----------
        checkpoint_prefix : str
            The checkpoint path.

        Returns
        -------
        Dict[Tuple[int, int], List[Dict[str, Any]]]
            All the loaded level information organized by size.
        """
        all_info = {}
        for size in self.sizes:
            h, w = size
            path = f"{checkpoint_prefix}_{h}x{w}.json"
            info = json.load(open(path, 'r'))
            self.update(size, info)
            all_info[size] = info
        return all_info

    def sample(self, size: Tuple[int, int], batch_size: int):
        """Sample a batch of levels of a given size.

        Parameters
        ----------
        size : Tuple[int, int]
            The requested level size.
        batch_size : int
            The number of levels to sample.

        Returns
        -------
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            The levels and their corresponding conditions and log rewards.
            If there is not enough levels to sample a batch at the given size (see 'Dataset.Config.cluster_threshold'),
            this function will return None.
        """
        items = self.items[size]
        if len(items) == 0: return None
        
        distribution = self.distributions[size]
        if distribution.leaf_count < self.config.cluster_threshold.get(size, 0): return None
        
        reward_offset = -distribution.minimum_log_prop

        levels = torch.zeros((batch_size, *size), dtype=torch.int64)
        conditions = torch.empty((batch_size, len(self.conditions)), dtype=torch.float)
        log_rewards = torch.empty((batch_size,), dtype=torch.float)

        for index in range(batch_size):
            log_reward = 0
            if self.config.diversity_sampling:
                item_idx, log_prop = distribution.sample()
                item = items[item_idx]
                if self.config.diversity_reward:
                    log_reward = reward_offset + log_prop
            else:
                item = random.choice(items)
                if self.config.diversity_reward:
                    log_reward = reward_offset + distribution.log_prop(item.info)
            
            levels[index] = random.choice(item.variants)
            conditions[index] = item.conditions

            log_reward += item.property_log_reward
            log_rewards[index] = log_reward
        
        return levels, conditions, log_rewards