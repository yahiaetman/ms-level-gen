from __future__ import annotations
from dataclasses import dataclass, field
from functools import reduce
import random
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

import torch
from sklearn.mixture import BayesianGaussianMixture

from common.config_tools import config
from games.game import Game

class ConditionModel:
    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: Any) -> None:
        self.config = config
        self.name = ""
        self.game = game
        self.conditions = conditions
        self.sizes = sizes

        self.condition_utility = game.condition_utility
        self.condition_utility.round = torch.round
        self.condition_utility.clamp = torch.clamp
        self.condition_utility.min = torch.minimum
        self.condition_utility.max = torch.maximum
        self.condition_utility.const = torch.tensor

        self.snap_functions = [self.condition_utility.get_snapping_function(name) for name in self.conditions]
    
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        raise NotImplementedError()
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()
    
    def load(self, path: str):
        raise NotImplementedError()

############################
############################

# This class is not meant to be saved and loaded
class KDEConditionModel(ConditionModel):
    @config
    @dataclass
    class Config:
        noise_factors: Dict[str: Tuple[float, float]] = field(default_factory=lambda:{})
        fallback_tree: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=lambda:{})
        diversity_sampling: bool = False
        cluster_key: Optional[str] = None
        cluster_threshold: Dict[Tuple[int, int]] = field(default_factory=lambda:{})

        @property
        def model_constructor(self):
            return lambda game, conditions, sizes: KDEConditionModel(game, conditions, sizes, self)
    
    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: KDEConditionModel.Config) -> None:
        super().__init__(game, conditions, sizes, config)
        self.config = config
        
        
        name = ""
        if self.config.diversity_sampling: name += "DIV_"
        self.name = name + "KDECOND"

        tolerence = {
            size: torch.tensor([self.condition_utility.get_tolerence(name, size) for name in conditions])
            for size in sizes
        }

        noise_min_factor, noise_max_factor = zip(*[self.config.noise_factors.get(name, (-2, 2)) for name in conditions])
        noise_min_factor, noise_max_factor = torch.tensor(noise_min_factor), torch.tensor(noise_max_factor) 
        self.noise_min = {size: noise_min_factor*tolerence for size, tolerence in tolerence.items()}
        self.noise_max = {size: noise_max_factor*tolerence for size, tolerence in tolerence.items()}


        self.range_estimates = {
            size:tuple(torch.tensor(bound) for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in conditions)))
            for size in sizes
        }
        
        self.items = {size:[] for size in sizes}
        self.cluster_key = (lambda *_: 0) if self.config.cluster_key is None else eval(f"lambda item, size: {self.config.cluster_key}")
        self.clusters = {size:{} for size in sizes}
        
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        items = self.items[size]
        clusters = self.clusters[size]

        for info in new_info:
            cluster_key = self.cluster_key(info, size)
            conditions = torch.tensor([info[name] for name in self.conditions])
            if cluster_key in clusters:
                clusters[cluster_key].append(len(items))
            else:
                clusters[cluster_key] = [len(items)]
            items.append(conditions)
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        condition_size = len(self.conditions)
        
        fallback_size = size        
        fallback_tree = self.config.fallback_tree
        while len(self.items[fallback_size]) < self.config.cluster_threshold.get(size, 0):
            fallback_size = fallback_tree.get(fallback_size)
            if fallback_size is None:
                break
        
        if fallback_size is not None:
            items = self.items[fallback_size]
            clusters = list(self.clusters[fallback_size].values())
            conditions = torch.empty((batch_size, condition_size), dtype=torch.float)
            for index in range(batch_size):
                if self.config.diversity_sampling:
                    cluster = random.choice(clusters)
                    item = items[random.choice(cluster)] 
                else:
                    item = random.choice(item)
                conditions[index] = item
        else:
            bound_min, bound_max = self.range_estimates[size]
            conditions = torch.lerp(bound_min, bound_max, torch.rand((batch_size, condition_size)))
        
        for index, snap_function in enumerate(self.snap_functions):
            conditions[:,index] = snap_function(conditions[:,index], size)
        return conditions

############################
############################

class ControllableConditionModel(ConditionModel):
    def SampleGiven(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
        raise NotImplementedError()

############################
############################

class UniformRangeConditionModel(ControllableConditionModel):
    @config
    @dataclass
    class Config:
        fallback_tree: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=lambda:{})

        @property
        def model_constructor(self):
            return lambda game, conditions, sizes: UniformRangeConditionModel(game, conditions, sizes, self)

    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: UniformRangeConditionModel.Config) -> None:
        super().__init__(game, conditions, sizes, config)
        self.config = config
        
        self.name = "URCOND"

        self.condition_utility = game.condition_utility
        self.condition_utility.round = torch.round
        self.condition_utility.clamp = torch.clamp
        self.condition_utility.min = torch.minimum
        self.condition_utility.max = torch.maximum
        self.condition_utility.const = torch.tensor

        self.range_estimates = {
            size:tuple(torch.tensor(bound) for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in conditions)))
            for size in sizes
        }
        
        self.ranges = {size:None for size in sizes}
        
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        cond_range = self.ranges[size]
        
        all_conditions = [torch.tensor([info[name] for name in self.conditions]) for info in new_info]

        if cond_range is None:
            cond_min = reduce(torch.minimum, all_conditions)
            cond_max = reduce(torch.maximum, all_conditions)
        else:
            cond_min = reduce(torch.minimum, all_conditions, cond_range[0])
            cond_max = reduce(torch.maximum, all_conditions, cond_range[1])

        self.ranges[size] = (cond_min, cond_max)
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        condition_size = len(self.conditions)
        
        fallback_size = size        
        fallback_tree = self.config.fallback_tree
        cond_range = self.ranges[fallback_size]
        while cond_range is None or torch.allclose(*cond_range):
            fallback_size = fallback_tree.get(fallback_size)
            if fallback_size is None:
                break
            cond_range = self.ranges[fallback_size]
        
        if fallback_size is not None:
            cond_min, cond_max = cond_range
        else:
            cond_min, cond_max = self.range_estimates[size]
        
        conditions = torch.lerp(cond_min, cond_max, torch.rand((batch_size, condition_size)))
        
        for index, snap_function in enumerate(self.snap_functions):
            conditions[:,index] = snap_function(conditions[:,index], size)
        return conditions
    
    def SampleGiven(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
        conditions = self.sample(size, batch_size)
        for condition_name, value in given.items():
            condition_index = self.conditions.index(condition_name)
            conditions[:, condition_index] = value
        return conditions
    
    def save(self, path: str):
        torch.save({
            size:{
                "minimum": cond_range[0],
                "maximum": cond_range[1]
            } 
            for size, cond_range in self.ranges.items() 
            if cond_range is not None
        }, path)
    
    def load(self, path: str):
        data = torch.load(path)
        for size, cond_range in data.items():
            if size in self.ranges:
                self.ranges[size] = (cond_range["minimum"], cond_range["maximum"])

############################
############################

# This class is not meant to be used during training
class GMMConditionModel(ControllableConditionModel):
    @config
    @dataclass
    class Config:
        snap: bool = False
        n_components: int = 16
        max_iterations: int = 100
        n_init: int = 8
        verbose: int = 0

        @property
        def model_constructor(self):
            return lambda game, conditions, sizes: GMMConditionModel(game, conditions, sizes, self)
        
    def __init__(self, game: Game, conditions: List[str], sizes: List[Tuple[int, int]], config: GMMConditionModel.Config) -> None:
        super().__init__(game, conditions, sizes, config)
        self.config = config
        
        self.name = "GMMCOND"

        self.condition_utility = game.condition_utility
        self.condition_utility.round = torch.round
        self.condition_utility.clamp = torch.clamp
        self.condition_utility.min = torch.minimum
        self.condition_utility.max = torch.maximum
        self.condition_utility.const = torch.tensor

        self.fallback_range_estimate = tuple(torch.tensor(bound) 
            for bound in zip(*(self.condition_utility.get_range_estimates(name, sizes[0]) for name in conditions))
        )
        
        self.gmms = {}
        self.cache = {}
        
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        all_conditions = torch.tensor([[info[name] for name in self.conditions] for info in new_info])
        gmm = BayesianGaussianMixture(
            n_components=self.config.n_components, 
            max_iter=self.config.max_iterations,
            n_init=self.config.n_init,
            verbose=self.config.verbose
        ).fit(all_conditions.numpy())

        if not gmm.converged_:
            warnings.warn("GMM Condition Model: The model did not converge")

        self.gmms[size] = {
            "means": torch.from_numpy(gmm.means_).float(),
            "covariances": torch.from_numpy(gmm.covariances_).float(),
            "weights": torch.from_numpy(gmm.weights_).float()
        }
        if size in self.cache:
            self.cache[size].clear()
        else:
            self.cache[size] = {}
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        if self.gmms:
            gmm = self.gmms.get(size)
            if gmm is None:
                h, w = size
                _, gmm = min((abs(hi-h)+abs(wi-w), gmmi) for (hi, wi), gmmi in self.gmms.items())
            means = gmm["means"]
            covariances = gmm["covariances"]
            weights = gmm["weights"]
            gmm_dist = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(weights),
                torch.distributions.MultivariateNormal(means, covariances)
            )
            conditions = gmm_dist.sample((batch_size,))
        else:
            cond_min, cond_max = self.fallback_range_estimate
            conditions = torch.lerp(cond_min, cond_max, torch.rand((batch_size, len(self.conditions))))
        
        if self.config.snap:
            for index, snap_function in enumerate(self.snap_functions):
                conditions[:,index] = snap_function(conditions[:,index], size)
        
        return conditions
    
    def SampleGiven(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
        condition_size = len(self.conditions)

        if self.gmms:
            if size in self.gmms:
                picked_size = size                
            else:
                h, w = size
                _, picked_size = min((abs(ht-h)+abs(wt-w), (ht, wt)) for ht, wt in self.gmms)
            
            gmm = self.gmms[picked_size]
            means = gmm["means"]
            covariances = gmm["covariances"]
            weights = gmm["weights"]
            
            given_names, given_values = tuple(zip(*list(sorted(given.items()))))
            given_values = torch.tensor(given_values)
            given_indices = torch.tensor([self.conditions.index(name) for name in given_names], dtype=int)
            random_indices = torch.tensor([index for index in range(condition_size) if index not in given_indices], dtype=int)
            
            cache = self.cache[size]
            if given_names in cache:
                (Caa, Cab, Cba, Cbb, Cbbinv, C) = self.cache[(size, given_names)]
            else:
                Caa = covariances[:, random_indices[:,None], random_indices]
                Cab = covariances[:, random_indices[:,None], given_indices]
                Cba = covariances[:, given_indices[:,None], random_indices]
                Cbb = covariances[:, given_indices[:,None], given_indices]
                Cbbinv = torch.linalg.inv(Cbb)
                C = Caa - torch.matmul(Cab, torch.matmul(Cbbinv, Cba))
                cache[given_names] = (Caa, Cab, Cba, Cbb, Cbbinv, C)

            given_x_u = given_values - means[:, given_indices]
            marginal = torch.exp(-0.5 * torch.matmul(given_x_u[:,None,:], torch.matmul(Cbbinv, given_x_u[:,:,None])))[:,0,0]
            conditional_weights = weights * marginal
            conditional_weights_sum = torch.sum(conditional_weights)
            if conditional_weights_sum != 0:
                conditional_weights /= conditional_weights_sum
            u = means[:, random_indices] + torch.matmul(torch.matmul(Cab, Cbbinv), given_x_u[:,:,None])[:,:,0]

            # picked_comp = torch.multinomial(conditional_weights, batch_size, replacement=True)
            # random_conditions = torch.distributions.MultivariateNormal(u[picked_comp], C[picked_comp]).sample()
            gmm_dist = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(conditional_weights),
                torch.distributions.MultivariateNormal(u, C)
            )
            random_conditions = gmm_dist.sample((batch_size,))

            conditions = torch.empty((batch_size, batch_size))
            conditions[:, given_indices] = given_values
            conditions[:, random_indices] = random_conditions
        else:
            cond_min, cond_max = self.fallback_range_estimate
            conditions = torch.lerp(cond_min, cond_max, torch.rand((batch_size, condition_size)))
        
        if self.config.snap:
            for index, snap_function in enumerate(self.snap_functions):
                conditions[:,index] = snap_function(conditions[:,index], size)
        
        return conditions
    
    def save(self, path: str):
        torch.save(self.gmms, path)
    
    def load(self, path: str):
        self.gmms = torch.load(path)

############################
############################

def get_condition_model_by_name(name: str) -> Type[ConditionModel]:
    return {
        "kde": KDEConditionModel,
        "uniformrange": UniformRangeConditionModel,
        "gmm": GMMConditionModel,
    }[name.lower()]