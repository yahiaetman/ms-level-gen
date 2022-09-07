from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
import random, math # Do not remove the math library, it is needed to eval functions from the config.
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

import torch
from sklearn.mixture import BayesianGaussianMixture

from common.config_tools import config
from games.game import Game
from .utils import find_closest_size, find_closest_smaller_size

class ConditionModel:
    """The base class for all the condition model.
    """

    def __init__(self, game: Game, conditions: List[str], config: Any) -> None:
        """The condition model initializer.

        Parameters
        ----------
        game : Game
            The game for which the condition model is created.
        conditions : List[str]
            The names of the conditions.
        config : Any
            The condition model configuration.
        """
        self.config = config
        self.name = ""
        self.game = game
        self.conditions = conditions

        self.condition_utility = game.condition_utility
        self.condition_utility.for_torch()

        self.snap_functions = [self.condition_utility.get_snapping_function(name) for name in self.conditions]
    
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        """Update the condition model using the new level info.

        Parameters
        ----------
        size : Tuple[int, int]
            The new levels' size.
        new_info : List[dict]
            The new level info.
        """
        raise NotImplementedError()
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        """Sample controls (conditions) unconditionally from the model.

        Parameters
        ----------
        size : Tuple[int, int]
            The level size for which the controls are sampled.
        batch_size : int
            The number of requested samples.

        Returns
        -------
        torch.Tensor
            The sampled controls (conditions).
        """
        raise NotImplementedError()

    def save(self, path: str):
        """Save the condition model to a file.

        Parameters
        ----------
        path : str
            The save file path.
        """
        raise NotImplementedError()
    
    def load(self, path: str):
        """Load the condition model from a file.

        Parameters
        ----------
        path : str
            The model's file path.
        """
        raise NotImplementedError()

############################
############################

# This class is not meant to be saved and loaded
class KDEConditionModel(ConditionModel):
    """This condition model stores a list of levels per size, then samples conditions from them.
    To boost diversity, some noise is added to the sampled conditions. In addition, you can
    enable diversity sampling. 
    """

    @config
    @dataclass
    class Config:
        """The KDE Condition Model configuration.
        The 'save' and 'load' functions are not implemented for this class as it is only meant to be used during training.

        It contains:
        - "noise_factors": the bounds of the uniform distribution used to sample noise for each condition.
        - "diversity_sampling":     whether diversity sampling is turned on or not.
        - "cluster_key":            the level's cluster key as a function of the level 'item' and it's size 'size'. If none, it will be 0.
        - "cluster_threshold":      a threshold on the number of clusters for each size. If this threshold is not satisfied, no levels will
                                    be sampled for experience replay from this size. This is a safety mechanism to ensure that the generator's
                                    range does not collapse to a single cluster during training. It is optional and turning it off did not
                                    seem to cause any problems. On the contrary, setting it to a high value causes delays in the training
                                    process. We still use it in the config files just to be safe but never set it to more than 2. The default
                                    value will use the value 1 (which is the minimum) for every size.
        """
        noise_factors: Dict[str: Tuple[float, float]] = field(default_factory=lambda:{}, metadata={'merge': False})
        diversity_sampling: bool = False
        cluster_key: Optional[str] = None
        cluster_threshold: Dict[Tuple[int, int]] = field(default_factory=lambda:{}, metadata={'merge': False})

        @property
        def model_constructor(self):
            """Return a constructor for the condition model
            """
            return lambda game, conditions: KDEConditionModel(game, conditions, self)
    
    def __init__(self, game: Game, conditions: List[str], config: KDEConditionModel.Config) -> None:
        """The condition model initializer.

        Parameters
        ----------
        game : Game
            The game used to get the condition utilities.
        conditions : List[str]
            The names of the properties corresponding to the conditions.
        config : KDEConditionModel.Config
            The condition model configuration.
        """
        super().__init__(game, conditions, config)
        self.config = config
        
        
        name = ""
        if self.config.diversity_sampling: name += "DIV_"
        self.name = name + "KDECOND" # The name is used to create an experiment name.

        self.noise = {}
        self.range_estimates = {}
        
        self.cluster_key = (lambda *_: 0) if self.config.cluster_key is None else eval(f"lambda item, size: {self.config.cluster_key}")
        
        self.items = defaultdict(list)
        self.clusters = defaultdict(dict)
        self.cluster_thresholds = {}
    
    # get the cluster threshold for a given size. If it is not found, the thresold for the closest size is returned.
    def __compute_cluster_threshold(self, size):
        if len(self.config.cluster_threshold) == 0:
            return 0
        if size in self.config.cluster_threshold:
            return self.config.cluster_threshold[size]
        else:
            closest_size = find_closest_smaller_size(size, self.config.cluster_threshold.keys())
            if closest_size is None:
                return 0
            else:
                return self.config.cluster_threshold[closest_size]

    def update(self, size: Tuple[int, int], new_info: List[dict]):
        """Update the condition model using new level info.

        Parameters
        ----------
        size : Tuple[int, int]
            The levels' size.
        new_info : List[dict]
            The info retrieved by game.analyze for the new levels.
        """
        if len(new_info) == 0: return
        items = self.items[size]
        clusters = self.clusters[size]

        if size not in self.cluster_thresholds:
            self.cluster_thresholds[size] = self.__compute_cluster_threshold(size)
        if size not in self.noise:
            delta = torch.tensor([2 * self.condition_utility.get_tolerence(name, size) for name in self.conditions])
            noise_min_factor, noise_max_factor = zip(*[self.config.noise_factors.get(name, (-1, 1)) for name in self.conditions])
            noise_min_factor = torch.tensor(noise_min_factor, dtype=torch.float)
            noise_max_factor = torch.tensor(noise_max_factor, dtype=torch.float)
            self.noise[size] = (noise_min_factor * delta, noise_max_factor * delta)

        for info in new_info:
            cluster_key = self.cluster_key(info, size)
            conditions = torch.tensor([info[name] for name in self.conditions])
            if cluster_key in clusters:
                clusters[cluster_key].append(len(items))
            else:
                clusters[cluster_key] = [len(items)]
            items.append(conditions)
    

    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        """Sample conditions from the model

        Parameters
        ----------
        size : Tuple[int, int]
            The requested level size to sample conditions for.
        batch_size : int
            The number of conditions to sample.

        Returns
        -------
        torch.Tensor
            A tensor of sampled conditions.
        """
        condition_size = len(self.conditions)
        
        available_sizes = [size for size, cluster in self.clusters.items() if len(cluster) >= self.cluster_thresholds[size]]
        closest_size = find_closest_smaller_size(size, available_sizes)

        if closest_size is None:
            bounds = self.range_estimates.get(size)
            if bounds is None:
                bounds = tuple(
                    torch.tensor(bound) 
                    for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in self.conditions))
                )
                self.range_estimates[size] = bounds
            conditions = torch.lerp(*bounds, torch.rand((batch_size, condition_size)))
        else:
            items = self.items[closest_size]
            clusters = list(self.clusters[closest_size].values())
            noise = self.noise[closest_size]
            conditions = torch.lerp(*noise, torch.rand((batch_size, condition_size)))
            for index in range(batch_size):
                if self.config.diversity_sampling:
                    cluster = random.choice(clusters)
                    item = items[random.choice(cluster)] 
                else:
                    item = random.choice(items)
                conditions[index] += item
        
        for index, snap_function in enumerate(self.snap_functions):
            conditions[:,index] = snap_function(conditions[:,index], size)
        return conditions

############################
############################

class ControllableConditionModel(ConditionModel):
    """Controllable condition model can be given values for a subset of the conditions
    and asked to sample the rest of the conditions. In other words, it is a conditional condition model.
    """

    def sample_given(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
        """Sample conditions given values for the other conditions.

        Parameters
        ----------
        size : Tuple[int, int]
            The requested level size for which the conditions are sampled.
        batch_size : int
            The number of conditions to sample.
        given : Dict[str, float]
            The given condition values in the form of {condition_name : condition_value}.

        Returns
        -------
        torch.Tensor
            A tensor of the sampled conditions.
        """
        raise NotImplementedError()

############################
############################

class UniformRangeConditionModel(ControllableConditionModel):
    """A condition model that samples conditions from a uniform distribution
    whose bounds are dynamically expanded at update.
    """
    @config
    @dataclass
    class Config:
        """The Uniform Range Condition Model configuration.
        It is empty since we do not need any configuration.
        """

        @property
        def model_constructor(self):
            """Return a constructor for the condition model
            """
            return lambda game, conditions: UniformRangeConditionModel(game, conditions, self)

    def __init__(self, game: Game, conditions: List[str], config: UniformRangeConditionModel.Config) -> None:
        """The condition model initializer.

        Parameters
        ----------
        game : Game
            The game used to get the condition utilities.
        conditions : List[str]
            The names of the properties corresponding to the conditions.
        config : UniformRangeConditionModel.Config
            The condition model configuration.
        """
        super().__init__(game, conditions, config)
        self.config = config
        
        self.name = "URCOND" # The name is used to create an experiment name.

        self.range_estimates = {}
        self.ranges = {}
        
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        """Update the condition model using new level info.

        Parameters
        ----------
        size : Tuple[int, int]
            The levels' size.
        new_info : List[dict]
            The info retrieved by game.analyze for the new levels.
        """
        cond_range = self.ranges.get(size)
        
        all_conditions = [torch.tensor([info[name] for name in self.conditions]) for info in new_info]

        if cond_range is None:
            cond_min = reduce(torch.minimum, all_conditions)
            cond_max = reduce(torch.maximum, all_conditions)
        else:
            cond_min = reduce(torch.minimum, all_conditions, cond_range[0])
            cond_max = reduce(torch.maximum, all_conditions, cond_range[1])

        self.ranges[size] = (cond_min, cond_max)
    
    def sample(self, size: Tuple[int, int], batch_size: int) -> torch.Tensor:
        """Sample conditions from the model

        Parameters
        ----------
        size : Tuple[int, int]
            The requested level size to sample conditions for.
        batch_size : int
            The number of conditions to sample.

        Returns
        -------
        torch.Tensor
            A tensor of sampled conditions.
        """
        condition_size = len(self.conditions)
        
        closest_size = find_closest_smaller_size(size, self.ranges.keys())
        
        if closest_size is not None:
            bounds = self.ranges[closest_size]
        else:
            bounds = self.range_estimates.get(size)
            if bounds is None:
                bounds = tuple(
                    torch.tensor(bound) 
                    for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in conditions))
                )
                self.range_estimates[size] = bounds
        
        conditions = torch.lerp(*bounds, torch.rand((batch_size, condition_size)))
        
        for index, snap_function in enumerate(self.snap_functions):
            conditions[:,index] = snap_function(conditions[:,index], size)
        return conditions
    
    def sample_given(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
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
    """This condition model fits a Bayesian Gaussian Mixture Model (GMM) using the given level data.
    It is not meant to be used during training since it can be updated incrementally. 
    """
    @config
    @dataclass
    class Config:
        """The GMM Condition model configuration.

        It contains:
        - "snap": should the sampled conditions be snapped to nearest valid value?
        - "n_components": the number of gaussians in the mixture.
        - "max_iterations": the number of optimization iterations applied to each initialization before aborting.
        - "n_init": the number of initializations to optimize. The one with the maximum lower bound on the likelihood will be picked.
        - "verbose": the verbosity level of sklearn's fitting process. 
        """
        snap: bool = False
        n_components: int = 16
        max_iterations: int = 100
        n_init: int = 8
        verbose: int = 0

        @property
        def model_constructor(self):
            """Return a constructor for the condition model
            """
            return lambda game, conditions: GMMConditionModel(game, conditions, self)
        
    def __init__(self, game: Game, conditions: List[str], config: GMMConditionModel.Config) -> None:
        """The condition model initializer.

        Parameters
        ----------
        game : Game
            The game used to get the condition utilities.
        conditions : List[str]
            The names of the properties corresponding to the conditions.
        config : GMMConditionModel.Config
            The condition model configuration.
        """
        super().__init__(game, conditions, config)
        self.config = config
        
        self.name = "GMMCOND" # The name is used to create an experiment name.

        self.range_estimates = {}
        self.gmms = {}
        self.cache = defaultdict(dict)
        
    def update(self, size: Tuple[int, int], new_info: List[dict]):
        """Update the condition model using new level info.
        IMPORTANT: This condition model discard all previous results of any previous update
        for the given size since it cannot incrementally update the GMM. So it is not
        designed to be used during training.

        Parameters
        ----------
        size : Tuple[int, int]
            The levels' size.
        new_info : List[dict]
            The info retrieved by game.analyze for the new levels.
        """
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
        """Sample conditions from the model

        Parameters
        ----------
        size : Tuple[int, int]
            The requested level size to sample conditions for.
        batch_size : int
            The number of conditions to sample.

        Returns
        -------
        torch.Tensor
            A tensor of sampled conditions.
        """
        if self.gmms:
            closest_size = find_closest_size(size, self.gmms.keys())
            gmm = self.gmms.get(closest_size)
            if gmm is None:
                h, w = size
                _, _, _, gmm = min((abs(hi-h)+abs(wi-w), (hi+wi), i, gmmi) for i, ((hi, wi), gmmi) in enumerate(self.gmms.items()))
            means = gmm["means"]
            covariances = gmm["covariances"]
            weights = gmm["weights"]
            gmm_dist = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(weights),
                torch.distributions.MultivariateNormal(means, covariances)
            )
            conditions = gmm_dist.sample((batch_size,))
        else:
            bounds = self.range_estimates.get(size)
            if bounds is None:
                bounds = tuple(
                    torch.tensor(bound) 
                    for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in self.conditions))
                )
                self.range_estimates[size] = bounds
            conditions = torch.lerp(*bounds, torch.rand((batch_size, len(self.conditions))))
        
        if self.config.snap:
            for index, snap_function in enumerate(self.snap_functions):
                conditions[:,index] = snap_function(conditions[:,index], size)
        
        return conditions
    
    def sample_given(self, size: Tuple[int, int], batch_size: int, given: Dict[str, float]) -> torch.Tensor:
        condition_size = len(self.conditions)

        given_names, given_values = tuple(zip(*list(sorted(given.items()))))
        given_values = torch.tensor(given_values)
        given_indices = torch.tensor([self.conditions.index(name) for name in given_names], dtype=int) # The indices of the given conditions
            
        if self.gmms:
            closest_size = find_closest_size(size, self.gmms.keys()) # Returns the closest size which has a GMM
            
            gmm = self.gmms[closest_size]
            means = gmm["means"]
            covariances = gmm["covariances"]
            weights = gmm["weights"]
            
            random_indices = torch.tensor([index for index in range(condition_size) if index not in given_indices], dtype=int) # The indics of the conditions that should be sampled
            
            cache = self.cache[size] # To avoid recomputing some matrix calculations, we cache the results
            if given_names in cache:
                (Caa, Cab, Cba, Cbb, Cbbinv, C) = cache[given_names]
            else:
                Caa = covariances[:, random_indices[:,None], random_indices]
                Cab = covariances[:, random_indices[:,None], given_indices]
                Cba = covariances[:, given_indices[:,None], random_indices]
                Cbb = covariances[:, given_indices[:,None], given_indices]
                Cbbinv = torch.linalg.inv(Cbb)
                C = Caa - torch.matmul(Cab, torch.matmul(Cbbinv, Cba)) # The Covariances for the conditions to be sampled.
                cache[given_names] = (Caa, Cab, Cba, Cbb, Cbbinv, C)

            given_x_u = given_values - means[:, given_indices]
            marginal = torch.exp(-0.5 * torch.matmul(given_x_u[:,None,:], torch.matmul(Cbbinv, given_x_u[:,:,None])))[:,0,0] # compute the marginal of each gaussian at the given values
            conditional_weights = weights * marginal # The weights after adding the marginal (before normalizing)
            conditional_weights_sum = torch.sum(conditional_weights)
            if conditional_weights_sum != 0:
                # Normalize the gaussians' weights after marginalizing on the given conditions.
                conditional_weights /= conditional_weights_sum
            else:
                # As a fallback, we use the original weights in this case
                conditional_weights = weights
            
            u = means[:, random_indices] + torch.matmul(torch.matmul(Cab, Cbbinv), given_x_u[:,:,None])[:,:,0] # The Means for the conditions to be sampled.

            gmm_dist = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(conditional_weights),
                torch.distributions.MultivariateNormal(u, C)
            )
            random_conditions = gmm_dist.sample((batch_size,))

            # Combine the given and sampled conditions.
            conditions = torch.empty((batch_size, condition_size))
            conditions[:, given_indices] = given_values
            conditions[:, random_indices] = random_conditions
        else:
            # If no GMMs are found, fallback to using uniform distributions whose bounds are the range estimates.
            # This should never be needed but it is here is for completeness.
            bounds = self.range_estimates.get(size)
            if bounds is None:
                bounds = tuple(
                    torch.tensor(bound) 
                    for bound in zip(*(self.condition_utility.get_range_estimates(name, size) for name in conditions))
                )
                self.range_estimates[size] = bounds
            conditions = torch.lerp(*bounds, torch.rand((batch_size, condition_size)))
            conditions[:, given_indices] = given_values
        
        if self.config.snap:
            for index, snap_function in enumerate(self.snap_functions):
                conditions[:,index] = snap_function(conditions[:,index], size)
        
        return conditions
    
    def save(self, path: str):
        torch.save(self.gmms, path)
    
    def load(self, path: str):
        self.gmms = torch.load(path)
