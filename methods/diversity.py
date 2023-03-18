from typing import Tuple, List, Dict, Any, Callable, Hashable, Union
import random
import math
from common.config_tools import config
from dataclasses import dataclass

ItemInfo = Dict[str, Any]

class Node:
    def sample(self) -> Tuple[int, float]:
        pass

    def add(self, info: ItemInfo, item: int) -> bool:
        return False

    def log_prop(self, info: ItemInfo) -> float:
        return 0
    
    @property
    def minimum_log_prop(self) -> float:
        return 0

    @property
    def size(self) -> int:
        return 0

    @property
    def leaf_count(self) -> int:
        return 1

NodeCreator = Callable[[], Node]

class NodeConfig:
    def creator(self, child_creator: NodeCreator, context: Dict[str, Any]) -> NodeCreator:
        pass

################
# Common Nodes #
################

class LeafNode(Node):
    cached_log_prop: float
    items: List[int]

    def __init__(self) -> None:
        self.cached_log_prop = 0
        self.items = []

    def sample(self) -> Tuple[int, float]:
        return random.choice(self.items), self.cached_log_prop
    
    def add(self, info: ItemInfo, item: int) -> bool:
        self.items.append(item)
        self.cached_log_prop = -math.log(len(self.items))
        return False
    
    def log_prop(self, info: ItemInfo) -> float:
        return self.cached_log_prop
    
    @property
    def minimum_log_prop(self) -> float:
        return self.cached_log_prop

    @property
    def size(self) -> int:
        return len(self.items)

class ClusterNode(Node):
    cached_leaf_count: int
    cached_log_prop: float
    cached_min_child_log_prop: float
    cluster_key_fn: Callable[[ItemInfo], Hashable]
    child_map: Dict[Hashable, Node]
    children: List[Node]
    child_creator: Callable[[], Node]

    @config
    @dataclass
    class Config(NodeConfig):
        key: str
        
        def creator(self, child_creator: NodeCreator, context: Dict[str, Any]) -> NodeCreator:
            key_fn_str = "lambda item"
            for name, value in context.items():
                key_fn_str += f", {name}={value}" 
            key_fn_str += f": {self.key}"
            key_fn = eval(key_fn_str)
            if child_creator is None:
                child_creator = lambda: LeafNode()
            return lambda: ClusterNode(key_fn, child_creator)

    def __init__(self, cluster_key_fn: Callable[[ItemInfo], Hashable], child_creator: Callable[[], Node]) -> None:
        self.cached_leaf_count = 0
        self.cached_log_prop = 0
        self.cached_min_child_log_prop = 0
        self.max_child_size = 0
        self.cluster_key_fn = cluster_key_fn
        self.child_map = {}
        self.children = []
        self.child_creator = child_creator

    def sample(self) -> Tuple[int, float]:
        child = random.choice(self.children)
        item, reward = child.sample()
        return item, self.cached_log_prop + reward
    
    def add(self, info: ItemInfo, item: int) -> bool:
        new_leaf = False
        key = self.cluster_key_fn(info)
        child = self.child_map.get(key)
        if child is None:
            child = self.child_creator()
            self.child_map[key] = child
            self.children.append(child)
            self.cached_log_prop = -math.log(len(self.children))
            new_leaf = True
        if child.add(info, item): new_leaf = True    
        self.cached_min_child_log_prop = min(self.cached_min_child_log_prop, child.minimum_log_prop)
        if new_leaf: self.cached_leaf_count += 1
        return new_leaf
    
    def log_prop(self, info: ItemInfo) -> float:
        log_prop = self.cached_log_prop
        key = self.cluster_key_fn(info)
        child = self.child_map.get(key)
        if child is not None:
            log_prop += child.log_prop(info)
        return log_prop
    
    @property
    def minimum_log_prop(self) -> float:
        return self.cached_log_prop + self.cached_min_child_log_prop

    @property
    def size(self) -> int:
        return len(self.children)
    
    @property
    def leaf_count(self) -> int:
        return self.cached_leaf_count

################
# Distribution #
################

class DiverseDistribution:
    root: Node

    def __init__(self, node_configs: Union[str, List[NodeConfig], None], context: Dict[str, Any]) -> None:
        if node_configs is None:
            self.root = LeafNode()
        else:
            if isinstance(node_configs, str):
                node_configs = [ClusterNode.Config(node_configs)]
            node_creator = None
            for config in node_configs[::-1]:
                node_creator = config.creator(node_creator, context)
            self.root = node_creator()
    
    def sample(self) -> Tuple[int, float]:
        return self.root.sample()
    
    def add(self, info: ItemInfo, item: int):
        self.root.add(info, item)
    
    def log_prop(self, info: ItemInfo) -> float:
        return self.root.log_prop(info)
    
    @property
    def minimum_log_prop(self) -> float:
        return self.root.minimum_log_prop

    @property
    def leaf_count(self) -> int:
        return self.root.leaf_count
    
### TEST CODE

if __name__ == "__main__":
    dataset = [
        {"x": 1, "y": 100},
        {"x": 2, "y": 101},
        {"x": 1, "y": 101},
        {"x": 3, "y": 0},
        {"x": 3, "y": 1},
        {"x": 3, "y": 2},
    ]

    #keys = "(item['x'], item['y']//10)"
    keys = [ClusterNode.Config("item['y']//10"), ClusterNode.Config("item['x']")]

    dist = DiverseDistribution(keys, {})
    for index, item in enumerate(dataset):
        dist.add(item, index)
        print(dist.leaf_count)
    
    for item in dataset:
        print(dist.log_prop(item) - dist.minimum_log_prop)

    for _ in range(10):
        print(dist.sample())