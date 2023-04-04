from typing import Tuple, List, Dict, Any, Callable, Hashable, Union
import random
import math
from common.config_tools import config
from dataclasses import dataclass

ItemInfo = Dict[str, Any]   # A key-value storage type for item properties.
                            # The key is the property name, and the value is the property value.

# The base type for all the diversity tree nodes
class Node:
    # Sample an item id (int) and return its probability
    def sample(self) -> Tuple[int, float]:
        pass

    # Add an item id (int) to the tree based on its properties which are stored in "info"
    def add(self, info: ItemInfo, item: int) -> bool:
        return False

    # Return the log likelihood of sampling an item with the given properties
    def log_prop(self, info: ItemInfo) -> float:
        return 0
    
    # Return the log likelihood of the item that is most unlikely to be sampled
    @property
    def minimum_log_prop(self) -> float:
        return 0

    # Return the number of children for this node (or items if it is leaf node)
    @property
    def size(self) -> int:
        return 0

    # Return the number of leaves that are descendant of this node
    @property
    def leaf_count(self) -> int:
        return 1

NodeCreator = Callable[[], Node] # A type for node constructors (takes no arguments and returns a node)

# A base class for configuration classes of all nodes
class NodeConfig:
    # Creates a node constructor given a constructor for its children and an dict containing any information needed as a context
    def creator(self, child_creator: NodeCreator, context: Dict[str, Any]) -> NodeCreator:
        pass

################
# Common Nodes #
################

# A leaf node that directly stores items (has no child nodes)
class LeafNode(Node):
    cached_log_prop: float  # The log likelihood of each item, cached for optimization
    items: List[int]        # A list of the items that this node contains

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
        return self.cached_log_prop # All items here are equally likely

    @property
    def size(self) -> int:
        return len(self.items)

# Node the clusters items into children nodes based on the item properties
class ClusterNode(Node):
    cached_leaf_count: int  # A caching of the number of leafs that are descendant to this node
    cached_log_prop: float  # A caching of the log likelihood of each child
    cached_min_child_log_prop: float    # A caching of the log likelihood of the item in any descendant that has the least likelihood of being sampled 
    cluster_key_fn: Callable[[ItemInfo], Hashable]  # A function that returns a clustering key from the item properties
    child_map: Dict[Hashable, Node] # A dict that maps every cluster key to a node
    children: List[Node]            # A list of the children (same as value of child_map but store separately to optimize sampling)
    child_creator: Callable[[], Node]   # A function that creates a child node

    @config
    @dataclass
    class Config(NodeConfig):
        key: str # A string defining the body of the "cluster_key_fn" where the param are "item" and keyword params defined by the context passed to "creator" 
                # For example, if key is "info['walls']//sum(size)", and context contains {"size":(7, 7)}, the "cluster_key_fn" will be:
                #   lambda item, size=(7, 7): info['walls']//sum(size)

        def creator(self, child_creator: NodeCreator, context: Dict[str, Any]) -> NodeCreator:
            key_fn_str = "lambda item"
            for name, value in context.items():
                key_fn_str += f", {name}={value}" 
            key_fn_str += f": {self.key}"
            key_fn = eval(key_fn_str)   # Create a lambda expression for cluster_key_fn
            if child_creator is None:   # If child_creator is None, then all the children are leaves
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
        if child is None: # if no child was found for the given item, we create a new one for it
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

# A class that defines a diverse distribution (used for diversity sampling)
class DiverseDistribution:
    root: Node

    # node_configs: A list of node configurations defining the node config at each level of the clustering tree
    #               If node_configs is a string, then it is the key for a single ClusterNode (the tree has a depth of 1)
    #               If node_configs is None, then the root is a LeafNode (the tree has a depth of 0)
    # context: A dict of contextual info passed to the node constructors
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