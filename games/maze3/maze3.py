from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .utils import *

"""
The Maze3 game implementation.

Tileset:
--------
.   empty
W   wall
"""

class Maze3(Game):
    def __init__(self, **kwargs) -> None:
        super().__init__("MAZE3", '.w', "games/maze3/sprites.png")
    
    @property
    def possible_augmentation_count(self) -> int:
        return 2 # double reflection
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        if augnmentation_bits is None: augnmentation_bits = random.randint(0, 1<<1 - 1)
        if augnmentation_bits & 1:
            level = level[::-1]
            level = [row[::-1] for row in level]
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], **kwargs) -> List[Level]:
        return super().generate_random(level_count, size) # Randomly selects tiles and every tile has an equal probability
    
    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for level in levels:
            h, w = len(level), len(level[0])
            area = h * w
            result = {"level": level}
            
            walls = sum(cell for row in level for cell in row)
            result["walls"] = walls
            result["wall-ratio"] = walls / area
            
            compilable = walls < area
            result["compilable"] = compilable
            if compilable:
                obstacles = [[tile == 1 for tile in row] for row in level]

                path_length = compute_shortest_path_length(obstacles)
                result["solvable"] = solvable = (path_length is not None)
                if solvable:
                    result["path-length"] = path_length
                    result["path-length-norm"] = path_length / area
                else:
                    result["path-length"] = -1
                    result["path-length-norm"] = -1
            else:
                result["solvable"] = False

            results.append(result)
        return results
    
    @property
    def condition_utility(self) -> ConditionUtility:
        return MazeConditionUtility()

class MazeConditionUtility(ConditionUtility):
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        
        if prop_name == "path-length-norm":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), None), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "wall-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(0), self.const(area-1)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        return (lambda x, _: x) if size is None else (lambda x: x)
    
    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        h, w = size
        
        if prop_name == "path-length-norm":
            return 0.5/(h*w)
        
        if prop_name == "wall-ratio":
            return 0.5/(h*w)

        return super().get_tolerence(prop_name, size)
    
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "path-length-norm":
            return 1/(h*w), 0.5
        
        if prop_name == "wall-ratio":
            area = h*w
            return 0.0, (area-1)/area
        
        return super().get_range_estimates(prop_name, size)