from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .utils import *

class Maze(Game):
    def __init__(self, **kwargs) -> None:
        super().__init__("MAZE", '.w', "games/maze/sprites.png")
    
    @property
    def possible_augmentation_count(self) -> int:
        return 4
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        if augnmentation_bits is None: augnmentation_bits = random.randint(0, 1<<2 - 1)
        if augnmentation_bits & 1:
            level = level[::-1]
        if augnmentation_bits & 2:
            level = [row[::-1] for row in level]
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], **kwargs) -> List[Level]:
        return super().generate_random(level_count, size)
    
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

                result["solvable"] = solvable = is_fully_connected(obstacles)
                if solvable:
                    max_path_length = compute_longest_shortest_path_length(obstacles)
                    result["maximum-path-length"] = max_path_length
                    result["maximum-path-length-norm"] = max_path_length / area
                else:
                    result["maximum-path-length"] = -1
                    result["maximum-path-length-norm"] = -1
            else:
                result["solvable"] = False

            results.append(result)
        return results
    
    @property
    def condition_utility(self) -> ConditionUtility:
        return MazeConditionUtility()

class MazeConditionUtility(ConditionUtility):
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        
        if prop_name == "maximum-path-length-norm":
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
        
        if prop_name == "maximum-path-length-norm":
            return 0.5/(h*w)
        
        if prop_name == "wall-ratio":
            return 0.5/(h*w)

        return super().get_tolerence(prop_name, size)
    
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "maximum-path-length-norm":
            return 1/(h*w), 0.5
        
        if prop_name == "wall-ratio":
            area = h*w
            return 0.0, (area-1)/area
        
        return super().get_range_estimates(prop_name, size)