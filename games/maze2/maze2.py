from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .utils import *

class Maze2(Game):
    def __init__(self, **kwargs) -> None:
        super().__init__("MAZE2", '.wgA', "games/maze2/sprites.png")
    
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
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], *, mode: str = "basic") -> List[Level]:
        if mode == "basic":
            return super().generate_random(level_count, size)
        elif mode == "naive":
            h, w = size
            area = h * w
            enemies = enemies or max(w, h)
            PLAYER_PROP = DOOR_PROP = 1/area
            WALL_PROP = EMPTY_PROP = (1 - PLAYER_PROP - DOOR_PROP)/2
            all_tiles_weights = [EMPTY_PROP, WALL_PROP, DOOR_PROP, PLAYER_PROP]  
            all_tiles = list(range(len(self.tiles)))
            return [[random.choices(all_tiles, all_tiles_weights, k=w) for _ in range(h)] for _ in range(level_count)]
        elif mode == "compilable":
            levels = []
            for _ in range(level_count):
                locations = {(j,i) for j in range(h) for i in range(w)}
                player_location, door_location = random.sample(list(locations), 2)
                locations -= {player_location, door_location}
                wall_locations = set(random.sample(list(locations), random.randint(0, len(locations))))
                def tile(location):
                    if location in wall_locations: return 1
                    if location == player_location: return 3
                    if location == door_location: return 2
                    return 0
                levels.append([[tile((j,i)) for i in range(w)] for j in range(h)])
            return levels
    
    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for level in levels:
            h, w = len(level), len(level[0])
            area = h * w
            result = {"level": level}
            
            counter = Counter(tile for row in level for tile in row)
            result["walls"] = walls = counter[1]
            result["wall-ratio"] = walls / area
            result["doors"] = doors = counter[2]
            result["players"] = players = counter[3]
            
            center_j, center_i = (h-1)/2, (w-1)/2
            door_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==2]
            result["door-L1"] = door_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in door_positions)/len(door_positions) if door_positions else 0
            result["door-L1-norm"] = door_L1 / (center_i + center_j)
            player_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==3]

            compilable = players == 1 and doors == 1
            result["compilable"] = compilable
            if compilable:
                obstacles = [[tile == 1 for tile in row] for row in level]
                if is_fully_connected(obstacles):
                    solution, _ = solve(obstacles, player_positions[0], door_positions[0])
                    _, explored = solve(obstacles, player_positions[0], door_positions[0], h_factor=1, g_factor=0)
                else:
                    solution, explored = None, -1
                result["solution"] = solution
                result["explored"] = explored
                result["explored-ratio"] = explored / area
                result["solvable"] = solvable = solution != None
                if solvable:
                    result["solution-length"] = len(solution)
                    result["solution-length-norm"] = len(solution) / area
                    result["difficulty"] = 1 - len(solution) / explored
                else:
                    result["solution-length"] = -1
                    result["solution-length-norm"] = -1
                    result["difficulty"] = -1
            else:
                result["solvable"] = False
                result["solution"] = None
                result["explored"] = -1
                result["explored-ratio"] = -1
                result["solution-length"] = -1
                result["solution-length-norm"] = -1
                result["difficulty"] = -1

            results.append(result)
        return results
    
    @property
    def condition_utility(self) -> ConditionUtility:
        return Maze2ConditionUtility()

class Maze2ConditionUtility(ConditionUtility):
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        
        if prop_name == "solution-length-norm":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), None), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "difficulty":
            def snap(x: float, size: Tuple[int, int]):
                return self.clamp(x, self.const(0), self.const(1))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "wall-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(0), self.const(area-4)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        return (lambda x, _: x) if size is None else (lambda x: x)
    
    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        h, w = size
        
        if prop_name == "solution-length-norm":
            return 0.5/(h*w)

        if prop_name == "difficulty":
            return 0.5/(h*w)
        
        if prop_name == "wall-ratio":
            return 0.5/(h*w)
        
        return super().get_tolerence(prop_name, size)
    
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "solution-length-norm":
            return 1/(h*w), 1.0

        if prop_name == "difficulty":
            return 0.0, 1.0
        
        if prop_name == "wall-ratio":
            area = h*w
            return 0.0, (area-2)/area
        
        return super().get_range_estimates(prop_name, size)