from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .utils import *

"""
The GVGAI version of Zelda game implementation.

Tileset:
--------
.   empty
W   wall
+   key
g   door
A   player
1   bat
2   spider
3   scorpion
"""

class Zelda(Game):
    def __init__(self, **kwargs) -> None:
        super().__init__("ZELDA", '.w+gA123', "games/zelda/sprites.png")
    
    @property
    def possible_augmentation_count(self) -> int:
        return 4 # 2 for the vertical flipping x 2 for the horizontal flipping
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        if augnmentation_bits is None: augnmentation_bits = random.randint(0, 1<<2 - 1)
        if augnmentation_bits & 1:
            level = level[::-1]
        if augnmentation_bits & 2:
            level = [row[::-1] for row in level]
        return level
    
    def generate_random(self, level_count: int, size: Tuple[int, int], *, mode: str = "basic", enemies: Optional[int] = None) -> List[Level]:
        if mode == "basic": # Randomly selects tiles and every tile has an equal probability
            return super().generate_random(level_count, size)
        elif mode == "naive": # Randomly selects tiles but each tile has a different probability according to how much it is expected to appear
            h, w = size
            area = h * w
            enemies = enemies or max(w, h)
            PLAYER_PROP = KEY_PROP = DOOR_PROP = 1/area
            ENEMY_PROP = enemies / area
            BAT_PROP = SPIDER_PROP = SCORPION_PROP = ENEMY_PROP / 3
            WALL_PROP = EMPTY_PROP = (1 - PLAYER_PROP - KEY_PROP - DOOR_PROP - ENEMY_PROP)/2
            all_tiles_weights = [EMPTY_PROP, WALL_PROP, KEY_PROP, DOOR_PROP, PLAYER_PROP, BAT_PROP, SPIDER_PROP, SCORPION_PROP]  
            all_tiles = list(range(len(self.tiles)))
            return [[random.choices(all_tiles, all_tiles_weights, k=w) for _ in range(h)] for _ in range(level_count)]
        elif mode == "compilable": # The tiles are added while making sure it satisfies the compilability constraints (e.g., only one player is allowed).
            h, w = size
            enemies = enemies or max(w, h)
            levels = []
            for _ in range(level_count):
                enemy_count = random.randint(1, enemies)
                locations = {(j,i) for j in range(h) for i in range(w)}
                enemy_locations = set(random.sample(list(locations), enemy_count))
                locations -= enemy_locations
                player_location, key_location, door_location = random.sample(list(locations), 3)
                locations -= {player_location, key_location, door_location}
                wall_locations = set(random.sample(list(locations), random.randint(0, len(locations))))
                def tile(location):
                    if location in wall_locations: return 1
                    if location in enemy_locations: return random.choice([5, 6, 7])
                    if location == player_location: return 4
                    if location == key_location: return 2
                    if location == door_location: return 3
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
            result["keys"] = keys = counter[2]
            result["doors"] = doors = counter[3]
            result["players"] = players = counter[4]
            bats, spiders, scorpions = counter[5], counter[6], counter[7]
            result["bats"], result["spiders"], result["scorpions"] = bats, spiders, scorpions
            enemies = bats + spiders + scorpions
            result["enemies"] = enemies
            result["enemies-ratio"] = enemies / area
            
            # The L1 distances of important objects from the level center
            center_j, center_i = (h-1)/2, (w-1)/2
            key_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==2]
            result["key-L1"] = key_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in key_positions)/len(key_positions) if key_positions else 0
            result["key-L1-norm"] = key_L1 / (center_i + center_j)
            door_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==3]
            result["door-L1"] = door_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in door_positions)/len(door_positions) if door_positions else 0
            result["door-L1-norm"] = door_L1 / (center_i + center_j)
            player_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==4]
            result["player-L1"] = player_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in player_positions)/len(player_positions) if player_positions else 0
            result["player-L1-norm"] = player_L1 / (center_i + center_j)
            enemy_positions = [(j,i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile > 4]

            compilable = players == 1 and keys == 1 and doors == 1 and 1 <= enemies <= max(w, h)
            result["compilable"] = compilable
            if compilable:
                obstacles = [[tile == 1 for tile in row] for row in level]
                distance_to_player = create_shortest_path_map(obstacles, player_positions)
                enemy_distances = [distance_to_player[j][i] for j, i in enemy_positions]
                key_set = set(key_positions)
                key_to_door_distance = compute_shortest_path_length(obstacles, door_positions, key_set)
                for j, i in door_positions: obstacles[j][i] = True
                player_to_key_distance = compute_shortest_path_length(obstacles, player_positions, key_set)
                
                result["nearest-enemy-distance"] = nearest_enemy_distance = min(enemy_distances)
                result["nearest-enemy-distance-norm"] = nearest_enemy_distance / area
                result["farthest-enemy-distance"] = farthest_enemy_distance = max(enemy_distances)
                result["farthest-enemy-distance-norm"] = farthest_enemy_distance / area

                result["solvable"] = solvable = (farthest_enemy_distance <= area) and player_to_key_distance is not None and key_to_door_distance is not None
                if solvable:
                    path_length = player_to_key_distance + key_to_door_distance
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
        return ZeldaConditionUtility()

class ZeldaConditionUtility(ConditionUtility):
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
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(0), self.const(area-4)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "enemies-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), self.const(max(h, w))), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "nearest-enemy-distance-norm":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), self.const(area)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        return (lambda x, _: x) if size is None else (lambda x: x)
    
    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        h, w = size
        
        if prop_name == "path-length-norm":
            return 0.5/(h*w)
        
        if prop_name == "wall-ratio" or prop_name == "enemies-ratio":
            return 0.5/(h*w)

        if prop_name == "nearest-enemy-distance-norm":
            return 0.5/(h*w)
        
        return super().get_tolerence(prop_name, size)
    
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "path-length-norm":
            return 1/(h*w), 1.0
        
        if prop_name == "wall-ratio":
            area = h*w
            return 0.0, (area-4)/area
        
        if prop_name == "enemies-ratio":
            area = h*w
            return 1/(h*w), max(h, w)/area

        if prop_name == "nearest-enemy-distance-norm":
            area = h*w
            return 1/area, 0.5
        
        return super().get_range_estimates(prop_name, size)