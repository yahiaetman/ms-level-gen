from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .solver import DaveSolver

"""
The Danger Dave game implementation.

Tileset:
--------
.   empty
W   wall
+   key
g   door
A   player
$   diamond
*   spike
"""

class Dave(Game):
    def __init__(self, cache_analysis: bool = True, **kwargs) -> None:
        super().__init__("DAVE", '.W+gA$*', "games/dave/sprites.png")
        self.cache = {} if cache_analysis else None
    
    @property
    def possible_augmentation_count(self) -> int:
        return 2 # 2 for the horizontal flipping (no vertical flipping due to gravity).
    
    def augment_level(self, level: Level, augnmentation_bits: Optional[int] = None) -> Level:
        if augnmentation_bits is None: augnmentation_bits = random.randint(0, 1<<1 - 1)
        if augnmentation_bits & 1:
            level = [row[::-1] for row in level]
        return level

    def generate_random(self, level_count: int, size: Tuple[int, int], *, mode: str = "basic", diamonds: Optional[int] = None, spikes: Optional[int] = None) -> List[Level]:
        if mode == "basic": # Randomly selects tiles and every tile has an equal probability
            return super().generate_random(level_count, size)
        elif mode == "naive": # Randomly selects tiles but each tile has a different probability according to how much it is expected to appear
            h, w = size
            diamonds = diamonds or max(w, h)
            spikes = spikes or (max(w, h) - 1)
            PLAYER_PROP = KEY_PROP = DOOR_PROP = 1/(w*h)
            DIAMOND_PROP = diamonds/(w*h)
            SPIKE_PROP = spikes/(w*h)
            WALL_PROP = EMPTY_PROP = (1 - PLAYER_PROP - KEY_PROP - DOOR_PROP - DIAMOND_PROP - SPIKE_PROP)/2
            all_tiles_weights = [EMPTY_PROP, WALL_PROP, KEY_PROP, DOOR_PROP, PLAYER_PROP, DIAMOND_PROP, SPIKE_PROP]  
            all_tiles = list(range(7)) 
            return [[random.choices(all_tiles, all_tiles_weights, k=w) for _ in range(h)] for _ in range(level_count)]
        elif mode == "compilable": # The tiles are added while making sure it satisfies the compilability constraints (e.g., only one player is allowed).
            h, w = size
            diamonds = diamonds or max(w, h)
            spikes = spikes or (max(w, h) - 1)
            levels = []
            for _ in range(level_count):
                diamond_count = random.randint(1, diamonds)
                spike_count = random.randint(spikes, (h*w)//2)
                
                locations = {(j,i) for j in range(h) for i in range(w)}
                
                player_location = random.choice(list(locations))
                locations.remove(player_location)
                stand_location = (player_location[0]+1, player_location[1])
                if stand_location in locations: locations.remove(stand_location)
                
                key_location, door_location = random.sample(list(locations), 2)
                locations.remove(key_location)
                locations.remove(door_location)
                
                diamond_locations = set(random.sample(list(locations), min(len(locations)-1, diamond_count)))
                locations -= diamond_locations
                
                spike_locations = set(random.sample(list(locations), min(len(locations), spike_count)))
                locations -= spike_locations
                
                wall_locations = set(random.sample(list(locations), random.randint(0, len(locations))))
                wall_locations.add(stand_location)
                
                def tile(location):
                    if location in wall_locations: return 1
                    if location == key_location: return 2
                    if location == door_location: return 3
                    if location == player_location: return 4
                    if location in diamond_locations: return 5
                    if location in spike_locations: return 6
                levels.append([[tile((j,i)) for i in range(w)] for j in range(h)])
            return levels

    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        results = [None]*len(levels)
        # uniquify the levels to avoid invoking the solver for duplicates
        cache = self.cache
        unique_levels_to_solve = []
        unique_level_to_result_indices = []
        seen_levels = {}
        for index, level in enumerate(levels):
            tile_key = tuple(tuple(row) for row in level)
            if cache is not None:
                if (result := cache.get(tile_key)) is not None:
                    results[index] = cache[tile_key]
                    continue
            if tile_key in seen_levels:
                unique_level_to_result_indices[seen_levels[tile_key]].append(index)
                continue
            seen_levels[tile_key] = len(unique_levels_to_solve)
            unique_levels_to_solve.append(level)
            unique_level_to_result_indices.append([index])
        
        def solve(level):
            solver = DaveSolver(level)
            if solver.compilable:
                solution = solver.solve()
                return {"compilable": True, "solution": solution}
            else:
                return {"compilable": False, "solution": None}

        solutions = map(solve, unique_levels_to_solve)
        # Tiles: ['.', 'W', '+', 'g', 'A', '$', '*']
        for result_indices, level, solution in zip(unique_level_to_result_indices, unique_levels_to_solve, solutions):
            h, w = len(level), len(level[0])
            area = h * w
            result = {"level": level, **solution}
            counter = Counter(tile for row in level for tile in row)
            
            walls = counter[1]
            result["walls"] = walls
            result["wall-ratio"] = walls / area
            
            diamonds = counter[5]
            spikes = counter[6]
            result["diamonds"] = diamonds
            result["diamonds-ratio"] = diamonds / max(h, w)
            result["spikes"] = spikes
            result["spikes-ratio"] = spikes / area

            center_j, center_i = (h-1)/2, (w-1)/2
            player_positions = [(j, i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==4]
            result["player-count"] = len(player_positions)
            player_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in player_positions)/len(player_positions) if player_positions else 0
            result["player-L1"] = player_L1 # Player L1 distance from the level center
            result["player-L1-norm"] = player_L1 / (center_j + center_i) # Player L1 distance from the center, normalized from 0 to 1
            
            solution: str = result["solution"]
            solvable = solution is not None
            result["solvable"] = solvable
            if solvable:
                solution_length = len(solution)
                result["solution-length"] = solution_length
                result["difficulty"] = solution_length / area
                jumps = solution.count('j')
                result["jumps"] = jumps
                result["jump-ratio"] = jumps / max(h, w)
            else:
                result["solution-length"] = -1
                result["difficulty"] = -1
                result["jumps"] = -1
                result["jump-ratio"] = -1
            
            for result_index in result_indices:
                results[result_index] = result
            
            if cache is not None:
                cache[tuple(tuple(row) for row in level)] = result
        
        return results
    
    @property
    def condition_utility(self) -> ConditionUtility:
        return DaveConditionUtility()

class DaveConditionUtility(ConditionUtility):
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        
        if prop_name == "difficulty":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), None), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "diamonds-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                wh_max = max(h, w)
                return self.mul(self.clamp(self.round(self.mul(x, self.const(wh_max))), self.const(1), self.const(wh_max)), self.const(1/wh_max))
            return snap if size is None else (lambda x: snap(x, size)) 
        
        if prop_name == "spikes-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                spikes_max = int((w - 1) * (h / 2))
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(0), self.const(spikes_max)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "jump-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                wh_max = max(h, w)
                return self.mul(self.clamp(self.round(self.mul(x, self.const(wh_max))), self.const(0), None), self.const(1/wh_max))
            return snap if size is None else (lambda x: snap(x, size)) 
        
        return (lambda x, _: x) if size is None else (lambda x: x)
    
    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        h, w = size
        
        if prop_name == "difficulty":
            return 0.5/(h*w)
        
        if prop_name == "diamonds-ratio":
            return 0.5/max(h, w)
        
        if prop_name == "spikes-ratio":
            return 0.5/(h*w)

        if prop_name == "jump-ratio":
            return 0.5/max(h, w)
        
        return super().get_tolerence(prop_name, size)
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "difficulty":
            return 1/(h*w), 1.0
        
        if prop_name == "diamonds-ratio":
            hw_max = max(h, w)
            return 1/hw_max, 1
        
        if prop_name == "spikes-ratio":
            area = h*w
            return 0, (area-3)/area

        if prop_name == "jump-ratio":
            area = h*w
            return 0.0, 1.0
        
        return super().get_range_estimates(prop_name, size)