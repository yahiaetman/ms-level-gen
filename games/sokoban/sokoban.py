from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from games.game import Game, Level, ConditionUtility
import random
from .parallel_solver import SokobanParallelSolver
from .solver import SokobanSolver as PythonSolver

class Sokoban(Game):
    def __init__(self, solver_iteration_limits: Dict[Tuple[int, int], int] = None, solver_cpu_count: Optional[int] = None, cache_analysis: bool = True, **kwargs) -> None:
        super().__init__("SOKOBAN", '.W01Ag+', "games/sokoban/sprites.png", {"expand": self.expand_dataset})
        solver_iteration_limits = solver_iteration_limits or {}
        if () in solver_iteration_limits:
            default_iteration_limit = solver_iteration_limits[()]
        else:
            default_iteration_limit = max(solver_iteration_limits.values(), default=int(1e6))
        iteration_limit_fn = lambda h, w: solver_iteration_limits.get((h, w), default_iteration_limit)
        self.solver = SokobanParallelSolver(self.tiles, iteration_limit_fn, solver_cpu_count)
        self.cache = {} if cache_analysis else None
    
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

    def generate_random(self, level_count: int, size: Tuple[int, int], *, mode: str = "basic", crates: Optional[int] = None) -> List[Level]:
        if mode == "basic":
            return super().generate_random(level_count, size)
        elif mode == "naive":
            h, w = size
            crates = crates or max(w,h)
            RAW_PLAYER_PROP = 1/(w*h)
            RAW_CRATE_PROP = RAW_GOAL_PROP = crates/(w*h)
            PLAYER_PROP = RAW_PLAYER_PROP * (1 - RAW_GOAL_PROP)
            PLAYER_GOAL_PROP = RAW_PLAYER_PROP * RAW_GOAL_PROP
            CRATE_PROP = RAW_CRATE_PROP * (1 - RAW_GOAL_PROP)
            CRATE_GOAL_PROP = RAW_CRATE_PROP * RAW_GOAL_PROP
            GOAL_PROP = RAW_GOAL_PROP - PLAYER_GOAL_PROP - CRATE_GOAL_PROP
            WALL_PROP = EMPTY_PROP = (1 - RAW_PLAYER_PROP - RAW_CRATE_PROP)/2
            all_tiles_weights = [EMPTY_PROP, WALL_PROP, GOAL_PROP, CRATE_PROP, PLAYER_PROP, CRATE_GOAL_PROP, PLAYER_GOAL_PROP]  
            all_tiles = list(range(7)) 
            return [[random.choices(all_tiles, all_tiles_weights, k=w) for _ in range(h)] for _ in range(level_count)]
        elif mode == "compilable":
            h, w = size
            crates = crates or max(w,h)
            levels = []
            for _ in range(level_count):
                crate_count = random.randint(1, crates)
                locations = {(j,i) for j in range(h) for i in range(w)}
                goal_locations = set(random.sample(list(locations), crate_count))
                crate_locations = set(random.sample(list(locations), crate_count))
                locations -= crate_locations
                player_location = random.sample(list(locations), 1)[0]
                locations.remove(player_location)
                locations -= goal_locations
                wall_locations = set(random.sample(list(locations), random.randint(0, len(locations))))
                def tile(location):
                    if location in wall_locations: return 1
                    is_goal = location in goal_locations
                    if location in crate_locations: return 5 if is_goal else 3
                    if location == player_location: return 6 if is_goal else 4
                    return 2 if is_goal else 0
                levels.append([[tile((j,i)) for i in range(w)] for j in range(h)])
            return levels

    def analyze(self, levels: List[Level], **kwargs) -> List[Dict[str, Any]]:
        results = [None]*len(levels)
        # uniquify
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
        
        solutions = self.solver.solve_levels(unique_levels_to_solve)
        # Tiles: ['.', 'W', '0', '1', 'A', 'g', '+']
        for result_indices, level, solution in zip(unique_level_to_result_indices, unique_levels_to_solve, solutions):
            h, w = len(level), len(level[0])
            area = h * w
            result = {"level": level, **solution}
            counter = Counter(tile for row in level for tile in row)
            
            walls = counter[1]
            result["walls"] = walls
            result["wall-ratio"] = walls / area
            
            crates = counter[3] + counter[5]
            goals = counter[2] + counter[5] + counter[6]
            result["crates"] = crates
            result["crate-ratio"] = crates / area
            result["goals"] = goals

            center_j, center_i = (h-1)/2, (w-1)/2
            player_positions = [(j, i) for j, row in enumerate(level) for i, tile in enumerate(row) if tile==4 or tile==6]
            result["player-count"] = len(player_positions)
            player_L1 = sum((abs(j-center_j) + abs(i-center_i)) for j, i in player_positions)/len(player_positions) if player_positions else 0
            result["player-L1"] = player_L1
            result["player-L1-norm"] = player_L1 / (center_j + center_i)
            
            solution = result["solution"]
            solvable = solution is not None
            result["solvable"] = solvable
            if solvable:
                solution_length = len(solution)
                result["solution-length"] = solution_length
                result["difficulty"] = solution_length / area
                dummy_solver = PythonSolver(level)
                plan = dummy_solver.build_high_level_plan(solution)
                PythonSolver.normalize_plan(plan)
                result["plan"] = plan
                result["push-signature"] = [action["direction"] for action in plan]
                crate_signature = [action["crate"] for action in plan]
                result["crate-signature"] = crate_signature
                pushed_crates = max(crate_signature) + 1
                result["pushed-crates"] = pushed_crates
                result["pushed-crate-ratio"] = pushed_crates / max(w, h)
            else:
                result["solution-length"] = -1
                result["difficulty"] = -1
                result["plan"] = None
                result["push-signature"] = None
                result["crate-signature"] = None
                result["pushed-crates"] = 0
                result["pushed-crate-ratio"] = 0
            
            for result_index in result_indices:
                results[result_index] = result
            
            if cache is not None:
                cache[tuple(tuple(row) for row in level)] = result
        
        return results
    
    def expand_dataset(self, dataset: List[Level]) -> List[Level]:
        expanded_dataset = []
        for level in dataset:
            solver = PythonSolver(level, True)
            solver.solve()
            states = solver.states
            if states is not None:
                for state in states[:-1]:
                    expanded_dataset.append(state)
        return expanded_dataset
    
    @property
    def condition_utility(self) -> ConditionUtility:
        return SokobanConditionUtility()

class SokobanConditionUtility(ConditionUtility):
    def get_snapping_function(self, prop_name: str, size: Optional[Tuple[int, int]] = None) -> Union[Callable[[float, Tuple[int, int]], float], Callable[[float], float]]:
        
        if prop_name == "difficulty":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), None), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "pushed-crate-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                wh_max = max(h, w)
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(wh_max))), self.const(1), self.const(area-2)), self.const(1/wh_max))
            return snap if size is None else (lambda x: snap(x, size)) 
        
        if prop_name == "wall-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(0), self.const(area-3)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size))
        
        if prop_name == "crate-ratio":
            def snap(x: float, size: Tuple[int, int]):
                h, w = size
                area = h*w
                return self.mul(self.clamp(self.round(self.mul(x, self.const(area))), self.const(1), self.const(area-2)), self.const(1/area))
            return snap if size is None else (lambda x: snap(x, size)) 
        
        return (lambda x, _: x) if size is None else (lambda x: x)
    
    def get_tolerence(self, prop_name: str, size: Tuple[int, int]) -> float:
        h, w = size
        
        if prop_name == "difficulty":
            return 0.5/(h*w)
        
        if prop_name == "pushed-crate-ratio":
            return 0.5/max(h, w)
        
        if prop_name == "wall-ratio":
            return 0.5/(h*w)

        if prop_name == "crate-ratio":
            return 0.5/(h*w)
        
        return super().get_tolerence(prop_name, size)
    
    def get_range_estimates(self, prop_name: str, size: Tuple[int, int]) -> Tuple[float, float]:
        h, w = size
        
        if prop_name == "difficulty":
            return 1/(h*w), 3.0
        
        if prop_name == "pushed-crate-ratio":
            hw_max = max(h, w)
            return 1/hw_max, (h*w - 2)/hw_max
        
        if prop_name == "wall-ratio":
            area = h*w
            return 0.0, (area-3)/area

        if prop_name == "crate-ratio":
            area = h*w
            return 1/area, (area-2)/area
        
        return super().get_range_estimates(prop_name, size)