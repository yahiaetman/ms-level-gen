from typing import Callable, List, Optional, Tuple
from games.game import Level
from concurrent.futures import ThreadPoolExecutor
import sokosolve
import os, queue
from threading import Lock

class SolverQueue:
    def __init__(self, height:int, width:int, capacity: int, cpu_count: Optional[int] = None) -> None:
        self.height = height
        self.width = width
        self.capacity = int(capacity)
        self.cpu_count = cpu_count or os.cpu_count()
        self.queue = queue.Queue(cpu_count)
        self.allocated = 0
        self.mutex = Lock()
    
    def get(self) -> sokosolve.SokobanSolver:
        with self.mutex:
            if self.allocated < self.cpu_count and self.queue.empty():
                self.allocated += 1
                return sokosolve.SokobanSolver(self.width, self.height, self.capacity)
            else:
                return self.queue.get()
    
    def put(self, solver: sokosolve.SokobanSolver):
        self.queue.put(solver)


class SokobanParallelSolver:
    def __init__(self, tiles: str, iteration_limit_fn: Optional[Callable[[int,int], int]] = None, cpu_count: Optional[int] = None):
        self.tiles = tiles
        self.cpu_count = cpu_count or os.cpu_count()
        self.iteration_limit_fn = iteration_limit_fn or (lambda *_: int(1e6))
        self.queues = {}
        self.executor = ThreadPoolExecutor(max_workers=cpu_count)

    def solve_level(self, level: Level) -> dict:
        h = len(level)
        if h == 0: return {"compilable": False, "solution": None, "iterations": 0}
        w = len(level[0])
        iteration_limit = int(self.iteration_limit_fn(h,w))
        if (h,w) not in self.queues:
            capacity = 4 * (iteration_limit or int(1e6))
            queue = SolverQueue(h, w, capacity, self.cpu_count)
            self.queues[(h, w)] = queue
        else:
            queue = self.queues[(h, w)]
        solver = queue.get()
        level_str = ''.join(''.join(self.tiles[tile] for tile in row) for row in level)
        compilable = solver.parse_level(level_str)
        if compilable:
            result = solver.solve_bfs(iteration_limit)
            solution = result.actions.decode("utf-8") if result.actions is not None else None
            answer = {"compilable": True, "solution": solution, "iterations": result.iterations}
        else:
            answer = {"compilable": False, "solution": None, "iterations": 0}
        queue.put(solver)
        return answer

    def solve_levels(self, levels: List[Level]) -> List[dict]:
        return list(self.executor.map(self.solve_level, levels))