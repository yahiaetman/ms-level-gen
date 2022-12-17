from collections import deque
import heapq
from typing import List, Optional, Tuple

DIRECTIONS = [(1,0),(-1,0),(0,1),(0,-1)]

def is_fully_connected(obstacles: List[List[bool]]) -> bool:
    walkable = {(y,x) for y, row in enumerate(obstacles) for x, obstacle in enumerate(row) if not obstacle}
    start = next(iter(walkable))
    walkable.remove(start)
    q = deque()
    q.append(start)
    while q:
        j, i = q.popleft()
        for c in ((j+dj, i+di) for dj, di in DIRECTIONS):
            if c in walkable:
                walkable.remove(c)
                q.append(c)
    return len(walkable) == 0

def solve(obstacles: List[List[bool]], starting_point: Tuple[int, int], goal_point: Tuple[int, int], h_factor: float = 1, g_factor: float = 1) -> Optional[Tuple[str,int]]:
    h, w = len(obstacles), len(obstacles[0])
    area = h*w

    walkable = {(y, x) for y, row in enumerate(obstacles) for x, obstacle in enumerate(row) if not obstacle}
    
    h_fn = lambda pos: abs(pos[0] - goal_point[0]) + abs(pos[1] - goal_point[1])

    initial_state = starting_point
    q = [(h_factor * h_fn(initial_state), 0, initial_state)]
    explored = set()
    parents = {initial_state: (None, None, 0)}
    actions = {'d':(1,0), 'u':(-1,0), 'r':(0,1), 'l':(0,-1)}
    while q:
        _, g, pos = heapq.heappop(q)
        explored.add(pos)
        if pos == goal_point:
            solution = []
            parent = pos
            while True:
                parent, action, _ = parents[parent]
                if action is None:
                    return ''.join(solution[::-1]), len(explored)
                solution.append(action)
        for action, (dy, dx) in actions.items():
            next_pos = (pos[0] + dy, pos[1] + dx)
            if next_pos not in walkable: continue
            if next_pos in explored: continue
            next_g = g + 1
            child = next_pos
            if child not in parents or parents[child][2] > next_g:
                parents[child] = (pos, action, next_g)
                heapq.heappush(q, (h_factor * h_fn(child) + g_factor * next_g, next_g, child))
    return None, len(explored)

if __name__ == "__main__":
    level = """
    ww..
    ...w
    ..g.
    A.w.
    """
    level = [row.strip() for row in level.strip().lower().splitlines()]
    player = next(((y, x) for y, row in enumerate(level) for x, cell in enumerate(row) if cell == 'a'), None)
    assert player is not None, "There is no player"
    goal = next(((y, x) for y, row in enumerate(level) for x, cell in enumerate(row) if cell == 'g'), None)
    assert goal is not None, "There is no goal"
    obstacles = [[cell == 'w' for cell in row] for row in level]
    print(solve(obstacles, player, goal))