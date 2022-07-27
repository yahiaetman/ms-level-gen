import heapq
from typing import List, Optional, Tuple

def cumulative(items, fn):
    items = list(items)
    for i in range(1, len(items)):
        items[i] = fn(items[i-1], items[i])
    return items

def solve(obstacles: List[List[bool]], starting_point: Tuple[int, int], goal_point: Tuple[int, int]) -> Optional[str]:
    h, w = len(obstacles), len(obstacles[0])
    area = h*w

    sun_up = cumulative(obstacles, lambda a, b: [i or j for i, j in zip(a, b)])
    sun_down = cumulative(obstacles[::-1], lambda a, b: [i or j for i, j in zip(a, b)])[::-1]
    sun_left = [cumulative(row, lambda a, b: a or b) for row in obstacles]
    sun_right = [cumulative(row[::-1], lambda a, b: a or b)[::-1] for row in obstacles]

    walkables = [{(y, x) for y, row in enumerate(shadowmap) for x, shadow in enumerate(row) if shadow and not obstacles[y][x]}
        for shadowmap in [sun_up, sun_right, sun_down, sun_left]]
    
    if starting_point not in walkables[0]:
        return None

    initial_state = (starting_point, 0)
    q = [(0, initial_state)]
    explored = set()
    parents = {initial_state: (None, None, 0)}
    actions = {'d':(1,0,0), 'u':(-1,0,0), 'r':(0,1,0), 'l':(0,-1,0), '.':(0,0,1)}
    while q:
        g, (pos, sun) = heapq.heappop(q)
        if pos == goal_point:
            solution = []
            parent = (pos, sun)
            while True:
                parent, action, _ = parents[parent]
                if action is None:
                    return ''.join(solution[::-1])
                solution.append(action)
        explored.add((pos, sun))
        for action, (dy, dx, dt) in actions.items():
            next_pos = (pos[0] + dy, pos[1] + dx)
            next_sun = (sun + dt)%4
            if next_pos not in walkables[next_sun]: continue
            if (next_pos, next_sun) in explored: continue
            cost = area if dt != 0 else 1
            next_g = g + cost
            child = (next_pos, next_sun)
            if child not in parents or parents[child][2] > next_g:
                parents[child] = ((pos, sun), action, next_g)
                heapq.heappush(q, (next_g, child))
    return None

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