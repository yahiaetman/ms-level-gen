from collections import deque
from typing import List, Optional, Set, Tuple

def compute_shortest_path_length(obstacles: List[List[bool]], starting_points: List[Tuple[int, int]], goal_points: Set[Tuple[int, int]]) -> Optional[int]:
    h, w = len(obstacles), len(obstacles[0])
    infinity = h*w + 1
    distance_map = [[infinity]*w for _ in range(h)]
    q = deque()
    for j, i in starting_points:
        if (j, i) in goal_points: return 0
        distance_map[j][i] = 0
        q.append((j, i))
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        j, i = q.popleft()
        distance = distance_map[j][i] + 1
        for cj, ci in ((j+dj, i+di) for dj, di in directions):
            if (cj, ci) in goal_points: return distance
            if not (0<=cj<h and 0<=ci<w) or obstacles[cj][ci] or distance_map[cj][ci] <= distance: continue
            distance_map[cj][ci] = distance
            q.append((cj, ci))
    return None

def create_shortest_path_map(obstacles: List[List[bool]], starting_points: List[Tuple[int, int]]) -> List[List[int]]:
    h, w = len(obstacles), len(obstacles[0])
    infinity = h*w + 1
    distance_map = [[infinity]*w for _ in range(h)]
    q = deque()
    for j, i in starting_points:
        distance_map[j][i] = 0
        q.append((j, i))
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        j, i = q.popleft()
        distance = distance_map[j][i] + 1
        for cj, ci in ((j+dj, i+di) for dj, di in directions):
            if not (0<=cj<h and 0<=ci<w) or obstacles[cj][ci] or distance_map[cj][ci] <= distance: continue
            distance_map[cj][ci] = distance
            q.append((cj, ci))
    return distance_map