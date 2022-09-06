from collections import deque
from typing import List, Optional, Set, Tuple

def compute_shortest_path_length(obstacles: List[List[bool]], starting_points: List[Tuple[int, int]], goal_points: Set[Tuple[int, int]]) -> Optional[int]:
    """Computes the shortest path length from any of the given starting points to any of the given goals.
    The path cannot intersect an obstacle and if no path is found, this function returns None.

    Parameters
    ----------
    obstacles : List[List[bool]]
        A 2D array where obstacles are marked with True.
    starting_points : List[Tuple[int, int]]
        All the starting points.
    goal_points : Set[Tuple[int, int]]
        All the goal points.

    Returns
    -------
    Optional[int]
        The shortest path length from any of the given starting points to any of the given goals.
        If no path is found, this function returns None.
    """
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
    """Given a list of starting points, this function returns a 2D map of the shortest path distance
    to any of the starting points. Unreachable points (including obstacles) will have the value (area + 1).

    Parameters
    ----------
    obstacles : List[List[bool]]
        A 2D array where obstacles are marked with True.
    starting_points : List[Tuple[int, int]]
        All the starting points.

    Returns
    -------
    List[List[int]]
        A 2D map of the shortest path distance to any of the starting points.
        Unreachable points (including obstacles) will have the value (area + 1).
    """
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