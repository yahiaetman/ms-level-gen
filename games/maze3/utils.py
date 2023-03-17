from collections import deque
from typing import List, Optional, Set, Tuple

DIRECTIONS = [(1,0),(-1,0),(0,1),(0,-1)]

def compute_all_distances(walkable: Set[Tuple[int, int]], start: Tuple[int, int]):
    distance = {start: 0}
    q = deque()
    q.append(start)
    while q:
        j, i = q.popleft()
        d = distance[(j, i)] + 1
        for c in ((j+dj, i+di) for dj, di in DIRECTIONS):
            if c not in walkable or c in distance: continue
            q.append(c)
            distance[c] = d
    return distance

def compute_shortest_path_length(obstacles: List[List[bool]]) -> Optional[int]:
    h, w = len(obstacles), len(obstacles[0])
    if obstacles[0][0] or obstacles[-1][-1]: return None
    walkable = {(y,x) for y, row in enumerate(obstacles) for x, obstacle in enumerate(row) if not obstacle}
    distances = compute_all_distances(walkable, (0, 0))
    return distances.get((h-1, w-1))

if __name__ == "__main__":
    # level = [
    #     "W...",
    #     "W.WW",
    #     "W..W",
    #     "W..."
    # ]
    level = [
        ".....",
        "WWWW.",
        ".....",
        ".WWWW",
        "....."
    ]
    obstacles = [[cell != '.' for cell in row] for row in level]
    print(compute_shortest_path_length(obstacles))