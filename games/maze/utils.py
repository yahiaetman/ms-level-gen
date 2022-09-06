from collections import deque
from typing import List, Optional, Set, Tuple

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


# returns None if any empty location cannot be reached from another empty location
def compute_longest_shortest_path_length(obstacles: List[List[bool]]) -> Optional[int]:
    h, w = len(obstacles), len(obstacles[0])
    infinity = h*w + 1
    walkable = {(y,x) for y, row in enumerate(obstacles) for x, obstacle in enumerate(row) if not obstacle}
    longest = 0
    for start in walkable:
        j, i = start
        distance_map = [[infinity]*w for _ in range(h)]
        distance_map[j][i] = 0
        q = deque()
        q.append(start)
        while q:
            j, i = q.popleft()
            distance = distance_map[j][i] + 1
            for cj, ci in ((j+dj, i+di) for dj, di in DIRECTIONS):
                if (cj, ci) not in walkable or distance_map[cj][ci] <= distance: continue
                distance_map[cj][ci] = distance
                q.append((cj, ci))
        longest = max(longest, max(distance_map[j][i] for j, i in walkable))
        if longest == infinity: return None
    return longest

# def compute_all_distances(walkable: Set[Tuple[int, int]], start: Tuple[int, int]):
#     distance = {start: 0}
#     q = deque()
#     q.append(start)
#     while q:
#         j, i = q.popleft()
#         d = distance[(j, i)] + 1
#         for c in ((j+dj, i+di) for dj, di in DIRECTIONS):
#             if c not in walkable or c in distance: continue
#             q.append(c)
#             distance[c] = d
#     return distance

# def compute_longest_shortest_path_length(obstacles: List[List[bool]]) -> Optional[int]:
#     walkable = {(y,x) for y, row in enumerate(obstacles) for x, obstacle in enumerate(row) if not obstacle}
#     visited = set()
#     longest = 0
#     for start in walkable:
#         if start in visited: continue
#         distances = compute_all_distances(walkable, start)
#         visited.update(distances.keys())
#         _, farthest = max((v, k) for k, v in distances.items())
#         distances = compute_all_distances(walkable, farthest)
#         farthest_distance, _ = max((v, k) for k, v in distances.items())
#         longest = max(longest, farthest_distance)
#     return longest

if __name__ == "__main__":
    # level = [
    #     "W...",
    #     "W.WW",
    #     "W..W",
    #     "W..."
    # ]
    level = [
        ".....",
        ".WWW.",
        ".WWW.",
        ".....",
        ".WWWW"
    ]
    obstacles = [[cell != '.' for cell in row] for row in level]
    print(is_fully_connected(obstacles))
    print(compute_longest_shortest_path_length(obstacles))