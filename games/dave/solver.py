from collections import deque
from typing import Optional

class DaveSolver:
    action_ascii = 'rl.j'

    def __init__(self, level, air_time: int = 3, hang_time: int = 1, iteration_limit: Optional[int]=None) -> None:
        self.air_time = air_time
        self.hang_time = hang_time
        self.h = len(level)
        self.w = len(level[0])

        self.players = set()
        self.keys = set()
        self.goals = set()
        self.open = set()
        self.diamonds = set()
        self.spikes = set()
        # The tiles: ['.', 'W', '+, 'g', 'A', '$', '*]
        for y, row in enumerate(level):
            for x, tile in enumerate(row):
                position = (y, x)
                if tile != 1:
                    self.open.add(position)
                if tile == 2:
                    self.keys.add(position)
                elif tile == 3:
                    self.goals.add(position)
                elif tile == 4:
                    self.players.add(position)
                elif tile == 5:
                    self.diamonds.add(position)
                elif tile == 6:
                    self.spikes.add(position)
        self.iteration_limit = iteration_limit
        
    @property
    def compilable(self) -> bool:
        if len(self.players) != 1:
            return False
        if len(self.keys) != 1:
            return False
        if len(self.goals) != 1:
            return False
        y, x = next(iter(self.players))
        if (y+1, x) in self.open: # Must start on the floor
            return False
        if not(0 < len(self.diamonds) <= max(self.w, self.h)):
            return False
        k = (self.w - 1) * (self.h // 2)
        if not(k // 3 <= len(self.spikes) <= k):
            return False
        return True

    def __construct_solution(self, visited, goal):
        solution = []
        parent = goal
        while True:
            parent, action = visited[parent]
            if action is None:
                return ''.join(solution[::-1])
            solution.append(action)
    
    def solve(self):
        key_location = next(iter(self.keys))
        goal_location = next(iter(self.goals))
        diamonds = self.diamonds
        for diamond in diamonds:
            #print('D', self.get_path_through([diamond]))
            if self.get_path_through([diamond]) is None: return None
        return self.get_path_through([key_location, goal_location])

    def get_path_through(self, points):
        open = self.open
        spikes = self.spikes
        
        def _apply_physics(y, x, air_time):
            if air_time > 0: # going up
                if air_time > self.hang_time: 
                    y -= 1
                    air_time -= 1
                    if (y, x) not in open: # hit ceiling
                        y += 1
                        air_time = min(air_time, self.hang_time)
                else:
                    air_time -= 1
            else: # going down
                y += 1
                if (y, x) not in open: # hit floor
                    y -= 1 
            return y, x, air_time

        iteration_limit = self.iteration_limit
        iterations = 0
        current = (next(iter(self.players)), 0, 0)
        Q = deque()
        Q.append(current)
        visited = {current: (None, None)}
        while Q and (iteration_limit is None or iterations <= iteration_limit):    
            iterations += 1
            current = Q.popleft()
            (y, x), air_time, points_reached = current
            children = [('.', y, x, air_time)]
            # walk actions
            for action, dx in [('r', 1), ('l', -1)]:
                nx = x + dx
                # check movement does not go into wall or spike
                if (y, nx) not in open or (y, nx) in spikes:
                    continue
                children.append((action, y, nx, air_time))
            if (y+1, x) not in open and (y-1, x) in open: # check if on ground and no ceiling above:
                children.append(('j', y, x, self.air_time))
            for action, y, x, air_time in children:
                y, x, air_time = _apply_physics(y, x, air_time)
                if (y, x) in spikes: continue
                child_points_reached = points_reached
                if (y, x) == points[points_reached]:
                    child_points_reached += 1
                child = ((y, x), air_time, child_points_reached)
                if child in visited: continue
                Q.append(child)
                visited[child] = (current, action)
                if child_points_reached == len(points): # it is a goal                    
                    return self.__construct_solution(visited, child)
        return None

if __name__ == "__main__":
    levels = ["""
    .........$g
    .+.......WW
    .WW...$....
    .....WWW..$
    .........WW
    A....$$$...
    WW..WWWWW..
    """, """
    ..........g
    .........WW
    +...$$.....
    WW**WW*WW..
    .........$$
    .........WW
    A..$$$$..WW
    """, """
    g.........+
    wwww..$..WW
    ......W....
    A..........
    WWW**WWW**.
    $$$......$$
    $$$......WW
    """, """
    .......$..+
    .......wwww
    $$...$$....
    ww...ww....
    ..$$$..$$$.
    ..www..www.
    A......wg..
    """, """
    A.........g
    ww.......ww
    ...........
    .....ww....
    .........$$
    $$...+$..ww
    ww..wwww...
    """]
    for level in levels:
        level = level.strip().lower().splitlines(keepends=False)
        #print([[tile for tile in row if tile not in '.w+ga$*'] for row in level])
        level = [['.w+ga$*'.index(tile) for tile in row.strip()] for row in level]
        solver = DaveSolver(level)
        print(solver.solve())