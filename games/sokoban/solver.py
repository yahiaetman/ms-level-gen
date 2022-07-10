from collections import deque
from typing import List, Optional

class SokobanSolver:
    action_ascii = 'rlduRLDU'
    action_to_index = {'r': 0, 'l': 1, 'd': 2, 'u': 3, 'R': 4, 'L': 5, 'D': 6, 'U': 7}
    action_to_angle = {'r': 0, 'u': 90, 'l': 180, 'd': 270}
    angle_to_action = {angle: action for action, angle in action_to_angle.items()}

    def __init__(self, level, return_states: bool = False, iteration_limit: Optional[int]=None) -> None:
        self.h = len(level)
        self.w = len(level[0])
        wh, ww = self.h+2, self.w+2
        self.players = set()
        self.walls = bytearray([True for _ in range(wh*ww)])
        self.goals = bytearray([False for _ in range(wh*ww)])
        self.crates = bytearray([False for _ in range(wh*ww)])
        self.directions = [1, -1, ww, -ww]
        position = ww + 1
        # The tiles: ['.', 'W', '0', '1', 'A', 'g', '+']
        for row in level:
            for tile in row:
                if tile != 1:
                    self.walls[position] = False
                if tile == 2:
                    self.goals[position] = True
                elif tile == 3:
                    self.crates[position] = True
                elif tile == 4:
                    self.players.add(position)
                elif tile == 5:
                    self.goals[position] = True
                    self.crates[position] = True
                elif tile == 6:
                    self.goals[position] = True
                    self.players.add(position)
                position += 1
            position += 2
        self.walls = bytes(self.walls)
        self.goals = bytes(self.goals)
        self.crates = bytes(self.crates)
        self.__return_states = return_states
        self.iteration_limit = iteration_limit
        self.solution = None
        self.states = None
        self.iterations = 0
    
    @property
    def compilable(self) -> bool:
        if len(self.players) != 1:
            return False
        if self.crates == self.goals:
            return False
        if sum(self.goals) != sum(self.crates):
            return False
        return True
    
    def __fill_nonlocked(self, nonlocked: bytearray, position: int):
        nonlocked[position] = True
        for d in self.directions:
            new = position + d
            if nonlocked[new] or self.walls[new]:
                continue
            if self.walls[new+d]:
                continue
            self.__fill_nonlocked(nonlocked, new)

    def __reconstruct_state(self, state):
        player, crates = state
        level = [[0] * self.w for _ in range(self.h)]
        position = self.w + 3
        # The tiles: ['.', 'W', '0', '1', 'A', 'g', '+']
        for row in level:
            for x in range(self.w):
                if self.walls[position]:
                    row[x] = 1
                else:
                    if self.goals[position]:
                        if crates[position]:
                            row[x] = 5
                        elif position == player:
                            row[x] = 6
                        else:
                            row[x] = 2
                    else:
                        if crates[position]:
                            row[x] = 3
                        elif position == player:
                            row[x] = 4
                position += 1
            position += 2
        return level

    def __construct_solution(self, visited, goal):
        solution = []
        parent = goal
        while True:
            parent, action = visited[parent]
            if action is None:
                self.solution = ''.join(solution[::-1])
                return
            solution.append(self.action_ascii[action])
    
    def __construct_solution_and_states(self, visited, goal):
        solution = []
        states = [self.__reconstruct_state(goal)]
        parent = goal
        while True:
            parent, action = visited[parent]
            if action is None:
                self.solution = ''.join(solution[::-1])
                self.states = states[::-1]
                return
            solution.append(self.action_ascii[action])
            states.append(self.__reconstruct_state(parent))
    
    def __check_single_2x2_blocks(self, crates: bytes, pos: int, dir: int):
        ortho = self.w + 3 - abs(dir)
        unsafe = 0 if self.goals[pos] else 1
        p10 = pos + dir
        if not(crates[p10] or self.walls[p10]): return False
        if crates[p10] and not self.goals[p10]: unsafe += 1
        for per in [ortho, -ortho]:
            unsafe_local = unsafe
            p01 = pos + per
            if not(crates[p01] or self.walls[p01]): continue
            if crates[p01] and not self.goals[p01]: unsafe_local += 1
            p11 = p10 + per
            if not(crates[p11] or self.walls[p11]): continue
            if crates[p11] and not self.goals[p11]: unsafe_local += 1
            if unsafe_local != 0: return True
        return False
    
    def __check_all_2x2_blocks(self):
        pos = 0
        down = self.w + 2
        for _ in range(self.h+1):
            for _ in range(self.w+1):
                unsafe = 0
                
                p = pos
                pos += 1
                c = self.crates[p]
                w = self.walls[p] 
                if not(c or w): continue
                if c and not self.goals[p]: unsafe += 1
                
                p += 1
                c = self.crates[p]
                w = self.walls[p] 
                if not(c or w): continue
                if c and not self.goals[p]: unsafe += 1
                
                p += down
                c = self.crates[p]
                w = self.walls[p] 
                if not(c or w): continue
                if c and not self.goals[p]: unsafe += 1
                
                p -= 1
                c = self.crates[p]
                w = self.walls[p] 
                if not(c or w): continue
                if c and not self.goals[p]: unsafe += 1
                
                if unsafe != 0: return True
            pos += 1
        return False

    def solve(self):
        walls = self.walls
        goals = self.goals
        directions = self.directions
        iteration_limit = self.iteration_limit
        iterations = 0
        self.iterations = 0
        size = (self.w + 2)*(self.h + 2)
        if self.__check_all_2x2_blocks():
            self.solution = None
            return
        nonlocked = bytearray([False for _ in range(size)])
        for position in range(size):
            if goals[position] and not nonlocked[position]:
                self.__fill_nonlocked(nonlocked, position)
        if any(self.crates[position] and not nonlocked[position] for position in range(size)):
            self.solution = None
            return
        if self.crates == goals:
            self.solution = []
            return
        current = (next(iter(self.players)), self.crates)
        Q = deque()
        Q.append(current)
        visited = {current: (None, None)}
        while Q and (iteration_limit is None or iterations <= iteration_limit):    
            iterations += 1
            current = Q.popleft()
            player, crates = current
            for action, d in enumerate(directions):
                new_player = player + d
                new_crates = crates
                if walls[new_player]:
                    continue
                change = False
                if crates[new_player]:
                    nxt = new_player+d
                    if crates[nxt] or walls[nxt] or not nonlocked[nxt] or self.__check_single_2x2_blocks(crates, nxt, d):
                        continue
                    new_crates = bytearray(crates)
                    new_crates[nxt] = True
                    new_crates[new_player] = False
                    new_crates = bytes(new_crates)
                    action += 4
                    change = True
                child = (new_player, new_crates)
                if child in visited:
                    continue
                Q.append(child)
                visited[child] = (current, action)
                if change and new_crates == goals:
                    if self.__return_states:
                        self.__construct_solution_and_states(visited, child)
                    else:
                        self.__construct_solution(visited, child)
                    self.iterations = iterations
                    return
        self.iterations = iterations

    def compute_pushed_crates(self, solution: str) -> int:
        crates = bytearray(self.crates)
        crate_locations = [index for index, is_crate in enumerate(crates) if is_crate]
        crate_ids = {location: cid for cid, location in enumerate(crate_locations)}
        directions = self.directions
        player = next(iter(self.players))
        pushed_crates = set()
        solution = solution.lower()
        for action in solution:
            action_index = self.action_to_index[action]
            direction = directions[action_index]
            player += direction
            if crates[player]:
                pushed_crates.add(crate_ids[player])
                new_crate = player + direction
                crates[player] = False
                crates[new_crate] = True
                crate_ids[new_crate] = crate_ids[player]
        return len(pushed_crates)
    
    def build_high_level_plan(self, solution: str) -> int:
        crates = bytearray(self.crates)
        crate_ids = {index: None for index, is_crate in enumerate(crates) if is_crate}
        directions = self.directions
        player = next(iter(self.players))
        plan = []
        crate_counter = 0
        current_action = None
        solution = solution.lower()
        for action in solution:
            action_index = self.action_to_index[action]
            direction = directions[action_index]
            player += direction
            if crates[player]:
                new_crate = player + direction
                crates[player] = False
                crates[new_crate] = True
                cid = crate_ids[player]
                if cid is None:
                    cid = crate_counter
                    crate_counter += 1
                crate_ids[new_crate] = cid
                if current_action is None or cid != current_action["crate"]:
                    if current_action is not None:
                        plan.append(current_action)
                    current_action = {"direction": action, "crate": cid, "count": 1}
                else:
                    current_action["count"] += 1
            else:
                if current_action is not None:
                    plan.append(current_action)
                    current_action = None
        if current_action is not None:
            plan.append(current_action)
        return plan
    
    @staticmethod
    def normalize_plan(plan):
        if len(plan) == 0: return plan
        angles = [SokobanSolver.action_to_angle[a["direction"]] for a in plan]
        diff = 360 - angles[0]
        angles = [(angle + diff)%360 for angle in angles]
        first_ortho = next((angle for angle in angles if angle == 90 or angle == 270), 90) 
        if first_ortho == 270:
            angles = [(360 - angle)%360 for angle in angles]
        for i, a in enumerate(plan):
            a["direction"] = SokobanSolver.angle_to_action[angles[i]]