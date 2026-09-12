from turtles import obstacle_field as obs
import numpy as np
import tkinter as tk
from tkinter import ttk as ttk
import functools
import time
from astar import AStar

ENEMY_COUNT = 10
GOAL_COUNT = 1

# cell states
EMPTY = 0
WALL = 1
JUNK = 2
GOAL = 3
HERO = 4
ENEMY = 5

# game results
GAME_OVER = -1

_movers = []
_hero = None
_goals = []

time_rate = 1

_rng = np.random.default_rng()

class Mover:
    def __init__(self, x, y, cell_type):
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.next_move = (0, 0)

    def move(self, strict = True):
        (dx, dy) = self.next_move
        if (abs(dx) == 1 and abs(dy) == 0) or (abs(dx) == 0 and abs(dy) == 1) or not strict:
            self.x += dx
            self.y += dy
            return True
        else:
            print(f'Illegal move by type {self.cell_type} at ({self.x}, {self.y}): ({dx}, {dy})')
            return False

    def collide(self, with_list):
        return # no-op for base class

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        return # no-op for base class

    def __str__(self):
        return f'({int(self.x)}, {int(self.y)}): type {self.cell_type}'

    def __repr__(self):
        return self.__str__()

class Enemy(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY)

    def move(self, strict = True):
        if self.next_move == (0, 0):
            return
        super().move()

    def collide(self, with_list):
        if self.cell_type == ENEMY: # dead enemies don't need to do anything
            for other in with_list:
                if other == self:
                    continue
                with_type = other.cell_type if isinstance(other, Mover) else other
                if with_type == HERO:
                    print(f'Enemy killed hero at ({self.x}, {self.y})!')
                    return GAME_OVER
                elif with_type == WALL or with_type == JUNK or with_type == ENEMY:
                    print(f'Enemy junked at ({self.x}, {self.y})!')
                    self.cell_type = JUNK
        return None

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        if self.cell_type == ENEMY:
            dx = hero.x - self.x
            dy = hero.y - self.y
            if abs(dx) > abs(dy):
                self.next_move = (1 if dx > 0 else -1, 0)
            else:
                self.next_move = (0, 1 if dy > 0 else -1)
        else:
            self.next_move = (0, 0)

class Hero(Mover, AStar):
    def __init__(self, x, y):
        Mover.__init__(self, x, y, HERO)
        AStar.__init__(self)
        self.teleports = 5
        self.teleporting = False
        self.cells = []
        self.goal = (x, y)

    def move(self, strict = True):
        if self.next_move == (0, 0):
            return
        super().move(strict = not self.teleporting)
        if self.teleporting:
            self.teleports -= 1

    def collide(self, with_list):
        for other in with_list:
            if other == self:
                continue
            with_type = other.cell_type if isinstance(other, Mover) else other
            if with_type == EMPTY:
                return None
            if with_type == HERO:
                print(f'Hero collided with self? ({self.x}, {self.y})')
                return GAME_OVER
            if with_type == GOAL:
                print(f'Hero reached goal at ({self.x}, {self.y})!')
                return GAME_OVER
            else:
                print(f'Hero hit obstacle at ({self.x}, {self.y})!')
                return GAME_OVER
        return None

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        self.next_move = (0, 0)
        self.teleporting = False
        if cells is not None:
            self.cells = cells
        for mover in movers:
            if mover == self:
                continue
            if abs(mover.x - self.x) <= 1 and abs(mover.y - self.y) <= 1 and self.teleports > 0:
                self.next_move = (_rng.integers(-self.x,  -self.x + obs.FIELD_WIDTH, dtype=int), _rng.integers(-self.y, -self.y + obs.FIELD_HEIGHT, dtype=int))
                self.teleporting = True
                print(f'Hero (threatened) teleporting to ({self.x + self.next_move[0]}, {self.y + self.next_move[1]}), {self.teleports - 1} remaining')
                return
        if goals is not None and len(goals) > 0:
            self.goal = goals[0]
            path = self.astar((self.x, self.y), self.goal)
            if path:
                path = list(path)
                #print(f'Hero path: {path}')
                path_next = list(path)[1] # the first item in path is always the current position
                self.next_move = (path_next[0] - self.x, path_next[1] - self.y)
            else:
                print(f'Hero at ({self.x, self.y}) has no path to goal at ({goals[0][0], goals[0][1]})')
                if self.teleports > 0:
                    self.next_move = (_rng.integers(-self.x, -self.x + obs.FIELD_WIDTH, dtype=int),
                                      _rng.integers(-self.y, -self.y + obs.FIELD_HEIGHT, dtype=int))
                    self.teleporting = True
                    print(f'Hero (stuck) teleporting to ({self.x + self.next_move[0]}, {self.y + self.next_move[1]}), {self.teleports - 1} remaining')
    #
    # AStar impl
    #
    def neighbors(self, node):
        x, y = node
        if 0 <= y < len(self.cells) and 0 <= x < len(self.cells[y]):
            return [(x + i[0], y + i[1])
                    for i in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= y + i[1] < len(self.cells) and 0 <= x + i[0] < len(self.cells[y])
                    and is_cell_empty(self.cells[y + i[1]][x + i[0]]) or (x + i[0] == self.x and y + i[1] == self.y)
                    or (x + i[0] == self.goal[0] and y + i[1] == self.goal[1])]
        return []

    def heuristic_cost_estimate(self, current, goal) -> float:
        cx, cy = current
        gx, gy = goal
        if 0 <= cy < len(self.cells) and 0 <= cx < len(self.cells[cy]):
            if not is_cell_empty(self.cells[cy][cx]) and (cx != self.x and cy != self.y) and (cx != gx and cy != gy):
                return float('inf')
            return np.sqrt(np.power(gx - cx, 2) + np.power(gy - cy, 2))
        return float('inf')

    def distance_between(self, n1, n2) -> float:
        return abs(n2[0] - n1[0]) + abs(n2[1] - n1[1])

    def is_goal_reached(self, current, goal) -> bool:
        return current[0] == goal[0] and current[1] == goal[1]

def is_cell_empty(value):
    if value is None or len(value) == 0:
        return True
    for v in value:
        if v != EMPTY:
            return False
    return True

def get_fill_color(value):
    if value is None or len(value) == 0:
        return 'white'
    cell_type = value[0]
    for v in value:
        if isinstance(v, Mover):
            cell_type = v.cell_type
            break
    if cell_type == EMPTY:
        return 'white'
    if cell_type == WALL:
        return 'black'
    if cell_type == JUNK:
        return '#502020'
    if cell_type == GOAL:
        return '#30ff30'
    if cell_type == HERO:
        return '#0020ff'
    if cell_type == ENEMY:
        return '#c01030'
    return '#ff00ff'

def start_time():
    global time_rate
    time_rate = 10
    step_time()

def step_time():
    global _movers, _hero, time_rate
    loop_start = time.perf_counter_ns()
    perf1 = loop_start
    if time_rate > 0:
        # all movers move simultaneously, so plan -> move -> collide
        #next_cells = obs.get_cells_copy()
        changes = []
        for mover in _movers:
            mover.update_plan(cells=obs._cells, hero=_hero, movers=_movers, goals=_goals)
            #next_cells[mover.y][mover.x].remove(mover)
            changes.append(functools.partial(lambda x, y, m: obs.remove_from_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Planning: {perf2 - perf1}')
            perf1 = perf2

        next_movers = []
        for mover in _movers:
            mover.move()
            #next_cells[mover.y][mover.x].append(mover)
            changes.append(functools.partial(lambda x, y, m: obs.add_to_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
            next_movers.append(mover)
        _movers = next_movers

        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Moving: {perf2 - perf1}')
            perf1 = perf2

        for change in changes:
            change()

        for mover in _movers:
            if mover.next_move != (0, 0):
                result = mover.collide(obs._cells[mover.y][mover.x])
                if result == GAME_OVER:
                    stop_time()
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Collisions: {perf2 - perf1}')
            perf1 = perf2

        obs.draw_grid(canvas)
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Drawing: {perf2 - perf1}')
            perf1 = perf2

        if time_rate > 0:
            if __debug__:
                print(f'Max loop rate: {1000000/(time.perf_counter_ns() - loop_start)}Hz')
            root_window.after(int(1000/time_rate - (time.perf_counter_ns() - loop_start)/1000000), step_time)

def stop_time():
    global time_rate
    time_rate = 0

def spawn_units():
    global _hero, _movers, _goals
    open_spaces = [(x, y) for y in range(len(obs._cells)) for x in range(len(obs._cells[y])) if len(obs._cells[y][x]) == 0 or obs._cells[y][x][0] == EMPTY]
    for i in range(ENEMY_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        _movers.append(Enemy(x, y))
        obs.add_to_cell(x, y, _movers[i])
        print(f'Placed enemy at ({x, y})')
    for i in range(GOAL_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        _goals.append((x, y))
        obs.add_to_cell(x, y, GOAL)
        print(f'Placed goal at ({x, y})')

    x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
    _hero = Hero(x, y)
    _movers.append(_hero)
    obs.add_to_cell(x, y, _hero)
    print(f'Placed hero at ({x, y})')

if __name__ == "__main__":
    # create grid
    obs.FIELD_HEIGHT = 64
    obs.FIELD_WIDTH = 64
    obs._cell_height = 10
    obs._cell_width = 10
    obs._fill_func = get_fill_color
    root_window, frame, canvas = obs.create_window()
    root_window.title("Flatland")
    obs.draw_grid(canvas)
    obs.populate_cells(0.2, True, canvas)

    # create units
    spawn_units()
    obs.draw_grid(canvas)

    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0)
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=2, column=0)

    frame.pack()
    root_window.mainloop()