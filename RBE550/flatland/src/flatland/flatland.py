from turtles import obstacle_field as obs
import numpy as np
import tkinter as tk
from tkinter import ttk as ttk
import functools
import time

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

movers = []
hero = None

time_rate = 1

_rng = np.random.default_rng()

class Mover:
    def __init__(self, x, y, cell_type):
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.next_move = (0, 0)

    def move(self):
        (dx, dy) = self.next_move
        if (abs(dx) == 1 and abs(dy) == 0) or (abs(dx) == 0 and abs(dy) == 1):
            self.x += dx
            self.y += dy
        else:
            print(f'Illegal move by type {self.cell_type} at ({self.x}, {self.y}): ({dx}, {dy})')

    def collide(self, with_list):
        return # no-op for base class

    def update_plan(self, cells = None, hero = None, movers = None):
        return # no-op for base class

    def __str__(self):
        return f'({int(self.x)}, {int(self.y)}): type {self.cell_type}'

    def __repr__(self):
        return self.__str__()

class Enemy(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY)

    def move(self):
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

    def update_plan(self, cells = None, hero = None, movers = None):
        if self.cell_type == ENEMY:
            dx = hero.x - self.x
            dy = hero.y - self.y
            if abs(dx) > abs(dy):
                self.next_move = (1 if dx > 0 else -1, 0)
            else:
                self.next_move = (0, 1 if dy > 0 else -1)
        else:
            self.next_move = (0, 0)

class Hero(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, HERO)
        self.teleports = 5

    def move(self):
        if self.next_move == (0, 0):
            return
        if abs(self.next_move[0]) <= 1 and abs(self.next_move[1]) <= 1:
            super().move()
            return
        if self.teleports > 0:
            self.teleports -= 1
            self.x += self.next_move[0]
            self.y += self.next_move[1]
            print(f'Hero teleported to ({self.x}, {self.y}), {self.teleports} remaining')
        else:
            self.next_move = (0,0)

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

    def update_plan(self, cells = None, hero = None, movers = None):
        self.next_move = (0, 0)
        for mover in movers:
            if mover == self:
                continue
            if abs(mover.x - self.x) <= 1 and abs(mover.y - self.y) <= 1:
                self.next_move = (_rng.integers(-self.x,  -self.x + obs.FIELD_WIDTH), _rng.integers(-self.y, -self.y + obs.FIELD_HEIGHT))

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
    global movers, hero, time_rate
    loop_start = time.perf_counter_ns()
    perf1 = loop_start
    if time_rate > 0:
        # all movers move simultaneously, so plan -> move -> collide
        # can't commit to main grid until everything is done
        #next_cells = obs.get_cells_copy()
        changes = []
        for mover in movers:
            mover.update_plan(cells=obs._cells, hero=hero, movers=movers)
            #next_cells[mover.y][mover.x].remove(mover)
            changes.append(functools.partial(lambda x, y, m: obs.remove_from_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Planning: {perf2 - perf1}')
            perf1 = perf2

        next_movers = []
        for mover in movers:
            mover.move()
            #next_cells[mover.y][mover.x].append(mover)
            changes.append(functools.partial(lambda x, y, m: obs.add_to_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
            next_movers.append(mover)
        movers = next_movers

        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Moving: {perf2 - perf1}')
            perf1 = perf2

        for change in changes:
            change()

        for mover in movers:
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
    global hero, movers
    open_spaces = [(x, y) for y in range(len(obs._cells)) for x in range(len(obs._cells[y])) if len(obs._cells[y][x]) == 0 or obs._cells[y][x][0] == EMPTY]
    for i in range(ENEMY_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        movers.append(Enemy(x, y))
        obs.add_to_cell(x, y, movers[i])
        print(f'Placed enemy at ({x, y})')
    for i in range(GOAL_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        obs.add_to_cell(x, y, GOAL)
        print(f'Placed goal at ({x, y})')

    x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
    hero = Hero(x, y)
    movers.append(hero)
    obs.add_to_cell(x, y, hero)
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
    obs.populate_cells(0.0, True, canvas)

    # create units
    spawn_units()
    obs.draw_grid(canvas)

    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0)
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=2, column=0)

    frame.pack()
    root_window.mainloop()