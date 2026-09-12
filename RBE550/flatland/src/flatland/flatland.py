from turtles import obstacle_field as obs
import numpy as np
import tkinter as tk
from tkinter import ttk as ttk
import functools

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

    def move(self):
        if self.next_move == (0, 0):
            return
        super().move()

    def collide(self, with_list):
        for other in with_list:
            if other == self:
                continue
            with_type = other.cell_type if isinstance(other, Mover) else other
            if with_type == EMPTY:
                return None
            if with_type == HERO:
                print(f'Hero collided with self? ({self.x}, {self.y})')
            if with_type == GOAL:
                print(f'Hero reached goal at ({self.x}, {self.y})!')
            else:
                print(f'Hero hit obstacle at ({self.x}, {self.y})!')
        return GAME_OVER

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
        return '#505050'
    if cell_type == GOAL:
        return '#30ff30'
    if cell_type == HERO:
        return '#0020ff'
    if cell_type == ENEMY:
        return '#901010'
    return '#ff00ff'

def start_time():
    global time_rate
    time_rate = 1
    step_time()

def step_time():
    global movers, hero, time_rate
    if time_rate > 0:
        print(movers)
        # all movers move simultaneously, so plan -> move -> collide
        # can't commit to main grid until everything is done
        next_cells = obs.get_cells_copy()
        changes = []
        for mover in movers:
            mover.update_plan(cells=next_cells, hero=hero, movers=movers)
            next_cells[mover.y][mover.x].remove(mover)
            changes.append(functools.partial(lambda x, y, m: obs.remove_from_cell(x, y, m), x = mover.x, y = mover.y, m = mover))

        next_movers = []
        for mover in movers:
            mover.move()
            next_cells[mover.y][mover.x].append(mover)
            changes.append(functools.partial(lambda x, y, m: obs.add_to_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
            next_movers.append(mover)
        movers = next_movers

        for mover in movers:
            if mover.next_move != (0, 0):
                result = mover.collide(next_cells[mover.y][mover.x])
                if result == GAME_OVER:
                    stop_time()

        for change in changes:
            change()

        obs.draw_grid(canvas)

        if time_rate > 0:
            root_window.after(int(1000/time_rate), step_time)

def stop_time():
    global time_rate
    time_rate = 0

if __name__ == "__main__":
    # create grid
    obs.FIELD_HEIGHT = 64
    obs.FIELD_WIDTH = 64
    obs._cell_height = 10
    obs._cell_width = 10
    obs._fill_func = get_fill_color
    root_window, frame, canvas = obs.create_window()
    obs.draw_grid(canvas)
    obs.populate_cells(0.05, True, canvas)

    # create units
    for i in range(10):
        (_x, _y) = (_rng.integers(0, obs.FIELD_WIDTH), _rng.integers(0, obs.FIELD_HEIGHT))
        movers.append(Enemy(_x, _y))
        obs.add_to_cell(_x, _y, movers[i])
        print(f'Placed enemy at ({int(_x), int(_y)})')
    (_x, _y) = (_rng.integers(0, obs.FIELD_WIDTH), _rng.integers(0, obs.FIELD_HEIGHT))
    hero = Hero(_x, _y)
    movers.append(hero)
    obs.add_to_cell(_x, _y, hero)
    print(f'Placed hero at ({int(_x), int(_y)})')
    obs.draw_grid(canvas)

    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0)
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=1, column=1)

    frame.pack()
    root_window.mainloop()