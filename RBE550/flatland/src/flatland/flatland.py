from turtles import obstacle_field as obs
import numpy as np
import tkinter as tk
from tkinter import ttk as ttk
import functools
import time
from astar import AStar
import fastquadtree

# game config
GRID_SIZE = 64
ENEMY_COUNT = 10
GOAL_COUNT = 1 # TODO: having multiple goals isn't supported yet
RUNNING_TIME_RATE = 1
COVERAGE = 0.2

# These constants affect how brave the hero is. Danger is a multiplier on cell cost of the form:
# max(boldness, cos(theta)*(range^strength/distance_to_enemy^strength))
# This effectively creates a prohibitively expensive area within a radius specified by DANGER_RANGE around each enemy.
# DANGER_STRENGTH and DANGER_CHECKS affect how enemies' "danger zones" overlap: higher strength makes enemies closer to
# the cell more important relative to enemies further away, and more checks means more enemies' zones are taken into
# account.
# (Note: the hero is also included in the nearest-neighbor checks and the spaces closest to the hero are the ones where
# the danger calculation is most relevant, so DANGER_CHECKS should always be at least 2 to ensure that the hero isn't
# the only Mover that gets checked. The hero doesn't think it's a danger to itself.)
DANGER_RANGE = 3
DANGER_STRENGTH = 4
DANGER_CHECKS = 3
BOLDNESS = 0.9
TELEPORTS = 0

# cell states
_EMPTY = 0
_WALL = 1
_JUNK = 2
_GOAL = 3
_HERO = 4
_ENEMY = 5

# game results
_GAME_OVER = -1

# UI-available options
_time_rate = RUNNING_TIME_RATE # target number of sim loops per second
_should_draw_path_bg = False
_should_draw_cost_bg = False
_should_draw_heur_bg = False

def should_draw_bg():
    return _should_draw_path_bg.get() or _should_draw_cost_bg.get() or _should_draw_heur_bg.get()

_rng = np.random.default_rng()

class Mover:
    """
    Mover is the base class for entities that inhabit cells and can move between them.
    """
    def __init__(self, x, y, cell_type):
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.next_move = (0, 0)

    def move(self, strict = True):
        '''
        Update the mover's knowledge of its position based on its next_move. Note that this doesn't update the underlying
        grid. Called after update_plan() in the game loop.
        :param strict: If True, only applies movement if next_move has a length of less than 1.
        :return: True if move was successful, False otherwise.
        '''
        (dx, dy) = self.next_move
        if (abs(dx) <= 1 and abs(dy) == 0) or (abs(dx) == 0 and abs(dy) <= 1) or not strict:
            self.x += dx
            self.y += dy
            return True
        else:
            print(f'Illegal move by {self}')
            return False

    def collide(self, with_list):
        """
        Handle collisions with other entities in the same cell. Called after move() in the game loop.
        :param with_list: List of entities in the grid cell.
        :return: _GAME_OVER if the hero dies or reaches the goal, else None.
        """
        return # no-op for base class

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        """
        Sets self.next_move to a 2-tuple representing the number of cells to move on each axis the next time move() is
        called.
        :param cells: The current grid to use for planning. Do not write to this.
        :param hero: The Hero object.
        :param movers: A quad-tree containing all Movers.
        :param goals: A list of 2-tuples containing goal coordinates.
        :return: None (implementation is expected to write value to self.next_move)
        """
        return # no-op for base class

    def __str__(self):
        return f'({int(self.x)}, {int(self.y)}): {self.__class__.__name__} (type {self.cell_type})'

    def __repr__(self):
        return self.__str__()

class Enemy(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, _ENEMY)

    def move(self, strict = True):
        super().move()

    def collide(self, with_list):
        if self.cell_type == _ENEMY: # dead enemies don't need to do anything
            for other in with_list:
                if other == self:
                    continue
                with_type = other.cell_type if isinstance(other, Mover) else other
                if with_type == _HERO:
                    print(f'Enemy killed hero at ({self.x}, {self.y})!')
                    return _GAME_OVER
                elif with_type == _WALL or with_type == _JUNK or with_type == _ENEMY:
                    print(f'Enemy junked at ({self.x}, {self.y})!')
                    self.cell_type = _JUNK
        return None

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        if self.cell_type == _ENEMY:
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
        Mover.__init__(self, x, y, _HERO)
        AStar.__init__(self)
        self.teleports = TELEPORTS
        self.teleporting = False
        self.cells = []
        self.goal = (x, y)
        self.path_bg = obs.AdaptiveGradientBackground()

    def move(self, strict = True):
        super().move(strict = not self.teleporting)
        if self.teleporting:
            self.teleports -= 1

    def collide(self, with_list):
        for other in with_list:
            if other == self:
                continue
            with_type = other.cell_type if isinstance(other, Mover) else other
            if with_type == _EMPTY:
                continue
            if with_type == _HERO:
                print(f'Hero collided with self? ({self.x}, {self.y})')
                return _GAME_OVER
            if with_type == _GOAL:
                print(f'Hero reached goal at ({self.x}, {self.y})!')
                return _GAME_OVER
            else:
                print(f'Hero hit obstacle at ({self.x}, {self.y})!')
                return _GAME_OVER
        return None

    def update_plan(self, cells = None, hero = None, movers = None, goals = None):
        """
        The Hero uses this strategy to plan moves:
        1. Teleport to a random space if enemies are within 1 space on both axes and any teleport charges remain.
        2. Use A* with customized costs and heuristics (see below) to calculate a path to the goal.
        3. If a path to the goal exists, move to the next space on the path.
        4. If no path is found and teleport charges remain, teleport to a random space.
        5. Do nothing.
        """
        self.next_move = (0, 0)
        self.teleporting = False
        if cells is not None:
            self.cells = cells
        for item in movers:
            mover = item.obj
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
                if _should_draw_path_bg.get():
                    for i in range(1, len(path)):
                        self.path_bg.cell_map[path[i]] = -10 # force the path to be the greenest thing
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
        """
        Implements AStar.neighbors. Cells are considered neighbors if they are 1 distance away and don't contain anything
        but the hero or the goal cell.
        :param node: (x, y) coordinates of the node to get neighbors for.
        :return: The valid neighbors of node.
        """
        x, y = node
        if 0 <= y < len(self.cells) and 0 <= x < len(self.cells[y]):
            return [(x + i[0], y + i[1])
                    for i in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= y + i[1] < len(self.cells) and 0 <= x + i[0] < len(self.cells[y])
                    and is_cell_empty(self.cells[y + i[1]][x + i[0]]) or (x + i[0] == self.x and y + i[1] == self.y)
                    or (x + i[0] == self.goal[0] and y + i[1] == self.goal[1])]
        return []

    def heuristic_cost_estimate(self, current, goal) -> float:
        """
        Implements AStar.heuristic_cost_estimate. Returns the Euclidean distance between current and goal multiplied by
        the danger level of the current cell. Dangerous cells should disincentivize expansion.
        :param current: (x, y) coordinates of the current cell.
        :param goal: (x, y) coordinates of the goal cell.
        """
        cx, cy = current
        gx, gy = goal
        if 0 <= cy < len(self.cells) and 0 <= cx < len(self.cells[cy]):
            if not is_cell_empty(self.cells[cy][cx]) and (cx != self.x and cy != self.y) and (cx != gx and cy != gy):
                return float('inf')
            danger = self.calculate_danger(cx, cy)
            result = danger * np.sqrt(np.power(gx - cx, 2) + np.power(gy - cy, 2))
            if _should_draw_heur_bg.get():
                self.path_bg.cell_map[current] = result
            return result
        return float('inf')

    def path_distance_between(self, n1, n2) -> float:
        """
        Implements AStar.path_distance_between. Returns the Manhattan distance between current and goal multiplied by the
        ratio of danger between the two (moving to a more dangerous space is more costly and vice versa).
        :param n1: (x, y) coordinates of the cell to move from.
        :param n2: (x, y) coordinates of the cell to move to.
        """
        danger1 = self.calculate_danger(n1.data[0], n1.data[1])
        danger2 = self.calculate_danger(n2.data[0], n2.data[1])
        result = (danger2/danger1) * abs(n2.data[0] - n1.data[0]) + abs(n2.data[1] - n1.data[1])
        if _should_draw_cost_bg.get():
            self.path_bg.cell_map[n2.data] = result + self.path_bg.cell_map.get(n2.data, 0) if _should_draw_heur_bg else 0
        return result

    def is_goal_reached(self, current, goal) -> bool:
        """
        Implements AStar.is_goal_reached. Returns True if the current cell is the goal, False otherwise.
        :param current: (x, y) coordinates of the current cell.
        :param goal: (x, y) coordinates of the goal cell.
        """
        return current[0] == goal[0] and current[1] == goal[1]

    def calculate_danger(self, x, y):
        """
        Calculates the danger level of another cell based on its relationship between the Hero's current location and
        those of nearby enemies.
        :param x: x coordinate of the cell to calculate danger for.
        :param y: y coordinate of the cell to calculate danger for.
        :return: A float on [1, inf) representing how dangerous the cell at (x, y) is.
        """
        danger = 1  # multiplier to cost based on enemy distance
        nearest = _movers.nearest_neighbors((x, y), DANGER_CHECKS)
        if nearest is not None:
            for n in nearest:
                if n.obj.cell_type == _ENEMY:
                    if n.x == x and n.y == y:
                        return np.power(DANGER_RANGE, DANGER_STRENGTH) # skip the math, the enemy cell is very dangerous
                    # Use the cosine of the angle between the hero-enemy and hero-point lines to make points more scary
                    # if they lead into an enemy and less scary if they lead away.
                    hero_point_vec = np.array((x - self.x, y - self.y))
                    hero_enemy_vec = np.array((n.obj.x - self.x, n.obj.y - self.y))
                    angle_scale = np.divide(np.dot(hero_enemy_vec, hero_point_vec),
                                            (np.linalg.norm(hero_enemy_vec) * np.linalg.norm(hero_point_vec))) \
                        if not np.array_equal(hero_point_vec, (0,0)) else 1
                    # The base of the exponential represents how many cells away danger applies, while the exponent
                    # represents the strength (3^4 seems to invoke a healthy level of fear in our hero).
                    danger *= max(BOLDNESS, angle_scale * np.power(DANGER_RANGE, DANGER_STRENGTH) / np.power(abs(n.x - x) + abs(n.y - y),DANGER_STRENGTH))
        return danger

_movers: fastquadtree.QuadTreeObjects|None = None # quad-trees are fast for finding nearest neighbors
_hero: Hero|None = None
_goals = []

def is_cell_empty(value):
    """
    Checks if a cell's contents should be considered empty for path-planning purposes.
    :param value: A list of the cell's contents.
    :return: True if the cell is empty-like, False otherwise.
    """
    if value is None or len(value) == 0:
        return True
    for v in value:
        if v != _EMPTY:
            return False
    return True

def get_fill_color(value):
    """
    Returns a Tk color string (color name or hex RGB code) to render a cell's contents as on the grid.
    :param value: A list of the cell's contents.
    """
    if value is None or len(value) == 0:
        return '#ffffff'
    cell_type = value[0]
    for v in value:
        if isinstance(v, Mover):
            cell_type = v.cell_type
            break
    if cell_type == _EMPTY:
        return '#ffffff'
    if cell_type == _WALL:
        return '#000000'
    if cell_type == _JUNK:
        return '#502020'
    if cell_type == _GOAL:
        return '#30ff30'
    if cell_type == _HERO:
        return '#0020ff'
    if cell_type == _ENEMY:
        return '#c01030'
    return '#ff00ff'

def start_time():
    """
    Starts/resumes the game loop and progresses sim time by one step.
    :return: None
    """
    global _time_rate
    _time_rate = RUNNING_TIME_RATE
    step_time()

def step_time():
    """
    Progress the simulation by one time step. This runs the following process:
    1. Calls update_plan on all movers
    2. Calls move on all movers
    3. Applies changes to the underlying grid
    4. Calls collide on all movers
    5. Redraws the grid
    6. If the simulation is unpaused, queue another step.
    Note that this function spams the console with performance data when running in debug mode (use -O interpreter flag
    to suppress).
    """
    global _movers, _hero, _time_rate
    loop_start = time.perf_counter_ns()
    perf1 = loop_start
    if _time_rate > 0:
        # all movers move simultaneously, so plan -> move -> collide
        changes = []
        if _hero is not None and should_draw_bg():
            _hero.path_bg.cell_map.clear()
            _hero.path_bg.value_cache = None
        for item in _movers:
            mover = item.obj
            mover.update_plan(cells=obs._cells, hero=_hero, movers=_movers, goals=_goals)
            changes.append(functools.partial(lambda x, y, m: obs.remove_from_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Planning: {perf2 - perf1}')
            perf1 = perf2

        for item in _movers:
            mover = item.obj
            mover.move()
            changes.append(functools.partial(lambda x, y, m: obs.add_to_cell(x, y, m), x = mover.x, y = mover.y, m = mover))
            _movers.update_by_object(mover, mover.x, mover.y)

        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Moving: {perf2 - perf1}')
            perf1 = perf2

        for change in changes:
            change()

        for item in _movers:
            mover = item.obj
            if mover.next_move != (0, 0):
                result = mover.collide(obs._cells[mover.y][mover.x])
                # don't color over movers with the path
                if _hero.path_bg.cell_map.get((mover.x, mover.y), None) is not None:
                    _hero.path_bg.cell_map.pop((mover.x, mover.y))
                if result == _GAME_OVER:
                    stop_time()
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Collisions: {perf2 - perf1}')
            perf1 = perf2

        obs.draw_grid(canvas, background=_hero.path_bg if _hero is not None and should_draw_bg() else None)
        if __debug__:
            perf2 = time.perf_counter_ns()
            print(f'Drawing: {perf2 - perf1}')
            perf1 = perf2

        if _time_rate > 0:
            if __debug__:
                print(f'Max loop rate: {1000000/(time.perf_counter_ns() - loop_start)}Hz')
            root_window.after(int(1000 / _time_rate - (time.perf_counter_ns() - loop_start) / 1000000), step_time)

def stop_time():
    """
    Pauses the game loop.
    """
    global _time_rate
    _time_rate = 0

def spawn_units():
    """
    Place the goal, Enemy, and Hero cells in empty spaces on the grid. This function is not idempotent.
    """
    global _hero, _movers, _goals
    _movers = fastquadtree.QuadTreeObjects((0, 0, obs.FIELD_WIDTH, obs.FIELD_HEIGHT), 4, dtype="i32")
    open_spaces = [(x, y) for y in range(len(obs._cells)) for x in range(len(obs._cells[y])) if len(obs._cells[y][x]) == 0 or obs._cells[y][x][0] == _EMPTY]
    for i in range(ENEMY_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        enemy = Enemy(x, y)
        _movers.insert((x, y), obj=enemy)
        obs.add_to_cell(x, y, enemy)
        print(f'Placed {enemy}')
    for i in range(GOAL_COUNT):
        x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
        _goals.append((x, y))
        obs.add_to_cell(x, y, _GOAL)
        print(f'Placed goal at ({x, y})')

    x, y = open_spaces.pop(_rng.integers(0, len(open_spaces), dtype=int))
    _hero = Hero(x, y)
    _movers.insert((x, y), obj=_hero)
    obs.add_to_cell(x, y, _hero)
    print(f'Placed {_hero}')

if __name__ == "__main__":
    # create grid
    obs.FIELD_HEIGHT = GRID_SIZE
    obs.FIELD_WIDTH = GRID_SIZE
    obs._cell_height = 10
    obs._cell_width = 10
    obs._fill_func = get_fill_color
    root_window, frame, canvas = obs.create_window()
    root_window.title("Flatland")
    obs.draw_grid(canvas)
    obs.populate_cells(COVERAGE, True, canvas)

    # create units
    spawn_units()

    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0)
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=2, column=0)

    _should_draw_path_bg = tk.BooleanVar(value=True)
    _should_draw_cost_bg = tk.BooleanVar(value=True)
    _should_draw_heur_bg = tk.BooleanVar(value=True)
    path_cbox = ttk.Checkbutton(frame, text='Draw Path', variable=_should_draw_path_bg)
    path_cbox.grid(row=1, column=3)
    cost_cbox = ttk.Checkbutton(frame, text='Draw Costs', variable=_should_draw_cost_bg)
    cost_cbox.grid(row=2, column=3)
    heur_cbox = ttk.Checkbutton(frame, text='Draw Heuristics', variable=_should_draw_heur_bg)
    heur_cbox.grid(row=3, column=3)

    obs.draw_grid(canvas)
    frame.pack()
    root_window.mainloop()