import functools

import fastquadtree as qt
import numpy as np
from astar import AStar, T, SearchNode

import valet.grid_sim as ui
import valet.obstacle_field as obs
import valet.planner as vp
import shrubbery as shr

class Wumpus(obs.DrawableGroup, AStar):
    def __init__(self, x: float, y: float, grid_width: int, grid_height: int, obs_quadtree: qt.RectQuadTree, wumpus_grid_res: int = 3, speed: float= 1.0) -> None:
        """
        The Wumpus takes up a cell in a grid with wumpus_grid_res times higher linear resolution than the global grid.
        It plans its movement using A* in this finer grid and maps obstacles from obs_quadtree into it.
        :param x: starting x position in global grid space
        :param y: starting y position in global grid space
        :param grid_width: maximum global grid x dimension
        :param grid_height: maximum global grid y dimension
        :param obs_quadtree: RectQuadTree containing obstacles in global grid space
        :param wumpus_grid_res: multiplier on linear resolution for Wumpus pathing grid relative to global grid. The
        Wumpus's size in global grid units is a 1/wumpus_grid_res square.
        :param speed: number of global grid units the Wumpus can move per simulation second. The Wumpus makes speed *
        wumpus_grid_res moves per simulation second, with each move traveling 1/wumpus_grid_res cells.
        """
        obs.DrawableGroup.__init__(self, x, y)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obs_quadtree = obs_quadtree
        self.wumpus_grid_res = wumpus_grid_res
        self.speed = speed

        self.body = obs.DrawableRect(0, 0, 1./wumpus_grid_res, 1./wumpus_grid_res, fill_color=(1.0, 0.0, 1.0, 1.0), origin=(0,0))
        self.add(self.body)

        self.step_timer = 0
        self.step_period = 1./(speed * wumpus_grid_res)
        self.moves = []

    def step(self, dt):
        while dt >= self.step_period - self.step_timer:
            dt -= self.step_period - self.step_timer
            self.step_timer = 0
            self.ignite_neighbors()
            if len(self.moves) == 0:
                self.replan()
            if len(self.moves) > 0:
                move = self.moves.pop(0)
                self.x = move[0]
                self.y = move[1]

        self.ignite_neighbors()
        self.step_timer += dt

    def replan(self):
        scaled_start = int(self.x * self.wumpus_grid_res), int(self.y * self.wumpus_grid_res)
        path = self.astar(scaled_start, (0, 0))
        if path is not None:
            self.moves = (np.array(list(path))/self.wumpus_grid_res).tolist()
        else:
            print(f'wumpus failed to path')

    def ignite_neighbors(self):
        for i in [(-1., 0), (1., 0), (0, -1.), (0, 1.)]:
            x, y = (self.x + i[0]/self.wumpus_grid_res, self.y + i[1]/self.wumpus_grid_res)
            # obstacles only exist within integer cell boundaries
            if (((x % 1 < 1./self.wumpus_grid_res) or (1 - (x % 1) < 1./self.wumpus_grid_res)) and
                ((y % 1 < 1. / self.wumpus_grid_res) or (1 - (y % 1) < 1. / self.wumpus_grid_res))):
                shr.ignite(int(x), int(y))

    @functools.cache
    def _check_collision_cached(self, wump_coords):
        world_coords = (int(float(wump_coords[0])/self.wumpus_grid_res), int(float(wump_coords[1])/self.wumpus_grid_res))
        return shr.get_state(*world_coords) > 0

    def check_collision(self, wump_coords = None):
        if wump_coords is None:
            wump_coords = (round(self.x * self.wumpus_grid_res), round(self.y * self.wumpus_grid_res))
        return self._check_collision_cached(wump_coords)

    """
    A* Impl
    """
    def neighbors(self, node):
        result = []
        for i in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (node[0] + i[0], node[1] + i[1])
            if self.check_collision(neighbor):
                continue
            result.append(neighbor)
        return result

    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        return abs(goal[0] - current[0]) + abs(goal[1] - current[1])

    def path_distance_between(self, n1: SearchNode[T], n2: SearchNode[T]) -> float:
        if self.check_collision(n2.data):
            return np.inf
        return abs(n2.data[0] - n1.data[0]) + abs(n2.data[1] - n1.data[1])

    def is_goal_reached(self, current, goal):
        return goal[0] == current[0] and goal[1] == current[1]