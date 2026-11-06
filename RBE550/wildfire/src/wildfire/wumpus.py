import fastquadtree as qt
import numpy as np
from astar import AStar, T, SearchNode

import valet.grid_sim as ui
import valet.obstacle_field as obs
import valet.planner as vp

class Wumpus(obs.DrawableGroup, AStar):
    def __init__(self, x: float, y: float, grid_height: int, grid_width: int, obs_quadtree: qt.RectQuadTree, wumpus_grid_res: int = 5, speed: float= 1.0) -> None:
        """
        The Wumpus takes up a cell in a grid with wumpus_grid_res times higher linear resolution than the global grid.
        It plans its movement using A* in this finer grid and maps obstacles from obs_quadtree into it.
        :param x: starting x position in global grid space
        :param y: starting y position in global grid space
        :param grid_height: maximum global grid y dimension
        :param grid_width: maximum global grid x dimension
        :param obs_quadtree: RectQuadTree containing obstacles in global grid space
        :param wumpus_grid_res: multiplier on linear resolution for Wumpus pathing grid relative to global grid. The
        Wumpus's size in global grid units is a 1/wumpus_grid_res square.
        :param speed: number of global grid units the Wumpus can move per simulation second. The Wumpus makes speed *
        wumpus_grid_res moves per simulation second, with each move traveling 1/wumpus_grid_res cells.
        """
        obs.DrawableGroup.__init__(self, x, y)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_quadtree = obs_quadtree
        self.wumpus_grid_res = wumpus_grid_res
        self.speed = speed

        self.body = obs.DrawableRect(0, 0, 1./wumpus_grid_res, 1./wumpus_grid_res, fill_color=(1.0, 0.0, 1.0, 1.0))
        self.add(self.body)

        self.step_timer = 0
        self.moves = []

    def step(self, dt):
        pass

    def replan(self):
        pass

    """
    A* Impl
    """
    def neighbors(self, node):
        pass

    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        pass

    def path_distance_between(self, n1: SearchNode[T], n2: SearchNode[T]) -> float:
        pass

    def is_goal_reached(self, current, goal):
        pass