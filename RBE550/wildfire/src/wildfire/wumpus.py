import functools
from dataclasses import dataclass
from typing import dataclass_transform

import fastquadtree as qt
import numpy as np
from astar import AStar, T, SearchNode

import valet.grid_sim as ui
import valet.obstacle_field as obs
import valet.planner as vp
import shrubbery as shr

@dataclass
class AStarData:
    data: tuple[int, int] | list[tuple[int, int]]
    is_goal: bool = False
    def __hash__(self):
        return hash(list(self.data) if self.is_goal else tuple(self.data)) + 101*hash(self.is_goal)

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
        self.truck = None

    def step(self, dt):
        while dt >= self.step_period - self.step_timer:
            dt -= self.step_period - self.step_timer
            self.step_timer = 0
            if len(self.moves) == 0:
                self.ignite_neighbors()
                self.replan()
            if len(self.moves) > 0:
                move = self.moves.pop(0)
                self.x = move[0]
                self.y = move[1]

        self.step_timer += dt

    def replan(self):
        burnables = shr.nearest_neighbors(self.x, self.y, shr.OBS_INTACT, 3)
        burnables.extend(shr.nearest_neighbors(self.x, self.y, shr.OBS_EXT, 3))
        goal = []
        best = None
        for tup in burnables:
            cells = [(self.wumpus_grid_res * cell[0] + i, self.wumpus_grid_res * cell[1] + j)
                     for i in range(self.wumpus_grid_res)
                     for j in range(self.wumpus_grid_res)
                     for cell in shr._obstacle_to_cells[shr.resolve_obstacle_origin(tup[1], tup[2])] ]
            heur = self.heuristic_cost_estimate(AStarData((int(round(self.x * self.wumpus_grid_res)),
                                                           int(round(self.y * self.wumpus_grid_res)))),
                                                AStarData(cells, True))
            if self.truck is not None:
                heur -= 2*self.heuristic_cost_estimate(AStarData((int(round(self.truck.x * self.wumpus_grid_res)),
                                                                  int(round(self.truck.y * self.wumpus_grid_res)))),
                                                       AStarData(cells, True))
            if best is None or heur < best[1]:
                best = (cells, heur)

        if best is not None:
            goal = best[0]
        scaled_start = (int(round(self.x * self.wumpus_grid_res)), int(round(self.y * self.wumpus_grid_res)))
        path = self.astar(AStarData(scaled_start), AStarData(goal if not len(goal) == 0 else [(0, 0)], True))
        if path is not None:
            nodes = [i.data for i in list(path)]
            if len(nodes) > 1 and nodes[0] == scaled_start:
                nodes = nodes[1:]
            elif len(nodes) == 1 and nodes[0] == scaled_start:
                nodes = []
            self.moves = (np.array(nodes)/self.wumpus_grid_res).tolist() if len(nodes) > 0 else []
        else:
            print(f'wumpus failed to path')

    def ignite_neighbors(self):
        wx = int(round(self.x * self.wumpus_grid_res))
        wy = int(round(self.y * self.wumpus_grid_res))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cx = (wx + dx) // self.wumpus_grid_res
            cy = (wy + dy) // self.wumpus_grid_res
            if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
                shr.ignite(cx, cy)

    @functools.cache
    def _check_collision_cached(self, wump_coords):
        cx = wump_coords[0] // self.wumpus_grid_res
        cy = wump_coords[1] // self.wumpus_grid_res
        if cx < 0 or cx >= self.grid_width or cy < 0 or cy >= self.grid_height:
            return True
        return shr.get_state(cx, cy) > 0

    def check_collision(self, wump_coords = None):
        if wump_coords is None:
            wump_coords = (round(self.x * self.wumpus_grid_res), round(self.y * self.wumpus_grid_res))
        return self._check_collision_cached(wump_coords)

    """
    A* Impl
    """
    def neighbors(self, node):
        if node.is_goal:
            return []
        result = []
        for i in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (node.data[0] + i[0], node.data[1] + i[1])
            if self.check_collision(neighbor):
                continue
            result.append(AStarData(neighbor))
        return result

    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        min_heur = np.inf
        if not current.is_goal and goal.is_goal and goal.data is not None:
            c = current.data
            for g in goal.data:
                min_heur = min(abs(g[0] - c[0]) + abs(g[1] - c[1]), min_heur)
        return min_heur

    def distance_between(self, n1: T, n2: T) -> float:
        if n1.is_goal or n2.is_goal:
            return np.inf
        if self.check_collision(n2.data):
            return np.inf
        return abs(n2.data[0] - n1.data[0]) + abs(n2.data[1] - n1.data[1])

    def is_goal_reached(self, current, goal):
        if not current.is_goal and goal.is_goal and goal.data is not None:
            c = current.data
            for g in goal.data:
                if abs(g[0] - c[0]) + abs(g[1] - c[1]) <= 1:
                    return True
        return False
