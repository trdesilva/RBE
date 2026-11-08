import fastquadtree as qt
import numpy as np

import valet.grid_sim as ui

OBS_INTACT = 1
OBS_BURNING = 2
OBS_EXT = 3
OBS_BURNED = 4

_state_qts: dict[int, qt.RectQuadTree] = {}
"""
Dictionary mapping from states to RectQuadTrees, which contain the cells of all obstacles with the given state.
"""
_cell_to_obstacle = {}
"""
Dictionary mapping from (x, y) coordinate pairs for obstacle cells to the origin (top-left) coordinate of the obstacle.
"""
_obstacle_to_cells = {}
"""
Dictionary mapping from (x, y) coordinate pairs for obstacle origins to lists of coordinates of individual cells.
"""
_obstacle_to_state = {}
"""
Dictionary mapping from (x, y) coordinate pairs for obstacle origins to burn states.
"""
_obstacle_to_qttups = {}
"""
Dictionary mapping from (x, y) coordinate pairs for obstacle origins to fastquadtree rectangle tuples (id, minx, miny, maxx, maxy).
"""
_burn_timers: dict[tuple[int, int], float] = {}
"""
Dictionary mapping from (x, y) coordinate pairs for obstacle origins to number of seconds left before the burning state
transitions to the burned state.
"""

def qt_rect(coord):
    return (coord[0], coord[1], coord[0] + 1, coord[1] + 1)

def init(field_width, field_height, cell_to_obstacle, obstacle_to_cells):
    global _cell_to_obstacle, _obstacle_to_cells
    print('Creating shrubbery quadtrees...')
    _state_qts[OBS_INTACT] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width*field_height*0.1))
    _state_qts[OBS_BURNING] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _state_qts[OBS_EXT] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _state_qts[OBS_BURNED] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _cell_to_obstacle = cell_to_obstacle
    _obstacle_to_cells = obstacle_to_cells
    print('Populating shrubbery quadtrees...')
    for o in _obstacle_to_cells.keys():
        create_state(o[0], o[1])

def create_state(x, y):
    global _state_qts
    _obstacle_to_state[(x, y)] = OBS_INTACT
    _obstacle_to_qttups[(x, y)] = []
    for coord in _obstacle_to_cells[(x, y)]:
        rect = qt_rect(coord)
        qttup = (_state_qts[OBS_INTACT].insert(qt_rect(coord)), *rect)
        _obstacle_to_qttups[(x, y)].append(qttup)
    
def ignite(x, y):
    state = get_state(x, y)
    if state > 0:
        obs_origin = _cell_to_obstacle[(x, y)]
        if state == OBS_INTACT or state == OBS_EXT:
            print(f'Obstacle at {obs_origin} ignited from cell {x, y}')
            for tup in _obstacle_to_qttups[obs_origin]:
                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_BURNING].insert(tup[1:], tup[0])
            _burn_timers[obs_origin] = 10.0
            _obstacle_to_state[obs_origin] = OBS_BURNING

def extinguish(x, y):
    state = get_state(x, y)
    if state > 0:
        obs_origin = _cell_to_obstacle[(x, y)]
        if state == OBS_BURNING:
            print(f'Obstacle at {obs_origin} extinguished from cell {x, y}')
            for tup in _obstacle_to_qttups[obs_origin]:
                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_EXT].insert(tup[1:], tup[0])
            _burn_timers.pop(obs_origin)
            _obstacle_to_state[obs_origin] = OBS_EXT

def finish_burning(x, y):
    state = get_state(x, y)
    if state > 0:
        obs_origin = _cell_to_obstacle[(x, y)]
        timer = _burn_timers[obs_origin]
        if state == OBS_BURNING and timer <= 0:
            print(f'Obstacle at {obs_origin} finished burning; igniting neighbors')
            for tup in _obstacle_to_qttups[obs_origin]:
                center = ((tup[1] + tup[3])/2, (tup[2] + tup[4])/2)
                range_sq = ui.meters_to_cells(30) ** 2
                intacts = _state_qts[OBS_INTACT].nearest_neighbors(center, int(range_sq))
                for rect in intacts:
                    rect_center = ((rect[1] + rect[3])/2, (rect[2] + rect[4])/2)
                    if abs(rect_center[0] - center[0]) ** 2 + abs(rect_center[1] - center[1]) ** 2 <= range_sq:
                        rect_origin = _cell_to_obstacle[(rect[1], rect[2])]
                        ignite(rect_origin[0], rect_origin[1])
                    else:
                        break

                exts = _state_qts[OBS_EXT].nearest_neighbors(center, int(range_sq))
                for rect in exts:
                    rect_center = ((rect[1] + rect[3]) / 2, (rect[2] + rect[4]) / 2)
                    if abs(rect_center[0] - center[0]) ** 2 + abs(rect_center[1] - center[1]) ** 2 <= range_sq:
                        rect_origin = _cell_to_obstacle[(rect[1], rect[2])]
                        ignite(rect_origin[0], rect_origin[1])
                    else:
                        break

                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_BURNED].insert(tup[1:], tup[0])
            _burn_timers.pop(obs_origin)
            _obstacle_to_state[obs_origin] = OBS_BURNED

def nearest_neighbors(x, y, state, k=1):
    if state in _state_qts:
        return _state_qts[state].nearest_neighbors((x, y), k)
    return []

def get_state(x, y):
    if (x, y) in _cell_to_obstacle:
        return _obstacle_to_state[_cell_to_obstacle[(x, y)]]
    return 0

def get_burn_timer(x, y):
    if (x, y) in _cell_to_obstacle:
        obs = _cell_to_obstacle[(x, y)]
        if _obstacle_to_state[obs] == OBS_BURNING and obs in _burn_timers:
            return _burn_timers[obs]
        elif _obstacle_to_state[obs] == OBS_BURNING or obs in _burn_timers:
            print(f'inconsistent state between _obstacle_to_state and _burn_timers for {obs}')
    return np.inf

def step(dt):
    obs_to_check = set(_burn_timers.keys())
    for obs in obs_to_check:
        if _obstacle_to_state[obs] == OBS_BURNING and obs in _burn_timers:
            _burn_timers[obs] -= dt
            finish_burning(*obs)
        else:
            print(f'burning obstacle {obs} is missing timer?')
