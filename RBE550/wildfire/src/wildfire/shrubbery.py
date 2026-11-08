import functools

import fastquadtree as qt
import numpy as np

import valet.grid_sim as ui
import valet.obstacle_field as obs

OBS_INTACT = 1
OBS_BURNING = 2
OBS_EXT = 3
OBS_BURNED = 4

EXT_RANGE = ui.meters_to_cells(10)

EXT_COLOR_BURNING = (1.0, 0.55, 0.0, 0.1)   # Orange with low alpha
EXT_COLOR_GOAL = (1.0, 0.95, 0.1, 0.1)       # Yellow with low alpha
EXT_COLOR_EXT = (0.35, 0.75, 1.0, 0.1)      # Light blue with low alpha

# Obstacle state colors
COLOR_OBS_INTACT = (0.0, 0.392, 0.0, 1.0)     # Dark green
COLOR_OBS_EXT = (0.55, 0.27, 0.07, 1.0)       # Brown (saddlebrown)
COLOR_OBS_BURNED = (0.0, 0.0, 0.0, 1.0)       # Black


def get_burn_color(time_remaining: float) -> tuple[float, float, float, float]:
    """
    Computes a gradient color between dark gray (time near 0) and orange (time near 10).
    Higher times are more orange, lower times are more gray.
    """
    t = max(0.0, min(1.0, float(time_remaining) / 10.0))
    # Interpolate from dark gray (0.2, 0.2, 0.2) to orange (1.0, 0.55, 0.0)
    r = 0.2 + t * (1.0 - 0.2)
    g = 0.2 + t * (0.55 - 0.2)
    b = 0.2 + t * (0.0 - 0.2)
    return (r, g, b, 1.0)

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
_obstacle_to_ext_region = {}
_current_goal = None
_obstacle_to_rects: dict[tuple[int, int], list[obs.DrawableRect]] = {}
_cell_to_rect: dict[tuple[int, int], obs.DrawableRect] = {}


class ExtRegion(obs.DrawableGroup):
    def __init__(self, obs_origin):
        super().__init__(visible=False)
        self.obs_origin = obs_origin
        self.verts = set()
        for tup in _obstacle_to_qttups[self.obs_origin]:
            for i in [(0, 0), (2, 0), (0, 2), (2, 2)]:
                self.verts.add((tup[1 + i[0]], tup[2 + i[1]]))

        self.circles: list[obs.DrawableCircle] = []
        for v in self.verts:
            circle = obs.DrawableCircle(
                x=v[0],
                y=v[1],
                radius=EXT_RANGE,
                fill_color=EXT_COLOR_BURNING,
                border_color=None,
                visible=True,
                parent=self,
            )
            self.circles.append(circle)
            self.add(circle)

        self._fill_color = EXT_COLOR_BURNING
        self._cached_combined_tri_verts: np.ndarray | None = None
        self.coordinate_frame = "grid"
        self.border_color = None
        self.border_width = 0.0

    @property
    def fill_color(self):
        return self._fill_color

    @fill_color.setter
    def fill_color(self, val):
        self._fill_color = val
        for c in self.circles:
            c.fill_color = val

    def set_color(self, color):
        self._fill_color = color
        for circle in self.circles:
            circle.fill_color = color

    def get_tri_vertices(self) -> np.ndarray:
        if self._cached_combined_tri_verts is None:
            verts_list = [c.get_tri_vertices() for c in self.circles]
            if verts_list:
                self._cached_combined_tri_verts = np.ascontiguousarray(
                    np.concatenate(verts_list), dtype=np.float32
                )
            else:
                self._cached_combined_tri_verts = np.empty((0, 2), dtype=np.float32)
        return self._cached_combined_tri_verts

    def update_visualization(self, is_goal: bool = False):
        state = _obstacle_to_state.get(self.obs_origin, 0)
        if is_goal and state == OBS_BURNING:
            self.visible = True
            self.set_color(EXT_COLOR_GOAL)
        elif state == OBS_BURNING:
            self.visible = True
            self.set_color(EXT_COLOR_BURNING)
        elif state == OBS_EXT:
            self.visible = True
            self.set_color(EXT_COLOR_EXT)
        else:
            self.visible = False

    @functools.lru_cache(maxsize=250)
    def contains(self, x, y):
        for v in self.verts:
            if (v[0] - x) ** 2 + (v[1] - y) ** 2 < EXT_RANGE ** 2:
                return True
        return False

    @functools.lru_cache(maxsize=250)
    def get_minimum_distance(self, x, y):
        minimum = np.inf
        min_vert= None
        for v in self.verts:
            check = (v[0] - x) ** 2 + (v[1] - y) ** 2
            if check < minimum:
                minimum = check
                min_vert = v
        return np.sqrt(minimum), min_vert


def set_current_goal(goal: ExtRegion | None):
    global _current_goal
    _current_goal = goal
    update_ext_regions()


def get_current_goal() -> ExtRegion | None:
    return _current_goal


def update_ext_regions():
    for origin, region in _obstacle_to_ext_region.items():
        is_goal = (
            _current_goal is not None
            and (_current_goal is region or getattr(_current_goal, 'obs_origin', None) == origin)
        )
        region.update_visualization(is_goal)


def resolve_obstacle_origin(x: int, y: int) -> tuple[int, int] | None:
    """
    Resolves coordinates (x, y) to its obstacle origin tuple.
    Handles both direct obstacle origin coordinates and cell coordinates within an obstacle,
    supporting tetrominoes whose bounding box origin (0, 0) is empty (e.g. T-blocks, L-blocks).
    """
    coord = (x, y)
    if coord in _obstacle_to_cells:
        return coord
    if coord in _cell_to_obstacle:
        return _cell_to_obstacle[coord]
    return None


def get_all_ext_regions() -> list[ExtRegion]:
    return list(_obstacle_to_ext_region.values())


def get_all_obstacle_rects() -> list[obs.DrawableRect]:
    """Returns all DrawableRect instances representing obstacle cells."""
    rects = []
    for r_list in _obstacle_to_rects.values():
        rects.extend(r_list)
    return rects


def get_obstacle_rects(x: int, y: int) -> list[obs.DrawableRect]:
    """Returns the list of DrawableRect instances for the obstacle at origin (x, y)."""
    origin = resolve_obstacle_origin(x, y)
    if origin is not None:
        return _obstacle_to_rects.get(origin, [])
    return []


def qt_rect(coord):
    return (coord[0], coord[1], coord[0] + 1, coord[1] + 1)


def init(field_width, field_height, cell_to_obstacle, obstacle_to_cells):
    global EXT_RANGE, _cell_to_obstacle, _obstacle_to_cells, _obstacle_to_rects, _cell_to_rect
    _obstacle_to_rects.clear()
    _cell_to_rect.clear()
    EXT_RANGE = ui.meters_to_cells(10)
    print('Creating shrubbery quadtrees...')
    _state_qts[OBS_INTACT] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width*field_height*0.1))
    _state_qts[OBS_BURNING] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _state_qts[OBS_EXT] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _state_qts[OBS_BURNED] = qt.RectQuadTree((0, 0, field_width, field_height),
                                                  int(field_width * field_height * 0.1))
    _cell_to_obstacle = dict(cell_to_obstacle) if cell_to_obstacle else {}
    _obstacle_to_cells = obstacle_to_cells
    for o in _obstacle_to_cells.keys():
        _cell_to_obstacle[o] = o
    print('Populating shrubbery quadtrees...')
    for o in _obstacle_to_cells.keys():
        create_state(o[0], o[1])


def create_state(x, y):
    global _state_qts
    _obstacle_to_state[(x, y)] = OBS_INTACT
    _obstacle_to_qttups[(x, y)] = []

    # NOTE on Grid Texturing Alternative:
    # Instead of mapping DrawableRects per cell in Pass 2, an alternative approach is to
    # update cell colors directly on the Pass 1 grid texture (via obstacle_field._grid_colors).
    # In that approach, _cells must NOT be modified to store non-numeric tuples (which would
    # break c[0] > 0 checks in valet and flatland); rather, shrubbery would map its obstacle
    # states/timers directly into _grid_colors[cy, cx], keeping _cells as pure boolean occupancy.
    # We use DrawableRects here as it cleanly encapsulates obstacle cell visualization inside
    # shrubbery without mutating obstacle_field internals or duplicating state across subsystems.
    _obstacle_to_rects[(x, y)] = []
    for coord in _obstacle_to_cells[(x, y)]:
        rect = qt_rect(coord)
        qttup = (_state_qts[OBS_INTACT].insert(qt_rect(coord)), *rect)
        _obstacle_to_qttups[(x, y)].append(qttup)
        cell_rect = obs.DrawableRect(
            x=coord[0],
            y=coord[1],
            width=1.0,
            height=1.0,
            origin=(0.0, 0.0),
            fill_color=COLOR_OBS_INTACT,
            border_color=None,
            coordinate_frame="grid",
            visible=True,
        )
        _obstacle_to_rects[(x, y)].append(cell_rect)
        _cell_to_rect[coord] = cell_rect

    region = ExtRegion((x, y))
    _obstacle_to_ext_region[(x, y)] = region
    region.update_visualization(False)


def ignite(x, y):
    obs_origin = resolve_obstacle_origin(x, y)
    if obs_origin is not None:
        state = _obstacle_to_state.get(obs_origin, 0)
        if state == OBS_INTACT or state == OBS_EXT:
            print(f'Obstacle at {obs_origin} ignited from {x, y}')
            for tup in _obstacle_to_qttups[obs_origin]:
                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_BURNING].insert(tup[1:], tup[0])
            _burn_timers[obs_origin] = 10.0
            _obstacle_to_state[obs_origin] = OBS_BURNING
            burn_col = get_burn_color(10.0)
            for r in _obstacle_to_rects.get(obs_origin, ()):
                r.fill_color = burn_col
            update_ext_regions()


def extinguish(x, y):
    obs_origin = resolve_obstacle_origin(x, y)
    if obs_origin is not None:
        state = _obstacle_to_state.get(obs_origin, 0)
        if state == OBS_BURNING:
            print(f'Obstacle at {obs_origin} extinguished from {x, y}')
            for tup in _obstacle_to_qttups[obs_origin]:
                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_EXT].insert(tup[1:], tup[0])
            _burn_timers.pop(obs_origin, None)
            _obstacle_to_state[obs_origin] = OBS_EXT
            for r in _obstacle_to_rects.get(obs_origin, ()):
                r.fill_color = COLOR_OBS_EXT
            update_ext_regions()


def finish_burning(x, y):
    obs_origin = resolve_obstacle_origin(x, y)
    if obs_origin is not None:
        state = _obstacle_to_state.get(obs_origin, 0)
        timer = _burn_timers.get(obs_origin, np.inf)
        if state == OBS_BURNING and timer <= 0:
            print(f'Obstacle at {obs_origin} finished burning; igniting neighbors')
            for tup in _obstacle_to_qttups[obs_origin]:
                center = ((tup[1] + tup[3])/2, (tup[2] + tup[4])/2)
                range_sq = ui.meters_to_cells(30) ** 2
                intacts = _state_qts[OBS_INTACT].nearest_neighbors(center, int(range_sq))
                for rect in intacts:
                    rect_center = ((rect[1] + rect[3])/2, (rect[2] + rect[4])/2)
                    if abs(rect_center[0] - center[0]) ** 2 + abs(rect_center[1] - center[1]) ** 2 <= range_sq:
                        rect_cell = (rect[1], rect[2])
                        rect_origin = resolve_obstacle_origin(*rect_cell)
                        if rect_origin is not None:
                            ignite(rect_origin[0], rect_origin[1])
                    else:
                        break

                exts = _state_qts[OBS_EXT].nearest_neighbors(center, int(range_sq))
                for rect in exts:
                    rect_center = ((rect[1] + rect[3]) / 2, (rect[2] + rect[4]) / 2)
                    if abs(rect_center[0] - center[0]) ** 2 + abs(rect_center[1] - center[1]) ** 2 <= range_sq:
                        rect_cell = (rect[1], rect[2])
                        rect_origin = resolve_obstacle_origin(*rect_cell)
                        if rect_origin is not None:
                            ignite(rect_origin[0], rect_origin[1])
                    else:
                        break

                _state_qts[state].delete_tuple(tup)
                _state_qts[OBS_BURNED].insert(tup[1:], tup[0])
            _burn_timers.pop(obs_origin, None)
            _obstacle_to_state[obs_origin] = OBS_BURNED
            for r in _obstacle_to_rects.get(obs_origin, ()):
                r.fill_color = COLOR_OBS_BURNED
            update_ext_regions()


def nearest_neighbors(x, y, state, k=1):
    if state in _state_qts:
        return _state_qts[state].nearest_neighbors((x, y), k)
    return []


def get_state(x, y):
    origin = resolve_obstacle_origin(x, y)
    if origin is not None and origin in _obstacle_to_state:
        return _obstacle_to_state[origin]
    return 0


def get_burn_timer(x, y):
    origin = resolve_obstacle_origin(x, y)
    if origin is not None:
        if _obstacle_to_state.get(origin) == OBS_BURNING and origin in _burn_timers:
            return _burn_timers[origin]
        elif _obstacle_to_state.get(origin) == OBS_BURNING or origin in _burn_timers:
            print(f'inconsistent state between _obstacle_to_state and _burn_timers for {origin}')
    return np.inf


def get_burn_timers():
    return _burn_timers.copy()


def get_ext_region(x, y):
    origin = resolve_obstacle_origin(x, y)
    if origin is not None and origin in _obstacle_to_ext_region:
        return _obstacle_to_ext_region[origin]
    return None


def step(dt):
    obs_to_check = set(_burn_timers.keys())
    for obs in obs_to_check:
        if _obstacle_to_state[obs] == OBS_BURNING and obs in _burn_timers:
            _burn_timers[obs] -= dt
            if _burn_timers[obs] <= 0:
                finish_burning(*obs)
            else:
                burn_col = get_burn_color(_burn_timers[obs])
                for r in _obstacle_to_rects.get(obs, ()):
                    r.fill_color = burn_col
        else:
            print(f'burning obstacle {obs} is missing timer?')
    update_ext_regions()
