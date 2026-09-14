import colorsys

import numpy as np
import parse
import tkinter as tk
from tkinter import ttk
import copy
import time


# occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

_rng = np.random.default_rng()
_cells = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
_rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
_bg_rects = []
_dirty_cells = [((i, j), []) for i in range(FIELD_WIDTH) for j in range(FIELD_HEIGHT)]
_fill_func = lambda value: 'black' if value is not None and len(value) > 0 and value[0] > 0 else 'white'

_cell_width = 6
_cell_height = 6
_cell_border = 1

class AdaptiveGradientBackground:
    """
    Background to pass to draw_grid that can scale get_gradient to a range of values. Put cells to be colored in cell_map
    and set value_cache to None when the gradient range needs to be updated.
    """
    def __init__(self):
        self.cell_map = {}
        self.value_cache = None

    def fill_func(self, value):
        if self.value_cache is None:
            self.value_cache = np.sort(np.array(list(self.cell_map.values())))
            self.value_cache = np.histogram_bin_edges(self.value_cache, 10)
        normalized = np.searchsorted(self.value_cache, value)/self.value_cache.size
        return get_hsv_gradient(float(normalized), 1.0)

def get_basic_tetrominoes():
    """
    :return: a list of arrays representing the tetrominoes shown in the assignment
    """
    print("Encoding basic tetromino set...")
    return [np.atleast_2d(np.array([1, 1, 1, 1])).transpose(),
               np.array([[1, 0, 0],
                          [1, 1, 1]]).transpose(), # L (upside down per fig. 2)
               np.array([[1,1,0],
                          [0,1,1]]).transpose(), # S (sideways)
               np.array([[0,1,0],
                          [1,1,1]]).transpose()] # T (sideways)

def get_all_tetrominoes():
    """
    :return: a list of arrays representing all valid tetrominoes, including rotations and mirrors
    """
    print("Enumerating all tetrominoes...")
    tetrominoes = []
    bar = np.atleast_2d(np.array([1, 1, 1, 1]))
    tetrominoes.append(bar)
    tetrominoes.append(bar.T)

    l = np.array([[1, 0, 0],
                 [1, 1, 1]])
    tetrominoes.append(l)
    tetrominoes.append(np.rot90(l))
    tetrominoes.append(np.rot90(l, k=2))
    tetrominoes.append(np.rot90(l, k=3))

    l = l.T # backwards L
    tetrominoes.append(l)
    tetrominoes.append(np.rot90(l))
    tetrominoes.append(np.rot90(l, k=2))
    tetrominoes.append(np.rot90(l, k=3))

    s = np.array([[1,1,0],
                  [0,1,1]])
    tetrominoes.append(s)
    tetrominoes.append(np.rot90(s))

    s = s.T # backwards S
    tetrominoes.append(s)
    tetrominoes.append(np.rot90(s))

    t = np.array([[0,1,0],
                  [1,1,1]])
    tetrominoes.append(t)
    tetrominoes.append(np.rot90(t))
    tetrominoes.append(np.rot90(t, k=2))
    tetrominoes.append(np.rot90(t, k=3))

    tetrominoes.append(np.array([[1,1],
                                 [1,1]]))
    return tetrominoes

def populate_cells(coverage: float, use_all_tets: bool = False, canvas = None):
    """
    :param coverage: approximate percentage of cells that should be occupied, on [0, 1]
    (tetrominoes may overlap so coverage is an upper bound)
    :param use_all_tets: if true, use all possible tetrominoes instead of just the ones on the assignment doc
    :param canvas: the canvas to draw on
    :return: None
    """
    global _cells, _dirty_cells, _rects
    _cells = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
    _rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH))

    tetrominoes = get_all_tetrominoes() if use_all_tets else get_basic_tetrominoes()
    total = int(coverage * FIELD_HEIGHT * FIELD_WIDTH / 4)
    print(f'Placing {total} tetrominoes...')
    tets_to_place = _rng.integers(len(tetrominoes), size=total)
    for tet in tets_to_place:
        tet_cells = tetrominoes[tet]
        tet_bounds = np.shape(tet_cells)
        x = _rng.integers(0, FIELD_WIDTH - tet_bounds[1], endpoint=True)
        y = _rng.integers(0, FIELD_HEIGHT - tet_bounds[0], endpoint=True)
        for i in np.ndindex(tet_bounds):
            cell = _cells[y + i[0]][x + i[1]]
            if tet_cells[i] == 1:
                if len(cell) > 0:
                    _cells[y + i[0]][x + i[1]][0] = tet_cells[i]
                else:
                    _cells[y + i[0]][x + i[1]].append(tet_cells[i])

    _dirty_cells = [((j, i), _cells[i][j]) for i in range(FIELD_WIDTH) for j in range(FIELD_HEIGHT)]
    if canvas is not None:
        draw_grid(canvas)

__bounds_cache = {}
def get_cell_bounds(x, y):
    """
     returns cell bounds in pixels exclusive of borders
    """
    global __bounds_cache
    if __bounds_cache.get((x, y), None) is None:
        __bounds_cache[(x, y)] = np.array([[x * _cell_width + (x) * _cell_border, y * _cell_height + (y) * _cell_border],
                                           [(x + 1) * _cell_width + (x) * _cell_border, (y + 1) * _cell_height + (y) * _cell_border]]) + 2*_cell_border
    return __bounds_cache[(x, y)]

def get_gradient(v):
    """
    :param v: percentage of red-yellow-green gradient to interpolate to, on [0, 1]
    :return: a string containing a hex representation of an RBG color
    """
    return f'#{int(min(255, 512 * v)):02x}{int(min(255, 512 - 512 * v)):02x}00'

def get_hsv_gradient(hue_percent, saturation = 1.0, value = 1.0):
    rgb = colorsys.hsv_to_rgb((1 - hue_percent)*(120/360), saturation, value)
    return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

def convert_text_to_rgb(text: str):
    full = parse.parse("#{:0x}", text)[0]
    return full >> 16 & 0xff, full >> 8 & 0xff, full & 0xff

def get_threshold(v, threshold):
    """
    :param v: any number
    :param threshold: black/white transition point
    :return: white (as hex RGB in string) if v > threshold, else black
    """
    return "#000000" if v > threshold else "#ffffff"

def draw_grid(canvas: tk.Canvas, background: AdaptiveGradientBackground|None = None):
    global _dirty_cells, _rects
    for (coords, values) in _dirty_cells:
        perf = time.perf_counter_ns()
        x, y = coords
        cell_bounds = get_cell_bounds(x, y)
        perf2 = time.perf_counter_ns()
        #print(f'bounds calc: {perf2 - perf}')
        #print(f'({x},{y}) => {cell_bounds}')
        if _rects[y][x] == 0:
            _rects[y][x] = canvas.create_rectangle(cell_bounds[0][0], cell_bounds[0][1], cell_bounds[1][0], cell_bounds[1][1], fill=_fill_func(values), outline="gray", width=_cell_border)
        else:
            canvas.itemconfigure(int(_rects[y][x]), fill=_fill_func(values))
        perf3 = time.perf_counter_ns()
        #print(f'rect drawing: {perf3 - perf2}')
    _dirty_cells.clear()

    if background is not None:
        canvas.delete(*_bg_rects)
        _bg_rects.clear()
        for (coords, value) in background.cell_map.items():
            x, y = coords
            cell_bounds = get_cell_bounds(x, y)

            current_color_rgb = convert_text_to_rgb(_fill_func(_cells[y][x]))
            bg_color_rgb = convert_text_to_rgb(background.fill_func(value))
            adjusted_color_rgb = np.add(0.75*np.array([*current_color_rgb]), 0.25*np.array([*bg_color_rgb]))
            _bg_rects.append(canvas.create_rectangle(cell_bounds[0][0], cell_bounds[0][1], cell_bounds[1][0], cell_bounds[1][1],
                                    fill=f'#{int(adjusted_color_rgb[0]):02x}{int(adjusted_color_rgb[1]):02x}{int(adjusted_color_rgb[2]):02x}',
                                    outline="gray", width=_cell_border))
    #print(f'dirty cells clear: {time.perf_counter_ns() - perf3}')

def add_to_cell(x, y, value):
    global _cells, _dirty_cells
    _cells[y][x].append(value)
    _dirty_cells.append(((x, y), list(_cells[y][x])))

def remove_from_cell(x, y, value):
    global _cells, _dirty_cells
    _cells[y][x].remove(value)
    _dirty_cells.append(((x, y), list(_cells[y][x])))

def get_cells_copy():
    return [[_cells[y][x].copy() for x in range(len(_cells[y]))] for y in range(len(_cells))]

def create_window():
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    canvas = tk.Canvas(frame, width=canvas_size[0], height=canvas_size[1], bg="gray")
    canvas.grid(columnspan=4)
    return root_window, frame, canvas

if __name__ == "__main__":
    _root_window, _frame, _canvas = create_window()
    draw_grid(_canvas)
    populate_cells(0.7, True, _canvas)
    _frame.pack()
    _root_window.mainloop()
