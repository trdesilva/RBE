import numpy as np
import tkinter as tk
from tkinter import ttk
import copy

# occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

_rng = np.random.default_rng()
_cells = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
_rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
_dirty_cells = [((i, j), []) for i in range(FIELD_WIDTH) for j in range(FIELD_HEIGHT)]
_fill_func = lambda value: 'black' if value is not None and len(value) > 0 and value[0] > 0 else 'white'

_cell_width = 6
_cell_height = 6
_cell_border = 1

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

def get_cell_bounds(x, y):
    """
     returns cell bounds in pixels exclusive of borders
    """
    return np.array([[x * _cell_width + (x) * _cell_border, y * _cell_height + (y) * _cell_border],
                     [(x + 1) * _cell_width + (x) * _cell_border, (y + 1) * _cell_height + (y) * _cell_border]]) + 2*_cell_border

def get_gradient(v):
    """
    :param v: percentage of red-yellow-green gradient to interpolate to, on [0, 1]
    :return: a string containing a hex representation of an RBG color
    """
    return f'#{int(min(255, 512 * v)):02x}{int(min(255, 512 - 512 * v)):02x}00'

def get_threshold(v, threshold):
    """
    :param v: any number
    :param threshold: black/white transition point
    :return: white (as hex RGB in string) if v > threshold, else black
    """
    return "#000000" if v > threshold else "#ffffff"

def draw_grid(canvas: tk.Canvas):
    global _dirty_cells, _rects
    for (coords, values) in _dirty_cells:
        x, y = coords
        cell_bounds = get_cell_bounds(x, y)
        #print(f'({x},{y}) => {cell_bounds}')
        if _rects[y][x] == 0:
            _rects[y][x] = canvas.create_rectangle(cell_bounds[0][0], cell_bounds[0][1], cell_bounds[1][0], cell_bounds[1][1], fill=_fill_func(values), outline="gray", width=_cell_border)
        else:
            canvas.itemconfigure(int(_rects[y][x]), fill=_fill_func(values))
    _dirty_cells = []

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
    canvas.grid()
    return root_window, frame, canvas

if __name__ == "__main__":
    _root_window, _frame, _canvas = create_window()
    draw_grid(_canvas)
    populate_cells(0.7, True, _canvas)
    _frame.pack()
    _root_window.mainloop()
