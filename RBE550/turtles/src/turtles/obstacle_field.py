import numpy as np
import tkinter as tk
from tkinter import ttk

# occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

rng = np.random.default_rng()
cells = np.zeros((FIELD_HEIGHT, FIELD_WIDTH))
#print(cells)

cell_width = 6
cell_height = 6
cell_border = 1

def get_basic_tetrominoes():
    """
    :return: a list of arrays representing the tetrominoes shown in the assignment
    """
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

def populate_cells(coverage: float, use_all_tets: bool = False):
    """
    :param coverage: approximate percentage of cells that should be occupied, on [0, 1]
    (tetrominoes may overlap so coverage is an upper bound)
    :param use_all_tets: if true, use all possible tetrominoes instead of just the ones on the assignment doc
    :return: None
    """
    total = int(coverage*FIELD_HEIGHT*FIELD_WIDTH/4)
    tetrominoes = get_all_tetrominoes() if use_all_tets else get_basic_tetrominoes()
    tets_to_place = rng.integers(len(tetrominoes), size=total)
    for tet in tets_to_place:
        tet_cells = tetrominoes[tet]
        #print(tet_cells)
        tet_bounds = np.shape(tet_cells)
        x = rng.integers(0, FIELD_WIDTH - tet_bounds[1])
        y = rng.integers(0, FIELD_HEIGHT - tet_bounds[0])
        for i in np.ndindex(tet_bounds):
            cells[y + i[0], x + i[1]] = tet_cells[i] or cells[y + i[0], x + i[1]]

def get_cell_bounds(x, y):
    """
     returns cell bounds in pixels exclusive of borders
    """
    return np.array([[x*cell_width + (x)*cell_border, y*cell_height + (y)*cell_border],
                     [(x + 1)*cell_width + (x)*cell_border, (y + 1)*cell_height + (y)*cell_border]]) + 2*cell_border

def get_gradient(v):
    return f'#{int(min(255, 512 * v)):02x}{int(min(255, 512 - 512 * v)):02x}00'

def get_threshold(v, threshold):
    return "#000000" if v > threshold else "#ffffff"

def draw_grid(canvas: tk.Canvas):
    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            cell_bounds = get_cell_bounds(x, y)
            #print(f'({x},{y}) => {cell_bounds}')
            canvas.create_rectangle(cell_bounds[0][0], cell_bounds[0][1], cell_bounds[1][0], cell_bounds[1][1], fill=get_threshold(cells[y][x], 0.5), outline="gray", width=cell_border)

if __name__ == "__main__":
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    canvas = tk.Canvas(frame, width=canvas_size[0] + cell_border, height=canvas_size[1] + cell_border, bg="gray")
    canvas.grid()
    populate_cells(1)
    draw_grid(canvas)
    frame.pack()
    root_window.mainloop()
