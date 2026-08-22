import numpy as np
import tkinter as tk
from tkinter import ttk

# occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

cells = np.random.uniform(0, 1, (FIELD_HEIGHT, FIELD_WIDTH))
print(cells)

cell_width = 6
cell_height = 6
cell_border = 1

def get_cell_bounds(x, y):
    """
     returns cell bounds in pixels exclusive of borders
    """
    return np.array([[x*cell_width + (x)*cell_border, y*cell_height + (y)*cell_border],
                     [(x + 1)*cell_width + (x)*cell_border, (y + 1)*cell_height + (y)*cell_border]]) + 2*cell_border

def get_gradient(v):
    return f'#{int(min(255, 512 * v)):02x}{int(min(255, 512 - 512 * v)):02x}00'

def get_threshold(v, threshold):
    return "#000000" if v < threshold else "#ffffff"

def draw_grid(canvas: tk.Canvas):
    for y in range(FIELD_HEIGHT):
        for x in range(FIELD_WIDTH):
            cell_bounds = get_cell_bounds(x, y)
            print(f'({x},{y}) => {cell_bounds}')
            canvas.create_rectangle(cell_bounds[0][0], cell_bounds[0][1], cell_bounds[1][0], cell_bounds[1][1], fill=get_threshold(cells[y][x], 0.5), outline="gray", width=cell_border)

if __name__ == "__main__":
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    canvas = tk.Canvas(frame, width=canvas_size[0] + cell_border, height=canvas_size[1] + cell_border, bg="gray")
    canvas.grid()
    draw_grid(canvas)
    frame.pack()
    root_window.mainloop()
