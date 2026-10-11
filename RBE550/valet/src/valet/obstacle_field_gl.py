import tkinter as tk
import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL.GL import *


class ObstacleFieldWidget(OpenGLFrame):
    def __init__(self, parent, rows, cols, cell_size, **kwargs):
        super().__init__(parent, **kwargs)
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self._cells = np.zeros((rows, cols), dtype=int)  # 0: empty, 1: obstacle

        # Precompute the static grid border lines to save CPU time
        self.line_vertices = self._compute_line_vertices()

    def initgl(self):
        """Called automatically by pyopengltk to initialize the OpenGL context."""
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # 2D Orthographic projection mapped to exact pixel dimensions of the grid
        width, height = self.cols * self.cell_size, self.rows * self.cell_size
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)

    def _compute_line_vertices(self):
        """Precomputes vertex coordinates for the grid borders."""
        vertices = []
        width = self.cols * self.cell_size
        height = self.rows * self.cell_size

        # Vertical lines
        for c in range(self.cols + 1):
            x = c * self.cell_size
            vertices.extend([x, 0, x, height])
        # Horizontal lines
        for r in range(self.rows + 1):
            y = r * self.cell_size
            vertices.extend([0, y, width, y])

        return np.array(vertices, dtype=np.float32)

    def draw_grid(self):
        """
        Replaces canvas drawing. Uses numpy to vectorize quad calculations
        and sends arrays to the GPU in a single pass.
        """
        glClear(GL_COLOR_BUFFER_BIT)

        rows, cols = self.rows, self.cols

        # 1. Vectorized calculation of all cell quad coordinates
        r, c = np.mgrid[0:rows, 0:cols]
        x0 = c * self.cell_size
        y0 = r * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size

        # Shape: (rows, cols, 4 vertices per quad, 2 coords per vertex)
        quads = np.empty((rows, cols, 4, 2), dtype=np.float32)
        quads[:, :, 0, 0] = x0;
        quads[:, :, 0, 1] = y0
        quads[:, :, 1, 0] = x1;
        quads[:, :, 1, 1] = y0
        quads[:, :, 2, 0] = x1;
        quads[:, :, 2, 1] = y1
        quads[:, :, 3, 0] = x0;
        quads[:, :, 3, 1] = y1

        # 2. Vectorized calculation of cell colors based on state
        # Assume 1 (obstacle) maps to dark grey (0.2) and 0 (empty) to light grey (0.9)
        color_vals = 0.9 - (0.7 * self._cells)
        colors = np.empty((rows, cols, 4, 3), dtype=np.float32)
        for i in range(3):  # Apply identical RGB values for grayscale
            colors[:, :, 0, i] = color_vals
            colors[:, :, 1, i] = color_vals
            colors[:, :, 2, i] = color_vals
            colors[:, :, 3, i] = color_vals

        # Flatten arrays for OpenGL C-struct compatibility
        quads = quads.flatten()
        colors = colors.flatten()

        # 3. Batch draw cell quads (GPU Accelerated)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(2, GL_FLOAT, 0, quads)
        glColorPointer(3, GL_FLOAT, 0, colors)
        glDrawArrays(GL_QUADS, 0, rows * cols * 4)

        glDisableClientState(GL_COLOR_ARRAY)

        # 4. Batch draw grid borders
        glColor3f(0.0, 0.0, 0.0)  # Black grid lines
        glVertexPointer(2, GL_FLOAT, 0, self.line_vertices)
        glDrawArrays(GL_LINES, 0, len(self.line_vertices) // 2)

        glDisableClientState(GL_VERTEX_ARRAY)
        glFlush()

    def redraw(self):
        """Hook called automatically by pyopengltk when a frame update is requested."""
        self.draw_grid()


def create_window(rows=50, cols=50, cell_size=10):
    root = tk.Tk()
    root.title("RBE - GPU Accelerated Obstacle Field")

    width = cols * cell_size
    height = rows * cell_size

    # Initialize the OpenGL widget in place of the old Canvas
    field_widget = ObstacleFieldWidget(root, rows, cols, cell_size, width=width, height=height)
    field_widget.pack(fill=tk.BOTH, expand=True)

    return root, field_widget


if __name__ == '__main__':
    # Initialize UI
    ROWS, COLS = 150, 150
    CELL_SIZE = 6
    root, field_widget = create_window(rows=ROWS, cols=COLS, cell_size=CELL_SIZE)

    # Generate a randomized test grid (e.g., 20% obstacles, 80% free space)
    field_widget._cells = np.random.choice([0, 1], size=(ROWS, COLS), p=[0.8, 0.2])

    # Run application
    root.mainloop()