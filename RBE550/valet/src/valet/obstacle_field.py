"""GPU-accelerated obstacle field module for valet.

Replaces the Tkinter Canvas object with an OpenGL-backed widget (ObstacleFieldWidget)
using pyopengltk and PyOpenGL for hardware-accelerated grid rendering.
"""

from __future__ import annotations

import colorsys
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
import parse
from OpenGL import GL
from OpenGL.GL import shaders
from pyopengltk import OpenGLFrame

# Default occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

_rng = np.random.default_rng()
_cells: list[list[list[Any]]] = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
_rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
_bg_rects: list[Any] = []
_dirty_cells: list[tuple[tuple[int, int], list[Any]]] = [
    ((x, y), []) for y in range(FIELD_HEIGHT) for x in range(FIELD_WIDTH)
]
_fill_func = lambda value: 'black' if value is not None and len(value) > 0 and value[0] > 0 else 'white'

_cell_width = 6
_cell_height = 6
_cell_border = 1

# Active widget reference for convenience
_active_widget: ObstacleFieldWidget | None = None
_grid_colors: np.ndarray | None = None

# Color caches and named color map
_color_cache: dict[str, tuple[int, int, int]] = {}
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'lightgray': (211, 211, 211),
    'lightgrey': (211, 211, 211),
    'darkgray': (169, 169, 169),
    'darkgrey': (169, 169, 169),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
}

# GLSL Shaders for rendering grid of cells and borders entirely on the GPU
_VERT_SRC = """
#version 120
attribute vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 120
uniform sampler2D u_texture;
uniform vec2 u_grid_dims;
uniform vec2 u_cell_size;
uniform float u_cell_border;
uniform vec2 u_canvas_size;
uniform vec2 u_viewport_size;
uniform vec4 u_border_color;

void main() {
    float px = gl_FragCoord.x;
    float py = u_viewport_size.y - gl_FragCoord.y;

    if (px >= u_canvas_size.x || py >= u_canvas_size.y) {
        gl_FragColor = u_border_color;
        return;
    }

    float margin = 2.0 * u_cell_border;
    if (px < margin || py < margin) {
        gl_FragColor = u_border_color;
        return;
    }

    float rel_x = px - margin;
    float rel_y = py - margin;
    float stride_x = u_cell_size.x + u_cell_border;
    float stride_y = u_cell_size.y + u_cell_border;

    int cell_x = int(floor(rel_x / stride_x));
    int cell_y = int(floor(rel_y / stride_y));

    if (cell_x < 0 || cell_x >= int(u_grid_dims.x) || cell_y < 0 || cell_y >= int(u_grid_dims.y)) {
        gl_FragColor = u_border_color;
        return;
    }

    float in_cell_x = mod(rel_x, stride_x);
    float in_cell_y = mod(rel_y, stride_y);

    if (in_cell_x >= u_cell_size.x || in_cell_y >= u_cell_size.y) {
        gl_FragColor = u_border_color;
        return;
    }

    vec2 uv = (vec2(float(cell_x) + 0.5, float(cell_y) + 0.5)) / u_grid_dims;
    gl_FragColor = texture2D(u_texture, uv);
}
"""


class ObstacleFieldWidget(OpenGLFrame):
    """
    OpenGL-accelerated Tkinter widget replacing the Tkinter Canvas.
    Draws the grid of cells and borders in parallel on the GPU.
    """
    def __init__(self, master=None, **kw):
        width = kw.get('width', 800)
        height = kw.get('height', 800)
        self.width = int(width)
        self.height = int(height)
        super().__init__(master, **kw)

        self.prog = None
        self.tex = None
        self.quad_vbo = None
        self._tex_dims: tuple[int, int] | None = None

        self.texture_data: np.ndarray | None = None
        self.texture_needs_upload = False

        self.grid_dims = (FIELD_WIDTH, FIELD_HEIGHT)
        self.cell_size = (float(_cell_width), float(_cell_height))
        self.cell_border = float(_cell_border)
        self.canvas_size = (float(self.width), float(self.height))
        self.border_color = (0.5, 0.5, 0.5, 1.0)

    def initgl(self):
        """Initializes OpenGL viewport, shaders, VBO, and textures."""
        GL.glViewport(0, 0, self.width, self.height)

        if self.prog is None:
            vert = shaders.compileShader(_VERT_SRC, GL.GL_VERTEX_SHADER)
            frag = shaders.compileShader(_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
            self.prog = shaders.compileProgram(vert, frag)

        if self.quad_vbo is None:
            self.quad_vbo = GL.glGenBuffers(1)
            quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_verts.nbytes, quad_verts, GL.GL_STATIC_DRAW)

        if self.tex is None:
            self.tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        if self.texture_data is not None:
            self._upload_texture()

    def set_grid_data(
        self,
        data: np.ndarray,
        grid_dims: tuple[int, int],
        cell_size: tuple[float, float],
        cell_border: float,
        canvas_size: tuple[float, float],
    ):
        """Sets the texture color buffer and uniforms for drawing."""
        self.texture_data = data
        self.grid_dims = grid_dims
        self.cell_size = cell_size
        self.cell_border = cell_border
        self.canvas_size = canvas_size
        self.texture_needs_upload = True
        if self.context_created and self.winfo_ismapped():
            self.display()

    def _upload_texture(self):
        """Uploads the CPU color buffer to the GPU OpenGL texture."""
        if self.tex is None or self.texture_data is None:
            return
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        h, w = self.texture_data.shape[:2]
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if self._tex_dims == (h, w):
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.texture_data)
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.texture_data)
            self._tex_dims = (h, w)
        self.texture_needs_upload = False

    def display(self):
        """Renders the current frame immediately and swaps the buffer."""
        if not self.context_created or not self.winfo_ismapped():
            return
        self.tkMakeCurrent()
        self.redraw()
        self.tkSwapBuffers()

    def redraw(self):
        """Draws the grid in a single GPU pass using the fragment shader."""
        if self.prog is None:
            return
        GL.glClearColor(self.border_color[0], self.border_color[1], self.border_color[2], 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.prog)

        if self.texture_needs_upload and self.texture_data is not None:
            self._upload_texture()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, 'u_texture'), 0)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, 'u_grid_dims'), float(self.grid_dims[0]), float(self.grid_dims[1]))
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, 'u_cell_size'), float(self.cell_size[0]), float(self.cell_size[1]))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, 'u_cell_border'), float(self.cell_border))
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, 'u_canvas_size'), float(self.canvas_size[0]), float(self.canvas_size[1]))
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, 'u_viewport_size'), float(self.width), float(self.height))
        GL.glUniform4f(GL.glGetUniformLocation(self.prog, 'u_border_color'), *self.border_color)

        loc = GL.glGetAttribLocation(self.prog, 'a_pos')
        GL.glEnableVertexAttribArray(loc)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
        GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glDisableVertexAttribArray(loc)

    # Tkinter Canvas compatibility stubs
    def create_rectangle(self, *args, **kwargs):
        return 1

    def itemconfigure(self, tagOrId, **kwargs):
        pass

    def delete(self, *args):
        pass


# Aliases
OpenGLGridWidget = ObstacleFieldWidget
ObstacleFieldGLWidget = ObstacleFieldWidget


class AdaptiveGradientBackground:
    """
    Background to pass to draw_grid that can scale get_gradient to a range of values. Put cells to be colored in cell_map
    and set value_cache to None when the gradient range needs to be updated.
    """
    def __init__(self):
        self.cell_map: dict[tuple[int, int], float] = {}
        self.value_cache: np.ndarray | None = None

    def fill_func(self, value):
        if self.value_cache is None:
            self.value_cache = np.sort(np.array(list(self.cell_map.values())))
            self.value_cache = np.histogram_bin_edges(self.value_cache, 10)
        normalized = np.searchsorted(self.value_cache, value) / self.value_cache.size
        return get_hsv_gradient(float(normalized), 1.0)


def get_basic_tetrominoes():
    """
    :return: a list of arrays representing the tetrominoes shown in the assignment
    """
    print("Encoding basic tetromino set...")
    return [
        np.atleast_2d(np.array([1, 1, 1, 1])).transpose(),
        np.array([[1, 0, 0], [1, 1, 1]]).transpose(),  # L (upside down per fig. 2)
        np.array([[1, 1, 0], [0, 1, 1]]).transpose(),  # S (sideways)
        np.array([[0, 1, 0], [1, 1, 1]]).transpose(),  # T (sideways)
    ]


def get_all_tetrominoes():
    """
    :return: a list of arrays representing all valid tetrominoes, including rotations and mirrors
    """
    print("Enumerating all tetrominoes...")
    tetrominoes = []
    bar = np.atleast_2d(np.array([1, 1, 1, 1]))
    tetrominoes.append(bar)
    tetrominoes.append(bar.T)

    l = np.array([[1, 0, 0], [1, 1, 1]])
    tetrominoes.append(l)
    tetrominoes.append(np.rot90(l))
    tetrominoes.append(np.rot90(l, k=2))
    tetrominoes.append(np.rot90(l, k=3))

    l = l.T  # backwards L
    tetrominoes.append(l)
    tetrominoes.append(np.rot90(l))
    tetrominoes.append(np.rot90(l, k=2))
    tetrominoes.append(np.rot90(l, k=3))

    s = np.array([[1, 1, 0], [0, 1, 1]])
    tetrominoes.append(s)
    tetrominoes.append(np.rot90(s))

    s = s.T  # backwards S
    tetrominoes.append(s)
    tetrominoes.append(np.rot90(s))

    t = np.array([[0, 1, 0], [1, 1, 1]])
    tetrominoes.append(t)
    tetrominoes.append(np.rot90(t))
    tetrominoes.append(np.rot90(t, k=2))
    tetrominoes.append(np.rot90(t, k=3))

    tetrominoes.append(np.array([[1, 1], [1, 1]]))
    return tetrominoes


def populate_cells(coverage: float, use_all_tets: bool = False, canvas=None):
    """
    :param coverage: approximate percentage of cells that should be occupied, on [0, 1]
    (tetrominoes may overlap so coverage is an upper bound)
    :param use_all_tets: if true, use all possible tetrominoes instead of just the ones on the assignment doc
    :param canvas: the canvas/widget to draw on
    :return: None
    """
    global _cells, _dirty_cells, _rects, _grid_colors
    _cells = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
    _rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
    _grid_colors = np.full((FIELD_HEIGHT, FIELD_WIDTH, 3), 255, dtype=np.uint8)

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

    _dirty_cells = [((x, y), _cells[y][x]) for y in range(FIELD_HEIGHT) for x in range(FIELD_WIDTH)]
    if canvas is not None:
        draw_grid(canvas)


__bounds_cache: dict[tuple[int, int, int, int, int], np.ndarray] = {}


def get_cell_bounds(x: int, y: int) -> np.ndarray:
    """
    returns cell bounds in pixels exclusive of borders
    """
    global __bounds_cache
    key = (x, y, _cell_width, _cell_height, _cell_border)
    if key not in __bounds_cache:
        __bounds_cache[key] = np.array([
            [x * _cell_width + x * _cell_border, y * _cell_height + y * _cell_border],
            [(x + 1) * _cell_width + x * _cell_border, (y + 1) * _cell_height + y * _cell_border],
        ]) + 2 * _cell_border
    return __bounds_cache[key]


def get_gradient(v: float) -> str:
    """
    :param v: percentage of red-yellow-green gradient to interpolate to, on [0, 1]
    :return: a string containing a hex representation of an RGB color
    """
    return f'#{int(min(255, 512 * v)):02x}{int(min(255, 512 - 512 * v)):02x}00'


def get_hsv_gradient(hue_percent: float, saturation: float = 1.0, value: float = 1.0) -> str:
    rgb = colorsys.hsv_to_rgb((1 - hue_percent) * (120 / 360), saturation, value)
    return f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'


def convert_text_to_rgb(text: Any) -> tuple[int, int, int]:
    """Converts a color string (hex or name), tuple, or array to an (R, G, B) tuple of ints in [0, 255]."""
    if isinstance(text, (tuple, list, np.ndarray)):
        if len(text) >= 3:
            if isinstance(text[0], float) and text[0] <= 1.0 and text[1] <= 1.0 and text[2] <= 1.0:
                return int(text[0] * 255), int(text[1] * 255), int(text[2] * 255)
            return int(text[0]), int(text[1]), int(text[2])
    if not isinstance(text, str):
        return 255, 255, 255

    cached = _color_cache.get(text)
    if cached is not None:
        return cached

    if text.startswith('#'):
        hex_val = text[1:]
        if len(hex_val) == 6:
            res = (int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16))
            _color_cache[text] = res
            return res
        elif len(hex_val) == 3:
            res = (int(hex_val[0] * 2, 16), int(hex_val[1] * 2, 16), int(hex_val[2] * 2, 16))
            _color_cache[text] = res
            return res

    named = _NAMED_COLORS.get(text.lower())
    if named is not None:
        _color_cache[text] = named
        return named

    try:
        full = parse.parse("#{x}", text)[0]
        res = (full >> 16 & 0xff, full >> 8 & 0xff, full & 0xff)
        _color_cache[text] = res
        return res
    except Exception:
        pass

    return 255, 255, 255


def get_threshold(v: float, threshold: float) -> str:
    """
    :param v: any number
    :param threshold: black/white transition point
    :return: white (as hex RGB in string) if v > threshold, else black
    """
    return "#000000" if v > threshold else "#ffffff"


def draw_grid(canvas=None, background: AdaptiveGradientBackground | None = None):
    """
    Renders the occupancy grid onto the OpenGL widget (or legacy Tkinter Canvas).
    """
    global _dirty_cells, _rects, _bg_rects, _grid_colors, _active_widget
    target = canvas if canvas is not None else _active_widget

    # Ensure color array matches field dimensions
    if _grid_colors is None or _grid_colors.shape[0] != FIELD_HEIGHT or _grid_colors.shape[1] != FIELD_WIDTH:
        _grid_colors = np.full((FIELD_HEIGHT, FIELD_WIDTH, 3), 255, dtype=np.uint8)

    # Process dirty cells into _grid_colors
    for (coords, values) in _dirty_cells:
        x, y = coords
        if 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT:
            color = _fill_func(values)
            _grid_colors[y, x] = convert_text_to_rgb(color)
    _dirty_cells.clear()

    # GPU-accelerated rendering path via ObstacleFieldWidget
    if isinstance(target, ObstacleFieldWidget):
        canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
        if background is not None:
            display_colors = _grid_colors.copy()
            for (coords, value) in background.cell_map.items():
                x, y = coords
                if 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT:
                    current_rgb = display_colors[y, x].astype(np.float32)
                    bg_rgb = np.array(convert_text_to_rgb(background.fill_func(value)), dtype=np.float32)
                    display_colors[y, x] = np.clip(0.75 * current_rgb + 0.25 * bg_rgb, 0, 255).astype(np.uint8)
            target.set_grid_data(
                display_colors,
                (FIELD_WIDTH, FIELD_HEIGHT),
                (float(_cell_width), float(_cell_height)),
                float(_cell_border),
                (float(canvas_size[0]), float(canvas_size[1])),
            )
        else:
            target.set_grid_data(
                _grid_colors,
                (FIELD_WIDTH, FIELD_HEIGHT),
                (float(_cell_width), float(_cell_height)),
                float(_cell_border),
                (float(canvas_size[0]), float(canvas_size[1])),
            )
        return

    # Fallback path for legacy Tkinter Canvas
    if isinstance(target, tk.Canvas):
        if _rects.shape != (FIELD_HEIGHT, FIELD_WIDTH):
            _rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
        for y in range(FIELD_HEIGHT):
            for x in range(FIELD_WIDTH):
                cell_bounds = get_cell_bounds(x, y)
                color = f'#{_grid_colors[y, x][0]:02x}{_grid_colors[y, x][1]:02x}{_grid_colors[y, x][2]:02x}'
                if _rects[y][x] == 0:
                    _rects[y][x] = target.create_rectangle(
                        cell_bounds[0][0], cell_bounds[0][1],
                        cell_bounds[1][0], cell_bounds[1][1],
                        fill=color, outline="gray", width=_cell_border,
                    )
                else:
                    target.itemconfigure(int(_rects[y][x]), fill=color)

        if background is not None:
            target.delete(*_bg_rects)
            _bg_rects.clear()
            for (coords, value) in background.cell_map.items():
                x, y = coords
                cell_bounds = get_cell_bounds(x, y)
                current_color_rgb = _grid_colors[y, x]
                bg_color_rgb = convert_text_to_rgb(background.fill_func(value))
                adjusted_color_rgb = np.add(0.75 * np.array([*current_color_rgb]), 0.25 * np.array([*bg_color_rgb]))
                _bg_rects.append(target.create_rectangle(
                    cell_bounds[0][0], cell_bounds[0][1],
                    cell_bounds[1][0], cell_bounds[1][1],
                    fill=f'#{int(adjusted_color_rgb[0]):02x}{int(adjusted_color_rgb[1]):02x}{int(adjusted_color_rgb[2]):02x}',
                    outline="gray", width=_cell_border,
                ))


def add_to_cell(x: int, y: int, value: Any):
    global _cells, _dirty_cells
    _cells[y][x].append(value)
    _dirty_cells.append(((x, y), list(_cells[y][x])))


def remove_from_cell(x: int, y: int, value: Any):
    global _cells, _dirty_cells
    _cells[y][x].remove(value)
    _dirty_cells.append(((x, y), list(_cells[y][x])))


def get_cells_copy() -> list[list[list[Any]]]:
    return [[_cells[y][x].copy() for x in range(len(_cells[y]))] for y in range(len(_cells))]


def create_window():
    """
    Creates the Tkinter root, a ttk.Frame, and the GPU-accelerated ObstacleFieldWidget.
    """
    global _active_widget
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    widget = ObstacleFieldWidget(frame, width=canvas_size[0], height=canvas_size[1])
    widget.grid(columnspan=4)
    _active_widget = widget
    return root_window, frame, widget


if __name__ == "__main__":
    _root_window, _frame, _canvas = create_window()
    draw_grid(_canvas)
    populate_cells(0.7, True, _canvas)
    _frame.pack()
    _root_window.mainloop()
