"""GPU-accelerated obstacle field module for valet.

Replaces the Tkinter Canvas object with an OpenGL-backed widget (ObstacleFieldWidget)
using pyopengltk and PyOpenGL for hardware-accelerated grid rendering.
Optimized with pure vectorized NumPy operations to eliminate Python for-loops.
Includes support for transformed 2D shapes (TF2DOwner, DrawableRect, DrawableGroup).
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
import fastquadtree as qt

# Default occupancy grid dimensions
FIELD_WIDTH = 128
FIELD_HEIGHT = 128

_rng = np.random.default_rng()
_cells: list[list[list[Any]]] = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
_rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)
_bg_rects: list[Any] = []
_fill_func = lambda value: 'black' if value is not None and len(value) > 0 and value[0] > 0 else 'white'

obs_quadtree = None

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

# GLSL Shaders for rendering transformed overlay shapes
_SHAPE_VERT_SRC = """
#version 120
attribute vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_SHAPE_FRAG_SRC = """
#version 120
uniform vec4 u_color;
void main() {
    gl_FragColor = u_color;
}
"""


def parse_color_rgba(color: Any, default_alpha: float = 1.0) -> tuple[float, float, float, float]:
    """
    Parses a color specified as RGBA tuple, RGB tuple, hex string, or named color
    into an (r, g, b, a) tuple of floats in [0.0, 1.0].
    """
    if color is None:
        return (0.0, 0.0, 0.0, 0.0)

    if isinstance(color, (tuple, list, np.ndarray)):
        arr = list(color)
        if len(arr) >= 4:
            r = arr[0] / 255.0 if arr[0] > 1.0 else float(arr[0])
            g = arr[1] / 255.0 if arr[1] > 1.0 else float(arr[1])
            b = arr[2] / 255.0 if arr[2] > 1.0 else float(arr[2])
            a = arr[3] / 255.0 if arr[3] > 1.0 else float(arr[3])
            return (r, g, b, a)
        elif len(arr) >= 3:
            r = arr[0] / 255.0 if arr[0] > 1.0 else float(arr[0])
            g = arr[1] / 255.0 if arr[1] > 1.0 else float(arr[1])
            b = arr[2] / 255.0 if arr[2] > 1.0 else float(arr[2])
            return (r, g, b, default_alpha)

    if isinstance(color, str):
        text = color.strip()
        if text.startswith('#') and len(text) == 9:  # #rrggbbaa
            r = int(text[1:3], 16) / 255.0
            g = int(text[3:5], 16) / 255.0
            b = int(text[5:7], 16) / 255.0
            a = int(text[7:9], 16) / 255.0
            return (r, g, b, a)
        rgb = convert_text_to_rgb(text)
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, default_alpha)

    return (0.0, 0.0, 0.0, default_alpha)


class TF2DOwner:
    """
    Base class representing an entity with a 2D rigid transform (translation x, y
    and heading theta in radians). Supports hierarchical parent-child transforms.
    All point and vector transformation APIs operate exclusively using column vectors.
    """
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 theta: float = 0.0,
                 parent: TF2DOwner | None = None
                 ):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)  # in radians
        self.parent = parent

    @property
    def degrees(self) -> float:
        """Rotation angle in degrees."""
        return float(np.degrees(self.theta))

    @degrees.setter
    def degrees(self, val: float):
        self.theta = float(np.radians(val))

    def set_pose(self, x: float, y: float, theta: float | None = None):
        """Sets the position and optional heading."""
        self.x = float(x)
        self.y = float(y)
        if theta is not None:
            self.theta = float(theta)

    def translate(self, dx: float, dy: float):
        """Translates the transform by (dx, dy)."""
        self.x += float(dx)
        self.y += float(dy)

    def translate_rel(self, dx: float, dy: float):
        """Translates relative to current heading using column vector rotation."""
        rot = self.get_heading_transform_matrix()
        d_pos = rot @ np.array([[dx], [dy]], dtype=np.float64)
        self.x += float(d_pos[0, 0])
        self.y += float(d_pos[1, 0])

    def rotate(self, dtheta: float):
        """Rotates the transform by dtheta radians."""
        self.theta += float(dtheta)

    def get_local_transform_matrix(self) -> np.ndarray:
        """
        Returns the 3x3 homogeneous transformation matrix for this local pose:
        [[cos(theta), -sin(theta), x],
         [sin(theta),  cos(theta), y],
         [         0,           0, 1]]
        """
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def get_world_transform_matrix(self) -> np.ndarray:
        """Recursively computes the world transform matrix by chaining parent transforms."""
        local = self.get_local_transform_matrix()
        if self.parent is not None:
            return self.parent.get_world_transform_matrix() @ local
        return local

    def get_heading_transform_matrix(self):
        """Get only the rotation part of the world transform matrix."""
        t = self.get_world_transform_matrix()
        return t[0:2, 0:2]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms local points into world coordinates using the world
        transformation matrix and column-vector convention (m @ h_pts).

        :param points: Local points as column vectors of shape (2, N) or a single
                       vector of shape (2,).
        :return: Transformed points as column vectors of shape (2, N) or (2,).
        """
        pts = np.asarray(points, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected 2-element vector, got shape {pts.shape}")
            pts = pts.reshape(2, 1)
        elif pts.ndim == 2:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected column vectors with shape (2, N), got {pts.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D array, got {pts.ndim}D array")

        ones = np.ones((1, pts.shape[1]), dtype=np.float64)
        h_pts = np.vstack([pts, ones])  # (3, N)
        m = self.get_world_transform_matrix()  # (3, 3)
        world_h = m @ h_pts  # (3, N)
        result = world_h[:2, :]

        return result.ravel() if is_1d else result


class DrawableRect(TF2DOwner):
    """
    A transformable 2D rectangle drawable on the grid or in screen pixels.
    All vertex APIs return column vectors of shape (2, N).
    """
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 1.0,
        height: float = 1.0,
        theta: float = 0.0,
        origin: tuple[float, float] = (0.5, 0.5),  # (0.5, 0.5) is centered at (x, y)
        fill_color: Any = (1.0, 0.0, 0.0, 1.0),
        border_color: Any = None,
        border_width: float = 1.0,
        coordinate_frame: str = "grid",  # "grid" or "pixel"
        visible: bool = True,
        parent: TF2DOwner | None = None,
    ):
        super().__init__(x=x, y=y, theta=theta, parent=parent)
        self.width = float(width)
        self.height = float(height)
        self.origin = (float(origin[0]), float(origin[1]))
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = float(border_width)
        self.coordinate_frame = coordinate_frame
        self.visible = visible

    def get_local_vertices(self) -> np.ndarray:
        """
        Returns the 4 local vertices of the rectangle relative to its pose origin
        as column vectors of shape (2, 4):
        columns: [top-left, top-right, bottom-right, bottom-left].
        """
        ox, oy = self.origin
        x0 = -ox * self.width
        x1 = (1.0 - ox) * self.width
        y0 = -oy * self.height
        y1 = (1.0 - oy) * self.height
        return np.array([
            [x0, x1, x1, x0],
            [y0, y0, y1, y1],
        ], dtype=np.float64)

    def get_world_vertices(self) -> np.ndarray:
        """
        Transforms local vertices into world (grid or root parent) coordinates
        as column vectors of shape (2, 4).
        """
        return self.transform_points(self.get_local_vertices())

    def get_aabb(self, qt_format = True):
        """
        Get coordinates of axis-aligned bounding box.
        :param qt_format: whether to use fastquadtree's convention of (minx, miny, maxx, maxy) or a matrix with two
        column vectors for the return value
        :return: coordinates of the bounding box's corners, ordered according to qt_format
        """
        world_verts = self.get_world_vertices()
        # world_verts has shape (2, 4): row 0 is x, row 1 is y
        minx = np.min(world_verts[0, :])
        miny = np.min(world_verts[1, :])
        maxx = np.max(world_verts[0, :])
        maxy = np.max(world_verts[1, :])
        # fastquadtree convention (minx, miny, maxx, maxy), else two column vectors
        return (minx, miny, maxx, maxy) if qt_format else np.array(((minx, maxx), (miny, maxy)))


class DrawableGroup(TF2DOwner):
    """
    A composite collection of DrawableRects and nested DrawableGroups that share
    a common 2D coordinate transform.
    """
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        visible: bool = True,
        parent: TF2DOwner | None = None,
    ):
        super().__init__(x=x, y=y, theta=theta, parent=parent)
        self.children: list[DrawableRect | DrawableGroup] = []
        self.visible = visible

    def add(self, drawable: DrawableRect | DrawableGroup):
        """Adds a child drawable to this group and binds this group as its parent."""
        if drawable not in self.children:
            self.children.append(drawable)
            drawable.parent = self

    def remove(self, drawable: DrawableRect | DrawableGroup):
        """Removes a child drawable from this group."""
        if drawable in self.children:
            self.children.remove(drawable)
            if drawable.parent is self:
                drawable.parent = None

    def clear(self):
        """Removes all children from this group."""
        for child in self.children:
            if child.parent is self:
                child.parent = None
        self.children.clear()

    def get_all_rects(self) -> list[DrawableRect]:
        """Recursively collects all visible DrawableRect leaves in this group."""
        if not self.visible:
            return []
        rects: list[DrawableRect] = []
        for child in self.children:
            if isinstance(child, DrawableRect):
                if child.visible:
                    rects.append(child)
            elif isinstance(child, DrawableGroup):
                rects.extend(child.get_all_rects())
        return rects

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)


def vectorized_hue_to_rgb(hue_percent: np.ndarray) -> np.ndarray:
    """
    Pure NumPy vectorized HSV to RGB conversion for hue_percent in [0, 1]
    mapped to hue in [120/360, 0] (Green to Red), with saturation=1.0, value=1.0.
    Returns an (N, 3) uint8 array of RGB colors in [0, 255].
    """
    hp = np.asarray(hue_percent, dtype=np.float32)
    h = (1.0 - hp) * (120.0 / 360.0) * 6.0  # h in [0, 2]
    r = np.clip(np.where(h <= 1.0, 1.0, 2.0 - h), 0.0, 1.0)
    g = np.clip(np.where(h <= 1.0, h, 1.0), 0.0, 1.0)
    b = np.zeros_like(h)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class GridBackgroundMap:
    """
    High-performance 2D NumPy array-backed map for cell background values.
    Provides a dictionary-like interface (indexing, get, pop, clear, contains, items)
    while storing values in a contiguous (height, width) float32 array for zero-copy
    vectorized operations.
    """
    def __init__(self, width: int = FIELD_WIDTH, height: int = FIELD_HEIGHT):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), np.nan, dtype=np.float32)

    def _ensure_dimensions(self, width: int, height: int):
        if self.width != width or self.height != height or self.grid.shape != (height, width):
            new_grid = np.full((height, width), np.nan, dtype=np.float32)
            common_h = min(self.height, height)
            common_w = min(self.width, width)
            new_grid[:common_h, :common_w] = self.grid[:common_h, :common_w]
            self.grid = new_grid
            self.width = width
            self.height = height

    def __setitem__(self, key: tuple[int, int], value: float):
        x, y = key
        if not (0 <= x < self.width and 0 <= y < self.height):
            self._ensure_dimensions(max(self.width, x + 1), max(self.height, y + 1))
        self.grid[y, x] = float(value)

    def __getitem__(self, key: tuple[int, int]) -> float:
        x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            v = self.grid[y, x]
            if not np.isnan(v):
                return float(v)
        raise KeyError(key)

    def get(self, key: tuple[int, int], default: Any = None) -> Any:
        x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            v = self.grid[y, x]
            if not np.isnan(v):
                return float(v)
        return default

    def pop(self, key: tuple[int, int], default: Any = None) -> Any:
        x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            v = self.grid[y, x]
            if not np.isnan(v):
                self.grid[y, x] = np.nan
                return float(v)
        return default

    def clear(self):
        self.grid.fill(np.nan)

    def __contains__(self, key: tuple[int, int]) -> bool:
        x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            return not np.isnan(self.grid[y, x])
        return False

    def __len__(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.grid)))

    def items(self):
        ys, xs = np.nonzero(~np.isnan(self.grid))
        for y, x in zip(ys, xs):
            yield (int(x), int(y)), float(self.grid[y, x])

    def keys(self):
        ys, xs = np.nonzero(~np.isnan(self.grid))
        for y, x in zip(ys, xs):
            yield (int(x), int(y))

    def values(self):
        vals = self.grid[~np.isnan(self.grid)]
        for v in vals:
            yield float(v)


class AdaptiveGradientBackground:
    """
    Background to pass to draw_grid that can scale get_gradient to a range of values.
    Uses GridBackgroundMap for O(1) cell access and pure vectorized blending in draw_grid.
    """
    def __init__(self, width: int = FIELD_WIDTH, height: int = FIELD_HEIGHT):
        self.cell_map = GridBackgroundMap(width, height)
        self.value_cache: np.ndarray | None = None

    def fill_func(self, value):
        if self.value_cache is None:
            if isinstance(self.cell_map, GridBackgroundMap):
                active = self.cell_map.grid[~np.isnan(self.cell_map.grid)]
            else:
                active = np.array(list(self.cell_map.values()))
            if active.size > 0:
                self.value_cache = np.histogram_bin_edges(np.sort(active), 10)
            else:
                self.value_cache = np.array([0.0, 1.0])
        normalized = np.searchsorted(self.value_cache, value) / self.value_cache.size
        return get_hsv_gradient(float(normalized), 1.0)


class ObstacleFieldWidget(TF2DOwner, OpenGLFrame):
    """
    OpenGL-accelerated Tkinter widget replacing the Tkinter Canvas.
    Draws the grid of cells and borders in parallel on the GPU, and renders
    collections of transformed shapes (DrawableRect, DrawableGroup) over the grid.
    Public coordinate conversion APIs work in column vectors of shape (2, N).
    """
    def __init__(self, master=None, **kw):
        super().__init__()
        width = kw.get('width', 800)
        height = kw.get('height', 800)
        self.width = int(width)
        self.height = int(height)
        super(TF2DOwner, self).__init__(master, **kw)

        # Shaders & Buffers
        self.prog = None
        self.tex = None
        self.quad_vbo = None
        self._tex_dims: tuple[int, int] | None = None

        self.shape_prog = None
        self.shape_vbo = None

        # Grid data
        self.texture_data: np.ndarray | None = None
        self.texture_needs_upload = False

        self.grid_dims = (FIELD_WIDTH, FIELD_HEIGHT)
        self.cell_size = (float(_cell_width), float(_cell_height))
        self.cell_border = float(_cell_border)
        self.canvas_size = (float(self.width), float(self.height))
        self.border_color = (0.5, 0.5, 0.5, 1.0)

        # Collection of overlay drawables
        self.drawables: list[DrawableRect | DrawableGroup] = []
        self._in_display = False

    def add_drawable(self, drawable: DrawableRect | DrawableGroup):
        """Adds a drawable shape or group to the overlay collection."""
        if drawable not in self.drawables:
            self.drawables.append(drawable)

    def remove_drawable(self, drawable: DrawableRect | DrawableGroup):
        """Removes a drawable shape or group from the overlay collection."""
        if drawable in self.drawables:
            self.drawables.remove(drawable)

    def clear_drawables(self):
        """Removes all drawable shapes and groups from the overlay collection."""
        self.drawables.clear()

    def grid_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """
        Converts (2, N) grid coordinates (column vectors) to window screen pixels (2, N):
        px = 2 * cell_border + gx * (cell_width + cell_border)
        py = 2 * cell_border + gy * (cell_height + cell_border)
        """
        pts = np.asarray(points, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected 2-element vector, got shape {pts.shape}")
            pts = pts.reshape(2, 1)
        elif pts.ndim == 2:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected column vectors with shape (2, N), got {pts.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D array, got {pts.ndim}D array")

        margin = 2.0 * self.cell_border
        stride_x = self.cell_size[0] + self.cell_border
        stride_y = self.cell_size[1] + self.cell_border
        px = margin + pts[0, :] * stride_x
        py = margin + pts[1, :] * stride_y
        res = np.vstack([px, py])
        return res.ravel() if is_1d else res

    def pixel_to_ndc(self, pixels: np.ndarray) -> np.ndarray:
        """
        Converts (2, N) window screen pixel coordinates (column vectors) to OpenGL NDC [-1, 1] (2, N):
        X_ndc = (2 * px / viewport_width) - 1.0
        Y_ndc = 1.0 - (2 * py / viewport_height)
        """
        pts = np.asarray(pixels, dtype=np.float64)
        is_1d = pts.ndim == 1
        if is_1d:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected 2-element vector, got shape {pts.shape}")
            pts = pts.reshape(2, 1)
        elif pts.ndim == 2:
            if pts.shape[0] != 2:
                raise ValueError(f"Expected column vectors with shape (2, N), got {pts.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D array, got {pts.ndim}D array")

        vw = max(float(self.width), 1.0)
        vh = max(float(self.height), 1.0)
        ndc_x = (2.0 * pts[0, :] / vw) - 1.0
        ndc_y = 1.0 - (2.0 * pts[1, :] / vh)
        res = np.vstack([ndc_x, ndc_y])
        return res.ravel() if is_1d else res

    def initgl(self):
        """Initializes OpenGL viewport, shaders, VBOs, and textures."""
        GL.glViewport(0, 0, self.width, self.height)

        # 1. Grid pass shaders & quad
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

        # 2. Shape overlay shader & buffer
        if self.shape_prog is None:
            s_vert = shaders.compileShader(_SHAPE_VERT_SRC, GL.GL_VERTEX_SHADER)
            s_frag = shaders.compileShader(_SHAPE_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
            self.shape_prog = shaders.compileProgram(s_vert, s_frag)

        if self.shape_vbo is None:
            self.shape_vbo = GL.glGenBuffers(1)

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
        """Renders the current frame immediately and swaps the buffer to display on screen."""
        if not self.context_created or not self.winfo_ismapped():
            return
        self.tkMakeCurrent()
        self._in_display = True
        try:
            self.redraw()
        finally:
            self._in_display = False
        self.tkSwapBuffers()

    def _display(self, evt=None):
        """Handles display callbacks (e.g. from <Expose> events)."""
        self.display()
        if self.animate > 0:
            self.cb = self.after(self.animate, self._display)

    def redraw(self):
        """
        Draws the grid in Pass 1, followed by all transformed overlay shapes in Pass 2.
        When invoked directly by external callers, delegates to display() to ensure
        context is current and front/back buffers are swapped.
        """
        if not getattr(self, '_in_display', False):
            self.display()
            return

        if self.prog is None:
            return

        GL.glViewport(0, 0, self.width, self.height)

        # -------------------------------------------------------------
        # PASS 1: Render background occupancy grid & borders
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # PASS 2: Render transformed overlay shapes (DrawableRects)
        # -------------------------------------------------------------
        if self.drawables and self.shape_prog is not None:
            rects: list[DrawableRect] = []
            for item in self.drawables:
                if isinstance(item, DrawableRect):
                    if item.visible:
                        rects.append(item)
                elif isinstance(item, DrawableGroup):
                    rects.extend(item.get_all_rects())

            if rects:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glUseProgram(self.shape_prog)

                loc_pos = GL.glGetAttribLocation(self.shape_prog, 'a_pos')
                loc_col = GL.glGetUniformLocation(self.shape_prog, 'u_color')
                GL.glEnableVertexAttribArray(loc_pos)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.shape_vbo)

                for rect in rects:
                    world_pts = rect.get_world_vertices()  # (2, 4) column vectors
                    if rect.coordinate_frame == "grid":
                        pixel_pts = self.grid_to_pixel(world_pts)  # (2, 4) column vectors
                    else:
                        pixel_pts = world_pts
                    ndc_pts = self.pixel_to_ndc(pixel_pts)  # (2, 4) column vectors

                    # Implementation detail: OpenGL buffer expects row vectors (4, 2)
                    ndc_pts_row = ndc_pts.T

                    # Filled quad (2 triangles: 0-1-2 and 0-2-3)
                    fill_rgba = parse_color_rgba(rect.fill_color)
                    if fill_rgba[3] > 0.0:
                        tri_verts = np.array([
                            ndc_pts_row[0], ndc_pts_row[1], ndc_pts_row[2],
                            ndc_pts_row[0], ndc_pts_row[2], ndc_pts_row[3],
                        ], dtype=np.float32)
                        GL.glUniform4f(loc_col, *fill_rgba)
                        GL.glBufferData(GL.GL_ARRAY_BUFFER, tri_verts.nbytes, tri_verts, GL.GL_DYNAMIC_DRAW)
                        GL.glVertexAttribPointer(loc_pos, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

                    # Outline border (line loop: 0-1-2-3)
                    if rect.border_color is not None and rect.border_width > 0.0:
                        border_rgba = parse_color_rgba(rect.border_color)
                        if border_rgba[3] > 0.0:
                            line_verts = np.ascontiguousarray(ndc_pts_row, dtype=np.float32)
                            GL.glLineWidth(max(1.0, float(rect.border_width)))
                            GL.glUniform4f(loc_col, *border_rgba)
                            GL.glBufferData(GL.GL_ARRAY_BUFFER, line_verts.nbytes, line_verts, GL.GL_DYNAMIC_DRAW)
                            GL.glVertexAttribPointer(loc_pos, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)

                GL.glDisableVertexAttribArray(loc_pos)
                GL.glDisable(GL.GL_BLEND)

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


def _sync_grid_colors_from_cells():
    """Synchronizes _grid_colors from _cells matrix."""
    global _grid_colors
    empty_rgb = convert_text_to_rgb(_fill_func([]))
    _grid_colors = np.full((FIELD_HEIGHT, FIELD_WIDTH, 3), empty_rgb, dtype=np.uint8)
    for y in range(min(len(_cells), FIELD_HEIGHT)):
        row = _cells[y]
        for x in range(min(len(row), FIELD_WIDTH)):
            if len(row[x]) > 0:
                _grid_colors[y, x] = convert_text_to_rgb(_fill_func(row[x]))


def populate_cells(coverage: float, use_all_tets: bool = True, canvas=None):
    """
    :param coverage: approximate percentage of cells that should be occupied, on [0, 1]
    (tetrominoes may overlap so coverage is an upper bound)
    :param use_all_tets: if true, use all possible tetrominoes instead of just the ones on the assignment doc
    :param canvas: the canvas/widget to draw on
    :return: A dictionary mapping (x, y) coordinate pairs to tetromino cell arrays placed there
    """
    global _cells, _rects, _grid_colors, obs_quadtree
    _cells = [[[] for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]
    _rects = np.zeros((FIELD_HEIGHT, FIELD_WIDTH), dtype=np.uint16)

    empty_rgb = convert_text_to_rgb(_fill_func([]))
    _grid_colors = np.full((FIELD_HEIGHT, FIELD_WIDTH, 3), empty_rgb, dtype=np.uint8)

    tetrominoes = get_all_tetrominoes() if use_all_tets else get_basic_tetrominoes()
    total = int(coverage * FIELD_HEIGHT * FIELD_WIDTH / 4)
    print(f'Placing {total} tetrominoes...')
    tets_to_place = _rng.integers(len(tetrominoes), size=total)
    placed_coords = {}
    for tet in tets_to_place:
        tet_cells = tetrominoes[tet]
        tet_bounds = np.shape(tet_cells)
        x = _rng.integers(0, FIELD_WIDTH - tet_bounds[1], endpoint=True)
        y = _rng.integers(0, FIELD_HEIGHT - tet_bounds[0], endpoint=True)
        for i in np.ndindex(tet_bounds):
            if tet_cells[i] == 1:
                cy, cx = y + i[0], x + i[1]
                cell = _cells[cy][cx]
                if len(cell) > 0:
                    cell[0] = tet_cells[i]
                else:
                    cell.append(tet_cells[i])
                    obs_quadtree.insert((cx, cy, cx + 1, cy + 1))

        placed_coords[(x, y)] = tet_cells

    # Vectorized color assignment for occupied cells
    wall_rgb = convert_text_to_rgb(_fill_func([1]))
    occupied = np.array([[len(c) > 0 and c[0] > 0 for c in row] for row in _cells], dtype=bool)
    _grid_colors[occupied] = wall_rgb

    if canvas is not None:
        draw_grid()
    return placed_coords

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


def _apply_vectorized_background(base_colors: np.ndarray, background: AdaptiveGradientBackground) -> np.ndarray:
    """
    Applies background gradient overlays onto base_colors using pure NumPy vectorized operations.
    Eliminates all scalar Python iteration.
    """
    cell_map = background.cell_map
    if isinstance(cell_map, GridBackgroundMap):
        cell_map._ensure_dimensions(FIELD_WIDTH, FIELD_HEIGHT)
        mask = ~np.isnan(cell_map.grid)
        if not np.any(mask):
            return base_colors

        active_vals = cell_map.grid[mask]
        if background.value_cache is None:
            background.value_cache = np.histogram_bin_edges(np.sort(active_vals), 10)

        norm = np.searchsorted(background.value_cache, active_vals) / background.value_cache.size
        bg_rgb = vectorized_hue_to_rgb(norm)

        display_colors = base_colors.copy()
        base_masked = display_colors[mask].astype(np.float32)
        display_colors[mask] = (0.75 * base_masked + 0.25 * bg_rgb.astype(np.float32)).astype(np.uint8)
        return display_colors

    elif isinstance(cell_map, dict) and cell_map:
        coords = list(cell_map.keys())
        xs = np.fromiter((c[0] for c in coords), dtype=int, count=len(coords))
        ys = np.fromiter((c[1] for c in coords), dtype=int, count=len(coords))
        vals = np.fromiter(cell_map.values(), dtype=np.float32, count=len(coords))

        valid = (xs >= 0) & (xs < FIELD_WIDTH) & (ys >= 0) & (ys < FIELD_HEIGHT)
        if not np.any(valid):
            return base_colors

        xs, ys, vals = xs[valid], ys[valid], vals[valid]
        if background.value_cache is None:
            background.value_cache = np.histogram_bin_edges(np.sort(vals), 10)

        norm = np.searchsorted(background.value_cache, vals) / background.value_cache.size
        bg_rgb = vectorized_hue_to_rgb(norm)

        display_colors = base_colors.copy()
        base_masked = display_colors[ys, xs].astype(np.float32)
        display_colors[ys, xs] = (0.75 * base_masked + 0.25 * bg_rgb.astype(np.float32)).astype(np.uint8)
        return display_colors

    return base_colors


def draw_grid(background: AdaptiveGradientBackground | None = None):
    """
    Renders the occupancy grid onto the OpenGL widget (or legacy Tkinter Canvas).
    Completely eliminates Python for-loops by relying on O(1) cell updates and NumPy vectorization.
    """
    global _rects, _bg_rects, _grid_colors, _active_widget
    target = _active_widget

    # Ensure color array matches field dimensions
    if _grid_colors is None or _grid_colors.shape[0] != FIELD_HEIGHT or _grid_colors.shape[1] != FIELD_WIDTH:
        _sync_grid_colors_from_cells()

    # Apply background overlay vectorially if requested
    display_colors = _grid_colors
    if background is not None:
        display_colors = _apply_vectorized_background(display_colors, background)

    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    target.set_grid_data(
        display_colors,
        (FIELD_WIDTH, FIELD_HEIGHT),
        (float(_cell_width), float(_cell_height)),
        float(_cell_border),
        (float(canvas_size[0]), float(canvas_size[1])),
    )

def add_to_cell(x: int, y: int, value: Any):
    """Adds value to cell (x, y) and immediately updates _grid_colors in O(1)."""
    global _cells, _grid_colors
    _cells[y][x].append(value)
    if _grid_colors is not None and 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT:
        _grid_colors[y, x] = convert_text_to_rgb(_fill_func(_cells[y][x]))


def remove_from_cell(x: int, y: int, value: Any):
    """Removes value from cell (x, y) and immediately updates _grid_colors in O(1)."""
    global _cells, _grid_colors
    _cells[y][x].remove(value)
    if _grid_colors is not None and 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT:
        _grid_colors[y, x] = convert_text_to_rgb(_fill_func(_cells[y][x]))


def get_cells_copy() -> list[list[list[Any]]]:
    return [[_cells[y][x].copy() for x in range(len(_cells[y]))] for y in range(len(_cells))]


def create_window():
    """
    Creates the Tkinter root, a ttk.Frame, and the GPU-accelerated ObstacleFieldWidget.
    """
    global _active_widget, obs_quadtree
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = get_cell_bounds(FIELD_WIDTH - 1, FIELD_HEIGHT - 1)[1]
    obs_quadtree = qt.RectQuadTree((0, 0, FIELD_WIDTH, FIELD_HEIGHT), capacity=int(0.7 * FIELD_WIDTH * FIELD_HEIGHT))
    widget = ObstacleFieldWidget(frame, width=canvas_size[0], height=canvas_size[1])
    widget.grid(columnspan=4)
    _active_widget = widget
    return root_window, frame, widget


if __name__ == "__main__":
    _root_window, _frame, _canvas = create_window()
    draw_grid()
    populate_cells(0.7, True, _canvas)
    _frame.pack()
    _root_window.mainloop()
