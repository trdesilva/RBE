# Valet

A GPU-accelerated robotics grid environment for RBE550, featuring an OpenGL-based replacement for Tkinter Canvas (`ObstacleFieldWidget`).

## Overview

The `obstacle_field` module in `valet` provides hardware-accelerated grid rendering for occupancy grids and obstacle simulations. It replaces standard CPU-bound Tkinter Canvas drawing—which suffered from severe performance bottlenecks when rendering tens of thousands of individual rectangles and border lines—with OpenGL fragment shaders powered by `pyopengltk` and `PyOpenGL`.

### Performance Highlights
- **Hardware-Accelerated Rendering**: Cell states and occupancy grids are transferred to GPU textures and rendered using a fragment shader in a single draw call.
- **Fast Border Calculations**: Grid margins and cell borders are computed in parallel per fragment on the GPU.
- **Seamless Drop-In API**: Maintains the interface and global configuration of `turtles.obstacle_field` (such as `FIELD_WIDTH`, `FIELD_HEIGHT`, `_cell_width`, `_cell_height`, `_fill_func`, `AdaptiveGradientBackground`, and `create_window()`).

## Dependencies

Managed via `uv`:
- `numpy`: Numerical array processing
- `parse`: String parsing utilities
- `pyopengltk`: Native Tkinter OpenGL frame widget integration (ctypes-based WGL/GLX)
- `pyopengl`: Python OpenGL bindings
- `pyopengl-accelerate`: C acceleration module for PyOpenGL

## Usage

```python
from valet import obstacle_field as obs

# Create window and GPU-accelerated widget
root_window, frame, widget = obs.create_window()

# Populate cells and render
obs.populate_cells(0.7, True, widget)
obs.draw_grid(widget)

frame.pack()
root_window.mainloop()
```
