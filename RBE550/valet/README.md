# Valet

A GPU-accelerated robotics grid environment for RBE550, featuring an OpenGL-based replacement for Tkinter Canvas (`ObstacleFieldWidget`).

## Overview

The `obstacle_field` module in `valet` provides hardware-accelerated grid rendering for occupancy grids, obstacle simulations, and 2D transformed shape overlays. It replaces standard CPU-bound Tkinter Canvas drawing—which suffered from severe performance bottlenecks when rendering tens of thousands of individual rectangles and border lines—with OpenGL fragment shaders powered by `pyopengltk` and `PyOpenGL`.

### Performance Highlights
- **Hardware-Accelerated Rendering**: Cell states and occupancy grids are transferred to GPU textures and rendered using a fragment shader in a single draw call.
- **Fast Border Calculations**: Grid margins and cell borders are computed in parallel per fragment on the GPU.
- **Vectorized Background Overlays**: A* search costs and gradient overlays are mapped and blended via contiguous 2D NumPy arrays (`GridBackgroundMap`), eliminating Python scalar loops.
- **Transformed 2D Shapes**: Draw rotated rectangles (`DrawableRect`) and hierarchical composite groups (`DrawableGroup`) anchored to 2D transforms (`TF2DOwner`) directly over the grid with GPU alpha blending.
- **Seamless Drop-In API**: Maintains the interface and global configuration of `turtles.obstacle_field` (such as `FIELD_WIDTH`, `FIELD_HEIGHT`, `_cell_width`, `_cell_height`, `_fill_func`, `AdaptiveGradientBackground`, and `create_window()`).

## Dependencies

Managed via `uv`:
- `numpy`: Numerical array processing
- `parse`: String parsing utilities
- `pyopengltk`: Native Tkinter OpenGL frame widget integration (ctypes-based WGL/GLX)
- `pyopengl`: Python OpenGL bindings
- `pyopengl-accelerate`: C acceleration module for PyOpenGL

## Usage

### Basic Grid Rendering

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

### Drawing Transformed Shapes & Groups Over Grid

```python
import numpy as np
from valet import DrawableGroup, DrawableRect, obstacle_field as obs

root_window, frame, widget = obs.create_window()
obs.populate_cells(0.2, True, widget)

# 1. Standalone Rectangle (e.g. parking bay)
bay = DrawableRect(
    x=15.0, y=15.0, width=8.0, height=4.0, theta=np.radians(15),
    fill_color=(0.1, 0.8, 0.2, 0.5), border_color="green", border_width=2.0
)
widget.add_drawable(bay)

# 2. Composite Group with Parent-Child Transforms (e.g. vehicle)
car = DrawableGroup(x=35.0, y=30.0, theta=np.radians(45))
chassis = DrawableRect(width=5.0, height=2.6, fill_color="blue", border_color="white")
sensor = DrawableRect(x=2.5, y=0.0, width=0.6, height=0.6, fill_color="red")
wheel_l = DrawableRect(x=1.8, y=-1.4, width=1.0, height=0.4, fill_color="black")
wheel_r = DrawableRect(x=1.8, y=1.4, width=1.0, height=0.4, fill_color="black")

car.add(chassis)
car.add(sensor)
car.add(wheel_l)
car.add(wheel_r)
widget.add_drawable(car)

# Dynamically moving the car moves all children relative to the car frame
car.translate(2.0, 1.0)
car.rotate(np.radians(10))

obs.draw_grid(widget)
frame.pack()
root_window.mainloop()
```
