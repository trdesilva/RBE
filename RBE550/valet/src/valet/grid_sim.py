import functools
import tkinter as tk
from tkinter import ttk
import time
import fastquadtree as qt

try:
    from valet import obstacle_field as obs
except ImportError:
    try:
        from . import obstacle_field as obs
    except ImportError:
        import obstacle_field as obs

time_rate = 1
fps = 30
_current_time_rate = 0
step_time_callback = None

# UI-available options matching flatland.py
_should_draw_path = False
_should_draw_cost = False
_should_draw_heur = False
_should_draw_collisions = False
checkbox_toggle_callback = None


def should_draw_path() -> bool:
    if isinstance(_should_draw_path, tk.BooleanVar):
        return _should_draw_path.get()
    return bool(_should_draw_path)


def should_draw_cost() -> bool:
    if isinstance(_should_draw_cost, tk.BooleanVar):
        return _should_draw_cost.get()
    return bool(_should_draw_cost)


def should_draw_heur() -> bool:
    if isinstance(_should_draw_heur, tk.BooleanVar):
        return _should_draw_heur.get()
    return bool(_should_draw_heur)


def should_draw_collisions() -> bool:
    if isinstance(_should_draw_collisions, tk.BooleanVar):
        return _should_draw_collisions.get()
    return bool(_should_draw_collisions)


def on_checkbox_toggle():
    if checkbox_toggle_callback is not None:
        checkbox_toggle_callback()
    if obs._active_widget is not None and getattr(obs._active_widget, 'context_created', False):
        obs._active_widget.redraw()


def create_window():
    """
    Creates the Tkinter root, a ttk.Frame, and the GPU-accelerated ObstacleFieldWidget.
    """
    global _should_draw_path, _should_draw_cost, _should_draw_heur, _should_draw_collisions
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = obs.get_cell_bounds(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1)[1]
    obs.obs_quadtree = qt.RectQuadTree((0, 0, obs.FIELD_WIDTH, obs.FIELD_HEIGHT),
                                       capacity=int(0.7 * obs.FIELD_WIDTH * obs.FIELD_HEIGHT))
    widget = obs.ObstacleFieldWidget(frame, width=canvas_size[0], height=canvas_size[1])
    widget.grid(columnspan=4)
    obs._active_widget = widget

    def step_time():
        global _current_time_rate
        loop_start = time.perf_counter_ns()
        if step_time_callback is not None:
            step_time_callback()
        if _current_time_rate > 0:
            root_window.after(int(1000 / fps - (time.perf_counter_ns() - loop_start) / 1000000), step_time)

    def start_time():
        global _current_time_rate
        _current_time_rate = time_rate
        step_time()

    def stop_time():
        global _current_time_rate
        _current_time_rate = 0

    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0)
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=2, column=0)

    _should_draw_path = tk.BooleanVar(value=True)
    _should_draw_cost = tk.BooleanVar(value=True)
    _should_draw_heur = tk.BooleanVar(value=True)
    _should_draw_collisions = tk.BooleanVar(value=True)

    path_cbox = ttk.Checkbutton(frame, text='Draw Path', variable=_should_draw_path, command=on_checkbox_toggle)
    path_cbox.grid(row=1, column=3)
    cost_cbox = ttk.Checkbutton(frame, text='Draw Costs', variable=_should_draw_cost, command=on_checkbox_toggle)
    cost_cbox.grid(row=2, column=3)
    heur_cbox = ttk.Checkbutton(frame, text='Draw Heuristics', variable=_should_draw_heur, command=on_checkbox_toggle)
    heur_cbox.grid(row=3, column=3)
    collide_cbox = ttk.Checkbutton(frame, text='Draw Collisions', variable=_should_draw_collisions, command=on_checkbox_toggle)
    collide_cbox.grid(row=4, column=3)

    return root_window, frame, widget
