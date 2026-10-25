import functools
import tkinter as tk
from tkinter import ttk
import time
import numpy as np
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

# Vehicle & MotionPrimitive manual control hooks
active_vehicle = None
run_astar_callback = None
clear_visualization_callback = None
enqueue_motion_callback = None

_preview_line_strip = None
_selected_mp = None
_mp_map = {}
_mp_combobox = None
_mp_var = None


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


def clear_preview():
    """Removes any active motion primitive preview strip from the active widget."""
    global _preview_line_strip
    if _preview_line_strip is not None:
        widget = obs._active_widget
        if widget is not None and getattr(widget, 'drawables', None) is not None:
            if _preview_line_strip in widget.drawables:
                widget.remove_drawable(_preview_line_strip)
        _preview_line_strip = None
        if widget is not None and getattr(widget, 'context_created', False):
            widget.redraw()


def update_preview_pose():
    """Updates the preview line strip's pose to match the vehicle's current pose."""
    if _preview_line_strip is not None and active_vehicle is not None:
        _preview_line_strip.set_pose(active_vehicle.x, active_vehicle.y, active_vehicle.theta)


def on_mp_selected(event=None):
    """Called when a motion primitive selection changes in the dropdown."""
    global _selected_mp, _preview_line_strip
    clear_preview()
    if _mp_combobox is None:
        return
    sel = _mp_combobox.get()
    _selected_mp = _mp_map.get(sel)
    if _selected_mp is not None and getattr(_selected_mp, 'line_strip', None) is not None:
        pose_x, pose_y, pose_theta = 0.0, 0.0, 0.0
        if active_vehicle is not None:
            pose_x = active_vehicle.x
            pose_y = active_vehicle.y
            pose_theta = active_vehicle.theta
        _preview_line_strip = _selected_mp.line_strip.copy()
        _preview_line_strip.set_pose(pose_x, pose_y, pose_theta)
        _preview_line_strip.color = (1.0, 0.8, 0.0, 1.0)
        _preview_line_strip.line_width = 3.0
        _preview_line_strip.visible = True
        if obs._active_widget is not None:
            obs._active_widget.add_drawable(_preview_line_strip)
            if getattr(obs._active_widget, 'context_created', False):
                obs._active_widget.redraw()


def enqueue_selected_mp():
    """Enqueues the selected motion primitive to the vehicle's queue."""
    global _selected_mp
    if _selected_mp is None and _mp_combobox is not None:
        _selected_mp = _mp_map.get(_mp_combobox.get())
    if _selected_mp is not None:
        if enqueue_motion_callback is not None:
            enqueue_motion_callback(_selected_mp)
        elif active_vehicle is not None:
            active_vehicle.queue_motion(_selected_mp)


def on_run_astar():
    """Runs A* search from current position."""
    if run_astar_callback is not None:
        run_astar_callback()
    elif active_vehicle is not None:
        try:
            from valet.planner import MPSearchNode
        except ImportError:
            try:
                from .planner import MPSearchNode
            except ImportError:
                from planner import MPSearchNode
        start_node = MPSearchNode(None, active_vehicle.x, active_vehicle.y, active_vehicle.theta, 1, active_vehicle.body)
        path = active_vehicle.astar(start_node, (8, 11, 10, 12))
        if path is not None:
            path = list(path)
            active_vehicle.optimal_path_line_strips = [
                getattr(s, 'line_strip', None) for s in path if getattr(s, 'line_strip', None) is not None
            ]
            for s in path:
                if s.via:
                    active_vehicle.queue_motion(s.via)
            active_vehicle.update_visualization()
            if obs._active_widget is not None and getattr(obs._active_widget, 'context_created', False):
                obs._active_widget.redraw()


def on_clear_visuals():
    """Clears all visualized motion primitives and search lines."""
    clear_preview()
    if clear_visualization_callback is not None:
        clear_visualization_callback()
    elif active_vehicle is not None and hasattr(active_vehicle, 'clear_visualizations'):
        active_vehicle.clear_visualizations()
    if obs._active_widget is not None and getattr(obs._active_widget, 'context_created', False):
        obs._active_widget.redraw()


def populate_motion_primitives(mps_by_dt=None):
    """Populates the motion primitives dropdown options from Car.mps_by_dt or provided dictionary."""
    global _mp_map, _mp_combobox
    _mp_map.clear()
    if mps_by_dt is None:
        if active_vehicle is not None and hasattr(active_vehicle, 'mps_by_dt'):
            mps_by_dt = active_vehicle.mps_by_dt
        else:
            try:
                from valet.planner import Car
            except ImportError:
                try:
                    from .planner import Car
                except ImportError:
                    from planner import Car
            mps_by_dt = Car.mps_by_dt

    for dt, mps in mps_by_dt.items():
        for i, mp in enumerate(mps):
            u_str = ""
            if mp.inputs and hasattr(mp.inputs[0], 'u'):
                u = mp.inputs[0].u
                if len(u) == 2:
                    deg = float(np.rad2deg(u[1]))
                    u_str = f"v={u[0]:.1f}, phi={deg:.1f} deg"
                else:
                    u_str = f"u={np.round(u, 2)}"
            label = f"dt={dt}s | {u_str}" if u_str else f"dt={dt}s #{i+1}"
            unique_label = label
            counter = 1
            while unique_label in _mp_map:
                unique_label = f"{label} ({counter})"
                counter += 1
            _mp_map[unique_label] = mp

    if _mp_combobox is not None:
        vals = list(_mp_map.keys())
        _mp_combobox['values'] = vals
        if vals and not _mp_combobox.get():
            _mp_combobox.set('')


def set_active_vehicle(vehicle):
    """Sets the active vehicle and refreshes the motion primitives dropdown if appropriate."""
    global active_vehicle
    active_vehicle = vehicle
    if vehicle is not None and hasattr(vehicle, 'mps_by_dt'):
        populate_motion_primitives(vehicle.mps_by_dt)
    elif not _mp_map:
        populate_motion_primitives()


def create_window(vehicle=None):
    """
    Creates the Tkinter root, a ttk.Frame, and the GPU-accelerated ObstacleFieldWidget.
    """
    global _should_draw_path, _should_draw_cost, _should_draw_heur, _should_draw_collisions
    global _mp_combobox, _mp_var, active_vehicle

    if vehicle is not None:
        active_vehicle = vehicle

    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = obs.get_cell_bounds(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1)[1]
    obs.obs_quadtree = qt.RectQuadTree((0, 0, obs.FIELD_WIDTH, obs.FIELD_HEIGHT),
                                       capacity=int(0.7 * obs.FIELD_WIDTH * obs.FIELD_HEIGHT))
    widget = obs.ObstacleFieldWidget(frame, width=canvas_size[0], height=canvas_size[1])
    widget.grid(row=0, column=0, columnspan=4)
    obs._active_widget = widget

    def step_time():
        global _current_time_rate
        loop_start = time.perf_counter_ns()
        if step_time_callback is not None:
            step_time_callback()
        update_preview_pose()
        if _current_time_rate > 0:
            root_window.after(int(1000 / fps - (time.perf_counter_ns() - loop_start) / 1000000), step_time)

    def start_time():
        global _current_time_rate
        _current_time_rate = time_rate
        step_time()

    def stop_time():
        global _current_time_rate
        _current_time_rate = 0

    # Column 0: Time control & A* / Clear buttons
    start_button = ttk.Button(frame, text='Start', command=start_time)
    start_button.grid(row=1, column=0, padx=5, pady=2, sticky='ew')
    stop_button = ttk.Button(frame, text='Stop', command=stop_time)
    stop_button.grid(row=2, column=0, padx=5, pady=2, sticky='ew')
    run_astar_button = ttk.Button(frame, text='Run A*', command=on_run_astar)
    run_astar_button.grid(row=3, column=0, padx=5, pady=2, sticky='ew')
    clear_button = ttk.Button(frame, text='Clear Primitives', command=on_clear_visuals)
    clear_button.grid(row=4, column=0, padx=5, pady=2, sticky='ew')

    # Columns 1-2: Motion Primitive selection and enqueue
    mp_label = ttk.Label(frame, text='Motion Primitive:')
    mp_label.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky='w')

    _mp_var = tk.StringVar()
    _mp_combobox = ttk.Combobox(frame, textvariable=_mp_var, state='readonly', width=32)
    _mp_combobox.grid(row=2, column=1, columnspan=2, padx=5, pady=2, sticky='ew')
    _mp_combobox.bind('<<ComboboxSelected>>', on_mp_selected)
    _mp_var.trace_add('write', lambda *_: on_mp_selected())

    enqueue_button = ttk.Button(frame, text='Enqueue Selected', command=enqueue_selected_mp)
    enqueue_button.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky='ew')

    # Column 3: Checkbuttons
    _should_draw_path = tk.BooleanVar(value=True)
    _should_draw_cost = tk.BooleanVar(value=True)
    _should_draw_heur = tk.BooleanVar(value=True)
    _should_draw_collisions = tk.BooleanVar(value=True)

    path_cbox = ttk.Checkbutton(frame, text='Draw Path', variable=_should_draw_path, command=on_checkbox_toggle)
    path_cbox.grid(row=1, column=3, padx=5, pady=2, sticky='w')
    cost_cbox = ttk.Checkbutton(frame, text='Draw Costs', variable=_should_draw_cost, command=on_checkbox_toggle)
    cost_cbox.grid(row=2, column=3, padx=5, pady=2, sticky='w')
    heur_cbox = ttk.Checkbutton(frame, text='Draw Heuristics', variable=_should_draw_heur, command=on_checkbox_toggle)
    heur_cbox.grid(row=3, column=3, padx=5, pady=2, sticky='w')
    collide_cbox = ttk.Checkbutton(frame, text='Draw Collisions', variable=_should_draw_collisions, command=on_checkbox_toggle)
    collide_cbox.grid(row=4, column=3, padx=5, pady=2, sticky='w')

    populate_motion_primitives()

    return root_window, frame, widget
