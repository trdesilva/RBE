import functools
import tkinter as tk
from tkinter import ttk
import obstacle_field as obs
import time

time_rate = 10
_current_time_rate = 0
step_time_callback = None

def create_window():
    """
    Creates the Tkinter root, a ttk.Frame, and the GPU-accelerated ObstacleFieldWidget.
    """
    root_window = tk.Tk()
    frame = ttk.Frame(root_window, padding=10)
    frame.grid()
    canvas_size = obs.get_cell_bounds(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1)[1]
    widget = obs.ObstacleFieldWidget(frame, width=canvas_size[0], height=canvas_size[1])
    widget.grid(columnspan=4)
    obs._active_widget = widget

    def step_time():
        global _current_time_rate
        loop_start = time.perf_counter_ns()
        if step_time_callback is not None:
            step_time_callback()
        if _current_time_rate > 0:
            root_window.after(int(1000 / _current_time_rate - (time.perf_counter_ns() - loop_start) / 1000000), step_time)

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

    return root_window, frame, widget