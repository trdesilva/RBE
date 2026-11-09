from __future__ import annotations
import atexit
import csv
import datetime
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import fastquadtree as qt

import valet.obstacle_field as obs
import valet.grid_sim as ui
import valet.planner as plan

from firetruck import Firetruck
from wumpus import Wumpus
import shrubbery as shr

sim_time = 0.0
firetruck_cumulative_cpu_time = 0.0
wumpus_cumulative_cpu_time = 0.0
last_log_time = -np.inf

class CSVStatsLogger:
    """
    Consumes stats records in a dedicated background worker thread and appends them to disk.
    """
    def __init__(self, log_paths: list[str]):
        self.log_paths = log_paths
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        files = []
        writers = []
        for path in self.log_paths:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                f = open(path, 'w', newline='', encoding='utf-8')
                writer = csv.writer(f)
                writer.writerow(['process_time_s', 'sim_time_s', 'firetruck_cpu_time_s', 'wumpus_cpu_time_s'])
                f.flush()
                files.append(f)
                writers.append(writer)
            except Exception as e:
                print(f"Failed to open log file {path}: {e}")

        try:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                for writer in writers:
                    writer.writerow(row)
                for f in files:
                    f.flush()
                self.queue.task_done()
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

    def log(self, process_time: float, sim_time_val: float, ft_cpu: float, wump_cpu: float) -> None:
        self.queue.put((process_time, sim_time_val, ft_cpu, wump_cpu))

    def close(self) -> None:
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)


def main() -> None:
    global sim_time, firetruck_cumulative_cpu_time, wumpus_cumulative_cpu_time
    obs.FIELD_HEIGHT = 25
    obs.FIELD_WIDTH = 25
    obs._cell_width = 25
    obs._cell_height = 25
    ui.METERS_PER_CELL = 5.
    ui.time_rate = 8.
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    truck_spawn_cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    wumpus_spawn_cells = [(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1)]
    blocked_cells = truck_spawn_cells + wumpus_spawn_cells
    obstacle_to_cells, cell_to_obstacle = obs.populate_cells(0.1, True, _canvas, blocked_cells=blocked_cells)

    shr.init(obs.FIELD_WIDTH, obs.FIELD_HEIGHT, cell_to_obstacle, obstacle_to_cells)

    for rect in shr.get_all_obstacle_rects():
        _canvas.add_drawable(rect)

    for region in shr.get_all_ext_regions():
        _canvas.add_drawable(region)

    truck = Firetruck(0.5, 1, np.pi/2, turn_radius=13, obs_quadtree=obs.obs_quadtree)
    wump = Wumpus(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1, obs.FIELD_WIDTH, obs.FIELD_HEIGHT, obs.obs_quadtree)
    wump.truck = truck
    _canvas.add_drawable(truck)
    _canvas.add_drawable(wump)
    if truck.default_goal is not None:
        _canvas.add_drawable(truck.default_goal)

    truck.prm.visualizer.canvas = _canvas
    ui.checkbox_toggle_callback = truck.prm.visualizer.update_visualization

    # Initialize CSV logger
    now = datetime.datetime.now()
    timestamp = now.strftime("%H_%M")
    filename = f"wildfire_stats_{timestamp}.csv"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    wildfire_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    repo_root = os.path.abspath(os.path.join(wildfire_root, "..", ".."))
    log_paths = list(dict.fromkeys([
        os.path.join(repo_root, "logs", filename),
        os.path.join(wildfire_root, "logs", filename),
    ]))
    csv_logger = CSVStatsLogger(log_paths)
    atexit.register(csv_logger.close)

    # Statistics UI frame & Read-only Textboxes
    stats_frame = ttk.LabelFrame(_frame, text="Simulation Statistics", padding=5)
    stats_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky='ew')

    ttk.Label(stats_frame, text="Simulation Time:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
    sim_time_var = tk.StringVar(value="0.00 s")
    sim_time_entry = ttk.Entry(stats_frame, textvariable=sim_time_var, state='readonly', width=12)
    sim_time_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')

    ttk.Label(stats_frame, text="Unburnt Obstacles:").grid(row=0, column=2, padx=5, pady=2, sticky='w')
    unburnt_obs_var = tk.StringVar(value=str(shr.get_unburnt_obstacle_count()))
    unburnt_obs_entry = ttk.Entry(stats_frame, textvariable=unburnt_obs_var, state='readonly', width=12)
    unburnt_obs_entry.grid(row=0, column=3, padx=5, pady=2, sticky='w')

    ttk.Label(stats_frame, text="Firetruck Score:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
    ft_score_var = tk.StringVar(value="0")
    ft_score_entry = ttk.Entry(stats_frame, textvariable=ft_score_var, state='readonly', width=12)
    ft_score_entry.grid(row=1, column=1, padx=5, pady=2, sticky='w')

    ttk.Label(stats_frame, text="Wumpus Score:").grid(row=1, column=2, padx=5, pady=2, sticky='w')
    wumpus_score_var = tk.StringVar(value="0")
    wumpus_score_entry = ttk.Entry(stats_frame, textvariable=wumpus_score_var, state='readonly', width=12)
    wumpus_score_entry.grid(row=1, column=3, padx=5, pady=2, sticky='w')

    def step_time_callback():
        global sim_time, firetruck_cumulative_cpu_time, wumpus_cumulative_cpu_time, last_log_time
        if sim_time >= 3600:
            proc_time = time.perf_counter()
            csv_logger.log(proc_time, sim_time, firetruck_cumulative_cpu_time, wumpus_cumulative_cpu_time)
            last_log_time = sim_time
            print(f'Simulation ended at {sim_time}; A WINNER IS {"WUMPUS" if shr._wumpus_score > shr._firetruck_score else
            "FIRETRUCK" if shr._firetruck_score > shr._wumpus_score else
            "YOU?" }')
            ui._current_time_rate = 0

        dt = ui.time_rate / ui.fps
        sim_time += dt

        # Firetruck step timed, excluding PRMVisualizer time
        t_ft0 = time.perf_counter()
        vis_t0 = truck.prm.visualizer.total_time
        truck.step(dt)
        vis_elapsed = truck.prm.visualizer.total_time - vis_t0
        firetruck_cumulative_cpu_time += max(0.0, (time.perf_counter() - t_ft0) - vis_elapsed)

        if truck.check_collision(obs.obs_quadtree):
            truck.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'truck crashed')

        # Wumpus step timed
        t_wump0 = time.perf_counter()
        wump.step(dt)
        wumpus_cumulative_cpu_time += max(0.0, time.perf_counter() - t_wump0)

        if wump.check_collision():
            wump.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'wumpus crashed?')

        shr.step(dt)

        # Log statistics to background thread
        if sim_time - 10 > last_log_time:
            proc_time = time.perf_counter()
            csv_logger.log(proc_time, sim_time, firetruck_cumulative_cpu_time, wumpus_cumulative_cpu_time)
            last_log_time = sim_time

        # Update UI statistics textboxes
        sim_time_var.set(f"{sim_time:.2f} s")
        ft_score_var.set(str(shr.get_firetruck_score()))
        wumpus_score_var.set(str(shr.get_wumpus_score()))
        unburnt_obs_var.set(str(shr.get_unburnt_obstacle_count()))

        ui.update_preview_pose()
        _canvas.redraw()

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()

    def run_planning():
        global firetruck_cumulative_cpu_time
        t_ft0 = time.perf_counter()
        vis_t0 = truck.prm.visualizer.total_time
        truck.prm.generate_roadmap()
        vis_elapsed = truck.prm.visualizer.total_time - vis_t0
        firetruck_cumulative_cpu_time += max(0.0, (time.perf_counter() - t_ft0) - vis_elapsed)
        _canvas.redraw()

    ui.run_planning_callback = run_planning

    def on_window_close():
        csv_logger.close()
        _root_window.destroy()

    _root_window.protocol("WM_DELETE_WINDOW", on_window_close)
    _root_window.mainloop()

if __name__ == '__main__':
    main()
