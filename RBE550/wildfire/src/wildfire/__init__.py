import numpy as np
import fastquadtree as qt

import valet.obstacle_field as obs
import valet.grid_sim as ui
import valet.planner as plan

from firetruck import Firetruck
from wumpus import Wumpus
import shrubbery as shr

sim_time = 0.0

def main() -> None:
    obs.FIELD_HEIGHT = 25
    obs.FIELD_WIDTH = 25
    obs._cell_width = 25
    obs._cell_height = 25
    ui.METERS_PER_CELL = 5.
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
    _canvas.add_drawable(truck)
    _canvas.add_drawable(wump)

    truck.prm.visualizer.canvas = _canvas
    ui.checkbox_toggle_callback = truck.prm.visualizer.update_visualization

    def step_time_callback():
        global sim_time
        dt = ui.time_rate / ui.fps
        sim_time += dt
        truck.step(dt)
        if truck.check_collision(obs.obs_quadtree):
            truck.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'truck crashed')
        wump.step(dt)
        if wump.check_collision():
            wump.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'wumpus crashed?')
        shr.step(dt)
        ui.update_preview_pose()
        _canvas.redraw()

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()

    def run_planning():
        truck.prm.generate_roadmap()
        _canvas.redraw()

    ui.run_planning_callback = run_planning
    _root_window.mainloop()

if __name__ == '__main__':
    main()
