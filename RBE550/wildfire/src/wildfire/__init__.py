import numpy as np

import valet.obstacle_field as obs
import valet.grid_sim as ui
import valet.planner as plan

from firetruck import Firetruck
from wumpus import Wumpus

OBS_INTACT = 1
OBS_BURNING = 2
OBS_EXT = 3
OBS_BURNED = 4

obstacle_states = [[], [], [], []]

def main() -> None:
    obs.FIELD_HEIGHT = 25
    obs.FIELD_WIDTH = 25
    obs._cell_width = 25
    obs._cell_height = 25
    ui.METERS_PER_CELL = 5.
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    obstacles = obs.populate_cells(0.1, True, _canvas)
    obstacle_states[OBS_INTACT].extend(obstacles.keys())

    # start and parking spots
    obs.clear_cell(0, 0)
    obs.clear_cell(0, 1)

    truck = Firetruck(0.5, 1, np.pi/2, turn_radius=13)
    wump = Wumpus(obs.FIELD_WIDTH - 1, obs.FIELD_HEIGHT - 1, obs.FIELD_WIDTH, obs.FIELD_HEIGHT, obs.obs_quadtree)
    _canvas.add_drawable(truck)
    _canvas.add_drawable(wump)

    def step_time_callback():
        truck.step(ui.time_rate / ui.fps)
        if truck.check_collision(obs.obs_quadtree):
            truck.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'truck crashed')
        wump.step(ui.time_rate / ui.fps)
        if wump.check_collision():
            wump.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'wumpus crashed?')
        ui.update_preview_pose()
        _canvas.redraw()

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()

    def run_planning():
        print(f'not implemented')

    ui.run_planning_callback = run_planning
    _root_window.mainloop()

if __name__ == '__main__':
    main()