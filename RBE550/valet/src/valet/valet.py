import obstacle_field as obs
import numpy as np
import grid_sim as ui

METERS_PER_CELL = 3.0

def meters_to_cells(meters: float):
    return meters/METERS_PER_CELL

def cells_to_meters(cells: float):
    return cells * METERS_PER_CELL

class Vehicle(obs.DrawableGroup):
    def __init__(self, length, width, x = -1, y = -1, theta = -1):
        global _canvas
        # default starting location is top-left facing down
        # (clockwise is positive in screenspace because +y is down)
        if theta == -1:
            theta = np.pi/2
        if x == -1:
            x = abs(np.cos(theta))*np.ceil(length)/2.0 + abs(np.sin(theta))*np.ceil(width)/2.0
        if y == -1:
            y = abs(np.cos(theta))*np.ceil(width)/2.0 + abs(np.sin(theta))*np.ceil(length)/2.0

        obs.DrawableGroup.__init__(self, x, y, theta, True, _canvas)
        body = obs.DrawableRect(width=length, height=width, fill_color = (0, 1, 0), parent = self)
        self.add(body)

    def _make_controls(self):
        """
        Create control functions for the vehicle. Control functions take u, an n-tuple representing inputs, and C, a
        3 x m matrix of the vehicle's state (x, y, theta) and its (m - 1) differentials. There is one control function
        for each cell in C, and it returns the differential of its respective cell in one timestep given u. This results
        in a 3 x n x m array of control functions being stored in self.controls.
        """
        pass

    def _generate_motion_primitives(self, u_disc: list[np.ndarray]):
        # enumerate lowest, closest to 0, and highest inputs
        basic_inputs = np.empty(shape=(len(u_disc), 3))
        for i in range(len(u_disc)):
            basic_inputs[i][0] = u_disc[i][0]
            basic_inputs[i][1] = np.sort(np.abs(u_disc[i]))[0]
            basic_inputs[i][2] = u_disc[i][len(u_disc[i]) - 1]

        basic_input_perms = np.empty((0))
        for i in range(len(u_disc)):
            # TODO: generate all valid input permutations

        pass

    def discretize_c_space(self, obstacles: np.ndarray) -> np.ndarray:
        pass

if __name__ == '__main__':
    obs.FIELD_HEIGHT = 12
    obs.FIELD_WIDTH = 12
    obs._cell_width = 20
    obs._cell_height = 20
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    obs.populate_cells(0.2, True, _canvas)

    car = Vehicle(meters_to_cells(5.2), meters_to_cells(1.8))
    _canvas.add_drawable(car)

    def step_time_callback():
        car.translate_rel(1.0/ui.time_rate, 0)
        car.rotate(np.pi/(2*ui.time_rate))
        _canvas.redraw()

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()
    _root_window.mainloop()