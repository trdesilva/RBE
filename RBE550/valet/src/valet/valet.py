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
            x = 15 #abs(np.cos(theta))*np.ceil(length)/2.0 + abs(np.sin(theta))*np.ceil(width)/2.0
        if y == -1:
            y = 15 #abs(np.cos(theta))*np.ceil(width)/2.0 + abs(np.sin(theta))*np.ceil(length)/2.0

        obs.DrawableGroup.__init__(self, x, y, theta, True, _canvas)
        body = obs.DrawableRect(width=length, height=width, fill_color = (0, 1, 0), parent = self)
        self.add(body)

        self._make_controls()
        self._generate_motion_primitives([np.linspace(-1, 1, 5), # u_x
                                          np.linspace(-1, 1, 5), # u_y
                                          np.linspace(-np.pi/2, np.pi/2, 5)]) # u_theta

    def eval_controls(self, u, s, dt = 1., apply = False):
        result = {'x': u[0] * dt,
                  'y': u[1] * dt,
                  'theta': u[2] * dt}
        if apply:
            self.x += result['x']
            self.y += result['y']
            self.theta += result['theta']
        return result

    def _make_controls(self):
        """
        Create control functions for the vehicle. Control functions take u, an n-tuple representing inputs, and s, a
        3 x m matrix of the vehicle's configuration (x, y, theta) and its (m - 1) differentials. There is one control
        function for each cell in C, and it returns the differential of its respective cell in one timestep given u.
        This results in a 3 x n x m array of control functions being stored in self.controls. The base implementation
        has three inputs, each corresponding to one axis of the vehicle's configuration, and has no differential
        constraints.
        """
        self.controls = np.array([[lambda u, s: u[0], lambda u, s: 0, lambda u, s: 0],
                                  [lambda u, s: 0, lambda u, s: u[1], lambda u, s: 0],
                                  [lambda u, s: 0, lambda u, s: 0, lambda u, s: u[2]]])

    def _generate_motion_primitives(self, u_disc: list[np.ndarray]):
        """
        Generate a set of motion primitives for discretizing the path search space and store them in
        self.motion_primitives. This assumes that self.controls is set correctly (see _make_controls). The base
        implementation here samples the action space at the minimum, maximum, and minimum-magnitude (i.e. closest to 0)
        values; calculates all permutations of these sampled actions; and for each permutation, stores the result of
        applying the permuted action to the vehicle for one time step.
        :param u_disc: a discretized sample of the action space. Each array in u_disc represents the valid range of
        values for one component of the action vector that serves as an input to the control functions.
        :return: None
        """
        # enumerate lowest, closest to 0, and highest inputs
        assert self.controls.shape[0] == 3, f'controls has incorrect number of rows: {self.controls.shape[0]}'
        assert self.controls.shape[1] == len(u_disc), f'u_disc has incorrect number of controls: {len(u_disc)}'
        basic_inputs = np.empty(shape=(len(u_disc), 3))
        for i in range(len(u_disc)):
            u_disc[i].sort()
            basic_inputs[i][0] = u_disc[i][0]
            basic_inputs[i][1] = np.sort(np.abs(u_disc[i]))[0]
            basic_inputs[i][2] = u_disc[i][len(u_disc[i]) - 1]

        basic_input_perms = _permute_vectors(basic_inputs)
        self.motion_primitives = []
        for i in range(basic_input_perms.shape[1]):
            result = self.eval_controls(basic_input_perms[:, i], self)
            self.motion_primitives.append([result['x'], result['y'], result['theta'], 1])
        print(self.motion_primitives)

    def calculate_c_obs(self, obstacles: np.ndarray) -> np.ndarray:
        pass

def _permute_vectors(mat: np.ndarray):
    if len(mat.shape) == 1 or (len(mat.shape) == 2 and mat.shape[0] == 1):
        return mat
    elif len(mat.shape) == 2:
        next = _permute_vectors(mat[1:])
        result = np.empty((mat.shape[0], 0))
        for i in range(mat.shape[1]):
            result = np.hstack((result,
                                np.concat((np.ones((1, next.shape[1])) * mat[0][i], next))))
        return result
    else:
        raise ValueError

if __name__ == '__main__':
    obs.FIELD_HEIGHT = 30
    obs.FIELD_WIDTH = 30
    obs._cell_width = 20
    obs._cell_height = 20
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    obs.populate_cells(0.2, True, _canvas)

    car = Vehicle(meters_to_cells(5.2), meters_to_cells(1.8))
    _canvas.add_drawable(car)
    print(f'showing {len(car.motion_primitives)} motion primitives')

    counter = 0
    def step_time_callback():
        global counter
        print(counter)
        motion = car.motion_primitives[counter]
        car.eval_controls(motion[0:3], car, dt = motion[3], apply = True)
        _canvas.redraw()
        counter = (counter + 1) % len(car.motion_primitives)

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()
    _root_window.mainloop()