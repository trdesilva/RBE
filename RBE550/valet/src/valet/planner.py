from dataclasses import dataclass
import numpy as np
import fastquadtree as qt

try:
    from valet import obstacle_field as obs
    from valet import grid_sim as ui
except ImportError:
    try:
        from . import obstacle_field as obs
        from . import grid_sim as ui
    except ImportError:
        import obstacle_field as obs
        import grid_sim as ui

_canvas = None

METERS_PER_CELL = 3.0

def meters_to_cells(meters: float):
    return meters/METERS_PER_CELL

def cells_to_meters(cells: float):
    return cells * METERS_PER_CELL

@dataclass
class InputFrame:
    u: np.ndarray | tuple[float]
    dt: float = 1

@dataclass
class VehicleState:
    x: float
    y: float
    theta: float

@dataclass
class StateFrame:
    state: VehicleState
    dt: float = 1

@dataclass
class MotionPrimitive:
    inputs: list[InputFrame]
    states: list[StateFrame]
    dt: float= 1

class Vehicle(obs.DrawableGroup):
    def __init__(self, length, width, x = -1, y = -1, theta = -1, controls = 3, state_shape = (3, 1), canvas = None):
        global _canvas
        parent_canvas = canvas if canvas is not None else _canvas
        # default starting location is top-left facing down
        # (clockwise is positive in screenspace because +y is down)
        if theta == -1:
            theta = np.pi/2
        if x == -1:
            x = 15 #abs(np.cos(theta))*np.ceil(length)/2.0 + abs(np.sin(theta))*np.ceil(width)/2.0
        if y == -1:
            y = 15 #abs(np.cos(theta))*np.ceil(width)/2.0 + abs(np.sin(theta))*np.ceil(length)/2.0

        obs.DrawableGroup.__init__(self, x, y, theta, True, parent_canvas)
        self.body = obs.DrawableRect(width=length, height=width, fill_color = (0, 1, 0), parent = self)
        self.add(self.body)

        self.state_shape = state_shape
        self.controls = controls
        self._generate_default_motion_primitives(self._get_default_u_disc())

        self.mp_queue = [] # MotionPrimitives to execute
        self.mp_frame = 0 # which frame in mp_queue[0] is active
        self.mp_time = 0 # how long mp_frame has been active

    def eval_controls(self, u, s, dt = 1., apply = False):
        """
        Evaluate the control function for the vehicle. The base implementation has three inputs, each corresponding to
        one axis of the vehicle's configuration, and has no differential constraints.
        :param u: an n-tuple representing inputs
        :param s: 3 x m matrix of the vehicle's configuration (x, y, theta) and its (m - 1) differentials (shape should
        match self.state_shape)
        :param dt: the length of the timestep to apply u for (default 1)
        :param apply: whether to apply the result to the vehicle's state (default False)
        :return: a dictionary containing the changes to the vehicle's state that applying the action u would cause. Keys
        are field names in the Vehicle class (x, y, theta) or its child classes.
        """
        result = VehicleState(u[0] * dt,
                              u[1] * dt,
                              u[2] * dt)
        if apply:
            self.x += result.x
            self.y += result.y
            self.theta += result.theta
        return result

    def _generate_default_motion_primitives(self, u_disc: list[np.ndarray]):
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
        assert self.controls == len(u_disc), f'u_disc has incorrect number of controls: {len(u_disc)}'
        basic_inputs = []
        for i in range(len(u_disc)):
            u_disc[i].sort()
            basic_inputs.append(np.array([
                u_disc[i][0],
                np.sort(np.abs(u_disc[i]))[0],
                u_disc[i][len(u_disc[i]) - 1]
            ]))

        basic_input_perms = _permute_vectors(basic_inputs)
        self.motion_primitives = []
        for i in range(basic_input_perms.shape[1]):
            result = self.eval_controls(basic_input_perms[:, i], VehicleState(0, 0, 0))
            mp = MotionPrimitive([InputFrame(basic_input_perms[:, i], 1)],
                                 [StateFrame(result, 1)])
            self.motion_primitives.append(mp)
        print(self.motion_primitives)

    def queue_motion(self, mp: MotionPrimitive):
        self.mp_queue.append(mp)

    def run_queued_motions(self, dt = 1.):
        if len(self.mp_queue) < 1 or dt <= 0:
            return

        # if time advances less than a frame, interpolate
        frame = self.mp_queue[0].states[self.mp_frame]
        if self.mp_time + dt < frame.dt:
            self.mp_time += dt
            self.translate_rel(frame.state.x * dt / frame.dt, frame.state.y * dt / frame.dt)
            self.rotate(frame.state.theta * dt / frame.dt)
            return

        # time advances at least one frame, so skip to the next one
        time_skipped = frame.dt - self.mp_time
        self.mp_time = 0
        self.translate_rel(frame.state.x * time_skipped / frame.dt, frame.state.y * time_skipped / frame.dt)
        self.rotate(frame.state.theta * time_skipped / frame.dt)

        self.mp_frame += 1
        if self.mp_frame >= len(self.mp_queue[0].states):
            self.mp_frame = 0
            self.mp_queue.pop(0)
        self.run_queued_motions(dt - time_skipped)


    def _get_default_u_disc(self):
        return [np.linspace(-1, 1, 5),  # u_x
                np.linspace(-1, 1, 5),  # u_y
                np.linspace(-np.pi / 2, np.pi / 2, 5)] # u_theta

    def _calculate_c_obs(self, obstacles: np.ndarray) -> np.ndarray:
        pass

    def check_collision(self, obs_quadtree: qt.RectQuadTree):
        body_verts = self.body.get_world_vertices()
        body_aabb = self.body.get_aabb()
        nearest_cells = obs_quadtree.query(body_aabb)
        for cell in nearest_cells:
            # query returns (id, minx, miny, maxx, maxy)
            # check if corners of the cell are in the body or vice versa
            minx = cell[1]
            miny = cell[2]
            maxx = cell[3]
            maxy = cell[4]
            cell_verts_inside = np.zeros((4,4), dtype=bool)

            for i in range(4):
                bv = body_verts[:, i]
                if minx <= bv[0] <= maxx and miny <= bv[1] <= maxy:
                    # body corner inside cell
                    return True
                bv_next = body_verts[:, (i + 1) % 4]
                # check all cell vertices for being to the left of edge (bv, bv_next)
                for j in range(4):
                    cell_vert = np.array([cell[2*int(j/2) + 1], cell[2*(j % 2) + 2]]).T
                    cell_verts_inside[i, j] = np.linalg.det(np.vstack((np.column_stack((bv, bv_next, cell_vert)),
                                                                      np.ones((1, 3))))) <= 0

            # if any cell vertex is inside (left of) all body edges
            if np.any(np.all(cell_verts_inside, axis=0)):
                return True

        return False


class Robot(Vehicle):
    def __init__(self, canvas = None):
        Vehicle.__init__(self, length=meters_to_cells(.7), width=meters_to_cells(.57), controls=2, canvas=canvas)
        self.children[0].fill_color = (0, 0.8, 0)
        self.children[0].border_color = (0, 0, 0, 0.5)

        self.generate_special_motion_primitives()

    def eval_controls(self, u, s, dt = 1, apply = False):
        result = VehicleState(u[0] * np.cos(s.theta) * dt,
                              u[0] * np.sin(s.theta) * dt,
                              u[1] * dt)
        if apply and s == self:
            self.x += result.x
            self.y += result.y
            self.theta += result.theta
        elif apply:
            print(f'eval_controls can\'t apply input to non-self state {s}')
        return result

    def generate_special_motion_primitives(self):
        u_disc = self._get_default_u_disc()
        input_perms = _permute_vectors([u_disc[0], np.array([u_disc[1][1], u_disc[1][2]])])
        for i in range(input_perms.shape[1]):
            result = self.eval_controls(input_perms[:, i], VehicleState(0, 0, 0))
            mp = MotionPrimitive([InputFrame(input_perms[:, i], 1)],
                                 [StateFrame(result, 1)])
            self.motion_primitives.append(mp)

    def _get_default_u_disc(self):
        return [np.array([-1, 0, 1]),
                np.linspace(-np.pi/2, np.pi/2, 5)]

def _permute_vectors(vecs: list[np.ndarray]) -> np.ndarray:
    """
    Permute arbitrary sets of vector components.
    :param vecs: A list of 1D numpy arrays, where the nth array contains all
                 possible values for the nth component of a result vector.
    :return: A 2D numpy array of shape (len(vecs), N) where each column is a
             permuted vector of cardinality len(vecs).
    """
    if isinstance(vecs, np.ndarray):
        if vecs.ndim == 1:
            vecs = [vecs]
        elif vecs.ndim == 2:
            vecs = [row for row in vecs]
        else:
            raise ValueError(f"Expected 1D or 2D array, got {vecs.ndim}D array")
    if len(vecs) == 0:
        return np.empty((0, 0))
    if any(len(v) == 0 for v in vecs):
        return np.empty((len(vecs), 0))
    vecs = [np.asarray(v).ravel() for v in vecs]
    grids = np.meshgrid(*vecs, indexing='ij')
    return np.stack(grids, axis=0).reshape(len(vecs), -1)

if __name__ == '__main__':
    obs.FIELD_HEIGHT = 30
    obs.FIELD_WIDTH = 30
    obs._cell_width = 25
    obs._cell_height = 25
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    obs.populate_cells(0.2, True, _canvas)

    #car = Vehicle(meters_to_cells(5.2), meters_to_cells(1.8))
    robot = Robot()
    _canvas.add_drawable(robot)
    print(f'showing {len(robot.motion_primitives)} motion primitives')

    counter = 0
    def step_time_callback():
        global counter, robot
        if len(robot.mp_queue) < 1:
            motion = robot.motion_primitives[counter]
            print(f'{counter}: {motion}')
            robot.queue_motion(motion)
            counter = (counter + 1) % len(robot.motion_primitives)
        robot.run_queued_motions(ui.time_rate/ui.fps)
        if robot.check_collision(obs.obs_quadtree):
            robot.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'robot crashed')
        _canvas.redraw()

    ui.step_time_callback = step_time_callback

    obs.draw_grid()
    _frame.pack()
    _root_window.mainloop()
