import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fastquadtree as qt
import numpy as np
import rsplan
import valet.grid_sim as ui
import valet.obstacle_field as obs
import valet.planner as vp


@dataclass
class PRMNode:
    """A configuration vertex in the Probabilistic Road Map."""
    id: int
    pose: Tuple[float, float, float]  # (x, y, theta) in cells and radians


@dataclass
class PRMEdge:
    """A directed collision-free Reeds-Shepp path connecting two PRM nodes."""
    source_id: int
    target_id: int
    path: rsplan.primitives.Path
    cost: float


class PRMPlanner:
    """
    Probabilistic Road Map (PRM) planner for non-holonomic vehicles using Reeds-Shepp paths.

    Collision checking and Reeds-Shepp path generation are intrinsic to the planner:
    it maintains a reference to the obstacle quadtree and a test vehicle footprint
    so collision evaluations remain consistent across the lifetime of the roadmap.
    """

    def __init__(
        self,
        obs_quadtree: qt.RectQuadTree,
        body_prototype: obs.DrawableRect,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 25.0, 25.0),
        turn_radius: float = 1.0,
        step_size: float = 0.1,
        num_samples: int = 100,
        connection_radius: float = 8.0,
        k_neighbors: int = 6,
    ):
        self.obs_quadtree = obs_quadtree
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.turn_radius = turn_radius
        self.step_size = step_size
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.k_neighbors = k_neighbors

        # Independent test body for intrinsic collision verification along trajectories
        self._test_body = body_prototype.copy()
        self._test_body.parent = None

        # Roadmap representation
        self.nodes: List[PRMNode] = []
        self.adj_list: Dict[int, List[PRMEdge]] = {}

    def is_pose_collision_free(self, x: float, y: float, theta: float) -> bool:
        """Checks whether placing the vehicle footprint at (x, y, theta) causes a collision."""
        self._test_body.set_pose(x, y, theta)
        return not vp.check_collision(self._test_body, self.obs_quadtree)

    def is_path_collision_free(self, path: rsplan.primitives.Path) -> bool:
        """
        Samples the Reeds-Shepp path along its waypoints to ensure the vehicle footprint
        remains collision-free throughout the entire motion.
        """
        for wp in path.waypoints():
            self._test_body.set_pose(wp.x, wp.y, wp.yaw)
            if vp.check_collision(self._test_body, self.obs_quadtree):
                return False
        return True

    def plan_local_path(
        self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float],
        runway_length: float = 0.0,
    ) -> Optional[rsplan.primitives.Path]:
        """
        Attempts to compute a collision-free Reeds-Shepp path between two poses.
        Returns the rsplan.Path if collision-free, otherwise None.
        """
        try:
            candidate_path = rsplan.path(
                start_pose,
                end_pose,
                turn_radius=self.turn_radius,
                runway_length=runway_length,
                step_size=self.step_size,
            )
        except Exception:
            return None

        if self.is_path_collision_free(candidate_path):
            return candidate_path
        return None

    def sample_free_pose(self) -> Optional[Tuple[float, float, float]]:
        """Uniformly samples a collision-free configuration (x, y, theta) within field bounds."""
        min_x, min_y, max_x, max_y = self.bounds
        for _ in range(100):  # limit attempts per sample call
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            theta = random.uniform(-math.pi, math.pi)
            if self.is_pose_collision_free(x, y, theta):
                return (x, y, theta)
        return None

    def generate_roadmap(self) -> None:
        """
        Constructs the roadmap once before simulation starts:
        1. Samples random collision-free configurations.
        2. Finds candidate neighbor pairs within connection_radius.
        3. Connects neighbors with collision-free Reeds-Shepp paths.
        """
        self.nodes.clear()
        self.adj_list.clear()

        # Step 1: Sample vertices
        while len(self.nodes) < self.num_samples:
            pose = self.sample_free_pose()
            if pose is not None:
                node_id = len(self.nodes)
                self.nodes.append(PRMNode(id=node_id, pose=pose))
                self.adj_list[node_id] = []

        # Step 2 & 3: Connect k-nearest neighbors with Reeds-Shepp paths
        for i, node in enumerate(self.nodes):
            distances = []
            for j, other in enumerate(self.nodes):
                if i == j:
                    continue
                d = math.hypot(node.pose[0] - other.pose[0], node.pose[1] - other.pose[1])
                if d <= self.connection_radius:
                    distances.append((d, j, other))

            distances.sort(key=lambda t: t[0])
            for _, target_id, target_node in distances[: self.k_neighbors]:
                path = self.plan_local_path(node.pose, target_node.pose)
                if path is not None:
                    edge = PRMEdge(
                        source_id=node.id,
                        target_id=target_id,
                        path=path,
                        cost=path.total_length,
                    )
                    self.adj_list[node.id].append(edge)

    def query(
        self,
        start_pose: Tuple[float, float, float],
        goal_region: Tuple[float, float, float, float],
    ) -> Optional[List[rsplan.primitives.Path]]:
        """
        Connects start_pose to the roadmap and finds an A* path to a vertex within goal_region.
        (Implementation will be expanded in subsequent steps).
        """
        # TODO: Implement A* search on the PRM graph
        raise NotImplementedError("A* query on PRM roadmap will be implemented in the next step.")


# TODO: Design PRMVisualizer
# Responsibilities to address:
# - Track newly added vs. already drawn nodes and edges to avoid re-rendering entire sets
# - Decouple PRM debug visualization from Firetruck actor
class PRMVisualizer:
    """Placeholder for PRM visualization; will be designed after planner logic stabilizes."""
    def __init__(self, canvas=None):
        self.canvas = canvas


class Firetruck(obs.DrawableGroup):
    """
    Physical vehicle entity for the wildfire module.
    Responsible only for vehicle physical dimensions, kinematics, and moving the body rectangle.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        length: float = 4.9,
        width: float = 2.2,
        wheelbase: float = 3.0,
        turn_radius: Optional[float] = None
    ):
        super().__init__(x, y, theta)

        self.length_cells = ui.meters_to_cells(length)
        self.width_cells = ui.meters_to_cells(width)
        self.wheelbase_cells = ui.meters_to_cells(wheelbase)
        self.wheelbase_offset = self.wheelbase_cells / 2.0
        self.y -= self.wheelbase_offset

        # Physical visual body representation
        self.body = obs.DrawableRect(
            width=self.length_cells,
            height=self.width_cells,
            fill_color=(0.9, 0.0, 0.0),
            parent=self,
        )
        self.body.origin = (0.5 - self.wheelbase_offset / self.length_cells, 0.5)
        self.forward = obs.DrawableLineStrip((length / 2., 0), 0, 0, 0, color=(1.0, 0.1, 0.5), line_width=2.0,
                                             parent=self.body)
        self.add(self.body)
        #self.add(self.forward)

        # Kinematics
        self.turn_radius = (
            ui.meters_to_cells(turn_radius) if turn_radius is not None else self.wheelbase_cells * 1.5
        )

        # Motion execution queue
        self._waypoint_queue: List[rsplan.primitives.Waypoint] = []
        self._queue_idx: int = 0

    def check_collision(self, obs_quadtree: qt.RectQuadTree) -> bool:
        """Tests whether the current body pose collides with the obstacle field."""
        return vp.check_collision(self.body, obs_quadtree)

    def queue_paths(self, paths: List[rsplan.primitives.Path]) -> None:
        """Enqueues waypoints from a sequence of rsplan Paths for navigation."""
        self._waypoint_queue.clear()
        self._queue_idx = 0
        for p in paths:
            self._waypoint_queue.extend(p.waypoints())

    def step(self, dt: float, speed_cells_per_sec: float = ui.meters_to_cells(10)) -> None:
        """
        Advances the vehicle pose along the queued Reeds-Shepp waypoints.
        Moves only the body rectangle as it navigates.
        """
        if not self._waypoint_queue or self._queue_idx >= len(self._waypoint_queue):
            return

        # Advance waypoints proportionally to dt * speed
        advance_steps = max(1, int(speed_cells_per_sec * dt * 10))
        self._queue_idx = min(self._queue_idx + advance_steps, len(self._waypoint_queue) - 1)
        target_wp = self._waypoint_queue[self._queue_idx]
        self.set_pose(target_wp.x, target_wp.y, target_wp.yaw)
