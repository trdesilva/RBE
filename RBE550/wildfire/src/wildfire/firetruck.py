import astar
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import fastquadtree as qt
import numpy as np
import rsplan
from astar import AStar

import valet.grid_sim as ui
import valet.obstacle_field as obs
import valet.planner as vp

import shrubbery as shr

@dataclass
class PRMNode:
    """A configuration vertex in the Probabilistic Road Map."""
    id: int
    pose: Tuple[float, float, float]  # (x, y, theta) in cells and radians
    def __hash__(self) -> int:
        return hash(self.id) + 11*hash(self.pose)

@dataclass
class PRMEdge:
    """A directed collision-free Reeds-Shepp path connecting two PRM nodes."""
    source_id: int
    target_id: int
    path: rsplan.primitives.Path
    cost: float
    line_strip: Optional[obs.DrawableLineStrip] = None

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id))

@dataclass
class AStarData:
    data: PRMNode|shr.ExtRegion
    is_goal: bool = False
    def __hash__(self) -> int:
        return hash(self.data) + 101*hash(self.is_goal)


class PRMVisualizer:
    """
    Manages debug visualization of PRM roadmap edges and A* query exploration
    using DrawableLineStrips.
    """
    def __init__(self, canvas=None):
        self.canvas = canvas
        self.edges_with_strips: List[PRMEdge] = []
        self.explored_edges: Set[PRMEdge] = set()
        self.cost_values: Dict[PRMEdge, float] = {}
        self.heur_values: Dict[int, float] = {}
        self.optimal_path_edges: Set[PRMEdge] = set()
        self.gradient_values: List[float] = []
        self.value_cache: Optional[np.ndarray] = None

    def add_edge(self, edge: PRMEdge, strip: obs.DrawableLineStrip) -> None:
        """Registers a new roadmap edge and adds its line strip to the canvas."""
        self.edges_with_strips.append(edge)
        canvas = self.canvas if self.canvas is not None else obs._active_widget
        if canvas is not None and getattr(canvas, 'drawables', None) is not None:
            if strip not in canvas.drawables:
                canvas.add_drawable(strip)

    def update_cost_display(self, edge: PRMEdge, cost: float) -> None:
        """Records the cost for an explored edge and updates its visual representation."""
        self.cost_values[edge] = cost
        self.explored_edges.add(edge)
        if edge.line_strip is not None and (ui.should_draw_cost() or ui.should_draw_heur()):
            heur = self.heur_values.get(edge.target_id, 0.0) if ui.should_draw_heur() else 0.0
            c = cost if ui.should_draw_cost() else 0.0
            val = c + heur
            edge.line_strip.color = self.get_gradient_color(val)
            edge.line_strip.line_width = 1.5

    def update_heuristic_display(self, node_id: int, heur: float) -> None:
        """Records the heuristic estimate for a PRM node."""
        self.heur_values[node_id] = heur

    def set_optimal_path(self, edges: List[PRMEdge]) -> None:
        """Sets the optimal path edges to highlight upon A* query completion."""
        self.optimal_path_edges = set(edges)

    def get_gradient_color(self, value: float) -> str:
        """Computes HSV gradient color for an active value using a dynamic histogram."""
        self.gradient_values.append(float(value))
        if len(self.gradient_values) > 1:
            self.value_cache = np.histogram_bin_edges(np.sort(self.gradient_values), 10)
            normalized = np.searchsorted(self.value_cache, value) / self.value_cache.size
        else:
            normalized = 0.0
        return obs.get_hsv_gradient(float(np.clip(normalized, 0.0, 1.0)), 1.0, alpha=0.5)

    def clear(self) -> None:
        """Clears all drawn line strips from the canvas and resets state."""
        canvas = self.canvas if self.canvas is not None else obs._active_widget
        if canvas is not None and getattr(canvas, 'drawables', None) is not None:
            for edge in self.edges_with_strips:
                if edge.line_strip is not None and edge.line_strip in canvas.drawables:
                    canvas.remove_drawable(edge.line_strip)
        self.edges_with_strips.clear()
        self.explored_edges.clear()
        self.cost_values.clear()
        self.heur_values.clear()
        self.optimal_path_edges.clear()
        self.gradient_values.clear()
        self.value_cache = None

    def update_visualization(self) -> None:
        """Re-evaluates visibility and gradient colors when checkboxes are toggled or display updates."""
        show_path = ui.should_draw_path()
        show_cost = ui.should_draw_cost()
        show_heur = ui.should_draw_heur()

        active_vals = []
        edge_vals = {}
        for edge in self.explored_edges:
            c = self.cost_values.get(edge, 0.0)
            h = self.heur_values.get(edge.target_id, 0.0)
            if show_cost and show_heur:
                v = c + h
            elif show_cost:
                v = c
            elif show_heur:
                v = h
            else:
                v = None
            edge_vals[edge] = v
            if v is not None:
                active_vals.append(v)

        edges = np.histogram_bin_edges(np.sort(active_vals), 10) if len(active_vals) > 1 else None

        for edge in self.edges_with_strips:
            strip = edge.line_strip
            if strip is None:
                continue

            strip.visible = show_path

            if edge in self.optimal_path_edges:
                strip.color = '#00ff00'
                strip.line_width = 3.0
            elif edge in self.explored_edges and (show_cost or show_heur):
                strip.line_width = 1.5
                v = edge_vals.get(edge)
                if v is not None and edges is not None and edges.size > 0:
                    norm = np.searchsorted(edges, v) / edges.size
                    strip.color = obs.get_hsv_gradient(float(np.clip(norm, 0.0, 1.0)), 1.0, alpha=0.5)
                else:
                    strip.color = obs.get_hsv_gradient(0.0, 1.0, alpha=0.5)
            else:
                strip.color = (0.5, 0.5, 0.5, 0.5)
                strip.line_width = 1.0

        canvas = self.canvas if self.canvas is not None else obs._active_widget
        if canvas is not None and getattr(canvas, 'context_created', False):
            canvas.redraw()


class PRMPlanner(AStar):
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
        self.node_quadtree = qt.QuadTreeObjects(bounds, capacity=num_samples)
        self.visualizer: PRMVisualizer = PRMVisualizer()

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
        self.node_quadtree.clear()
        if self.visualizer is not None:
            self.visualizer.clear()

        # Step 1: Sample vertices
        print(f'PRM sampling {self.num_samples} vertices...')
        while len(self.nodes) < self.num_samples:
            pose = self.sample_free_pose()
            if pose is not None:
                node_id = len(self.nodes)
                new_node = PRMNode(id=node_id, pose=pose)
                self.nodes.append(new_node)
                self.adj_list[node_id] = []
                self.node_quadtree.insert((pose[0], pose[1]), new_node)

        # Step 2 & 3: Connect k-nearest neighbors with Reeds-Shepp paths
        print(f'PRM connecting {len(self.nodes)} nodes...')
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
                    strip = obs.DrawableLineStrip(
                        vertices=[(wp.x, wp.y) for wp in path.waypoints()],
                        color=(0.5, 0.5, 0.5, 0.5),
                        line_width=1.0,
                        visible=ui.should_draw_path(),
                    )
                    edge = PRMEdge(
                        source_id=node.id,
                        target_id=target_id,
                        path=path,
                        cost=path.total_length,
                        line_strip=strip,
                    )
                    self.adj_list[node.id].append(edge)
                    if self.visualizer is not None:
                        self.visualizer.add_edge(edge, strip)
            if (i + 1) % (self.num_samples / 10) < 1:
                print(f'PRM connected {i + 1} nodes')

        if self.visualizer is not None:
            self.visualizer.update_visualization()

    def query(
        self,
        start_pose: Tuple[float, float, float],
        goal_region: shr.ExtRegion,
    ) -> Optional[List[rsplan.primitives.Path]]:
        """
        Connects start_pose to the roadmap and finds an A* path to a vertex within goal_region.
        """
        if len(self.nodes) == 0:
            self.generate_roadmap()
        nearest_nodes = self.node_quadtree.nearest_neighbors((start_pose[0], start_pose[1]), 20)
        valid_connections = []
        for node in nearest_nodes:
            path = self.plan_local_path(start_pose, node.obj.pose)
            if path is not None:
                valid_connections.append((path, node.obj))
                if len(valid_connections) >= 10:
                    break
        if len(valid_connections) > 0:
            valid_connections.sort(key=lambda p: p[0].total_length)
            start_path = valid_connections[0][0]
            first_node = valid_connections[0][1]
            path = [start_path]
            rest = self.astar(AStarData(first_node), AStarData(goal_region, True))
            if rest is not None:
                node_list = list(rest)
                optimal_edges = []
                for a, b in zip(node_list[:-1], node_list[1:]):
                    if not a.is_goal and not b.is_goal:
                        prm_a = a.data
                        prm_b = b.data
                        for edge in self.adj_list.get(prm_a.id, []):
                            if edge.target_id == prm_b.id:
                                optimal_edges.append(edge)
                                path.append(edge.path)
                                break
                if self.visualizer is not None:
                    self.visualizer.set_optimal_path(optimal_edges)
                    self.visualizer.update_visualization()
                return path
            else:
                print(f'ExtRegion {goal_region} is not reachable from {start_pose}.')
                return None
        else:
            print(f'No valid connections found for {start_pose}')
            return None

    """
    A* Impl
    """
    def path_neighbors(self, node: astar.SearchNode[AStarData]):
        if node.data.is_goal:
            return []
        prm_node = node.data.data
        return [AStarData(self.nodes[edge.target_id]) for edge in self.adj_list[prm_node.id]]

    def heuristic_cost_estimate(self, current: AStarData, goal: AStarData) -> float:
        if not current.is_goal and goal.is_goal:
            result = goal.data.get_minimum_distance(current.data.pose[0], current.data.pose[1])
            if self.visualizer is not None:
                self.visualizer.update_heuristic_display(current.data.id, result)
            return result
        return np.inf

    def path_distance_between(self, n1: astar.SearchNode[AStarData], n2: astar.SearchNode[AStarData]) -> float:
        if n1.data.is_goal or n2.data.is_goal:
            return np.inf
        prm_n1 = n1.data.data
        prm_n2 = n2.data.data
        for edge in self.adj_list[prm_n1.id]:
            if edge.target_id == prm_n2.id:
                if self.visualizer is not None:
                    self.visualizer.update_cost_display(edge, edge.cost)
                return edge.cost
        return np.inf

    def is_goal_reached(self, current: AStarData, goal: AStarData):
        if not current.is_goal and goal.is_goal:
            return goal.data.contains(current.data.pose[0], current.data.pose[1])
        return False


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
        turn_radius: Optional[float] = None,
        speed: float = 10,
        obs_quadtree: qt.RectQuadTree|None = None
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
        self.speed = ui.meters_to_cells(speed)

        self.prm = PRMPlanner(obs_quadtree, self.body, (0, 0, 25, 25), self.turn_radius, 0.1, 25*25, 5, 5)

        # Motion execution queue
        self._segment_queue = []
        self._waypoint_queue: List[rsplan.primitives.Waypoint] = []
        self._step_dist = 0

    def check_collision(self, obs_quadtree: qt.RectQuadTree) -> bool:
        """Tests whether the current body pose collides with the obstacle field."""
        return vp.check_collision(self.body, obs_quadtree)

    def queue_paths(self, paths: List[rsplan.primitives.Path]) -> None:
        """Enqueues waypoints from a sequence of rsplan Paths for navigation."""
        for path in paths:
            self._segment_queue.extend(path.segments)
        if len(self._segment_queue) > 0:
            self._waypoint_queue = self._segment_queue[0].waypoints()

    def replan(self):
        timer_pairs = [(coords, time) for coords, time in shr.get_burn_timers().items()]
        timer_pairs.sort(key=lambda pair: pair[1])
        best = None
        for timer in timer_pairs:
            if best is None or timer[1] < best[1] or best[1] < 5 < timer[1]:
                best = timer
        if best is not None:
            print(f'Firetruck pathing to {best[0]} ({best[1]:0.2f}s remaining)')
            goal = shr.get_ext_region(*best[0])
            path = self.prm.query((self.x, self.y, self.theta), goal)
            if path is not None:
                print(f'Firetruck found path to {best[0]}')
                self.queue_paths(path)
            else:
                print(f'Firetruck pathing failed')

    def step(self, dt: float) -> None:
        """
        Advances the vehicle pose along the queued Reeds-Shepp waypoints.
        Moves only the body rectangle as it navigates.
        """
        if len(self._segment_queue) == 0:
            self.replan()

        if len(self._segment_queue) == 0:
            return

        if dt <= 0 or self.speed <= 0 or not self._waypoint_queue:
            return

        segment = self._segment_queue[0]
        seg_len = abs(segment.length)
        if seg_len == 0 or len(self._waypoint_queue) == 0:
            return

        distance_to_cover = dt * self.speed
        num_intervals = max(len(self._waypoint_queue) - 1, 1)
        distance_per_waypoint = seg_len / num_intervals
        if distance_per_waypoint == 0:
            return

        rem = self._step_dist % distance_per_waypoint
        dist_to_next = distance_per_waypoint - rem if rem > 1e-9 else distance_per_waypoint

        while distance_to_cover >= dist_to_next - 1e-9:
            distance_to_cover = max(0.0, distance_to_cover - dist_to_next)
            self._step_dist += dist_to_next
            if self._step_dist >= seg_len - 1e-9:
                self._step_dist -= seg_len
                if self._step_dist < 0:
                    self._step_dist = 0.0
                last_wp = self._waypoint_queue[-1]
                self._segment_queue.pop(0)
                if not self._segment_queue:
                    self._waypoint_queue = []
                    self.set_pose(last_wp.x, last_wp.y, last_wp.yaw)
                    return
                segment = self._segment_queue[0]
                self._waypoint_queue = segment.waypoints() if callable(getattr(segment, "waypoints", None)) else getattr(segment, "waypoints", [])
                if not self._waypoint_queue:
                    return
                seg_len = abs(segment.length)
                if seg_len == 0:
                    return
                num_intervals = max(len(self._waypoint_queue) - 1, 1)
                distance_per_waypoint = seg_len / num_intervals
                if distance_per_waypoint == 0:
                    return

            segment_progress = self._step_dist / seg_len
            idx = min(int(segment_progress * num_intervals), len(self._waypoint_queue) - 1)
            wp = self._waypoint_queue[idx]
            self.set_pose(wp.x, wp.y, wp.yaw)
            dist_to_next = distance_per_waypoint

        self._step_dist += distance_to_cover
        segment_progress = self._step_dist / seg_len
        waypoint_progress = (self._step_dist % distance_per_waypoint) / distance_per_waypoint if distance_per_waypoint > 0 else 0.0

        idx = min(int(segment_progress * num_intervals), len(self._waypoint_queue) - 1)
        if idx < len(self._waypoint_queue) - 1:
            wp1 = self._waypoint_queue[idx]
            wp2 = self._waypoint_queue[idx + 1]
            interp_x = wp1.x + waypoint_progress * (wp2.x - wp1.x)
            interp_y = wp1.y + waypoint_progress * (wp2.y - wp1.y)
            dyaw = (wp2.yaw - wp1.yaw + math.pi) % (2.0 * math.pi) - math.pi
            interp_yaw = wp1.yaw + waypoint_progress * dyaw
            self.set_pose(interp_x, interp_y, interp_yaw)
        else:
            wp = self._waypoint_queue[idx]
            self.set_pose(wp.x, wp.y, wp.yaw)
