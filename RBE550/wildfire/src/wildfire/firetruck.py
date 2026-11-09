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
        k_neighbors: int = 4,
        sample_dist_ratio = 4.0
    ):
        self.obs_quadtree = obs_quadtree
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.turn_radius = turn_radius
        self.step_size = step_size
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.k_neighbors = k_neighbors
        self.sample_dist_ratio = sample_dist_ratio

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

    def add_node(self, pose: Tuple[float, float, float]) -> PRMNode:
        """Adds a new collision-free PRMNode to the roadmap and spatial index."""
        node_id = len(self.nodes)
        new_node = PRMNode(id=node_id, pose=pose)
        self.nodes.append(new_node)
        self.adj_list[node_id] = []
        self.node_quadtree.insert((pose[0], pose[1]), new_node)
        return new_node

    def sample_uniform_nodes(self, target_count: int, batch_size: int = 200) -> None:
        """Uniformly samples collision-free configurations using NumPy batches until target_count is reached."""
        min_x, min_y, max_x, max_y = self.bounds
        while len(self.nodes) < target_count:
            xs = np.random.uniform(min_x, max_x, batch_size)
            ys = np.random.uniform(min_y, max_y, batch_size)
            thetas = np.random.uniform(-np.pi, np.pi, batch_size)
            for x, y, theta in zip(xs, ys, thetas):
                fx, fy, ftheta = float(x), float(y), float(theta)
                if self.is_pose_collision_free(fx, fy, ftheta):
                    self.add_node((fx, fy, ftheta))
                    if len(self.nodes) >= target_count:
                        break

    def sample_gaussian_nodes(
        self,
        target_count: int,
        batch_size: int = 200,
        std: float = 0.5,
        std_theta: float = np.pi/16,
        max_batches: int = 1000,
    ) -> None:
        """
        Samples configurations near obstacles using NumPy batches of Gaussian perturbations.
        For each pair (s1, s2) where s1 is uniform and s2 ~ N(s1, sigma), adds the collision-free
        sample to self.nodes if and only if one sample collides and the other does not.
        """
        min_x, min_y, max_x, max_y = self.bounds
        batch_count = 0
        while len(self.nodes) < target_count and batch_count < max_batches:
            batch_count += 1
            xs1 = np.random.uniform(min_x, max_x, batch_size)
            ys1 = np.random.uniform(min_y, max_y, batch_size)
            thetas1 = np.random.uniform(-np.pi, np.pi, batch_size)

            xs2 = np.random.normal(xs1, std)
            ys2 = np.random.normal(ys1, std)
            thetas2 = (np.random.normal(thetas1, std_theta) + np.pi) % (2.0 * np.pi) - np.pi

            valid_mask = (xs2 >= min_x) & (xs2 <= max_x) & (ys2 >= min_y) & (ys2 <= max_y)
            valid_xs1 = xs1[valid_mask]
            valid_ys1 = ys1[valid_mask]
            valid_thetas1 = thetas1[valid_mask]
            valid_xs2 = xs2[valid_mask]
            valid_ys2 = ys2[valid_mask]
            valid_thetas2 = thetas2[valid_mask]

            for x1, y1, th1, x2, y2, th2 in zip(
                valid_xs1, valid_ys1, valid_thetas1, valid_xs2, valid_ys2, valid_thetas2
            ):
                fx1, fy1, fth1 = float(x1), float(y1), float(th1)
                fx2, fy2, fth2 = float(x2), float(y2), float(th2)

                c1 = not self.is_pose_collision_free(fx1, fy1, fth1)
                c2 = not self.is_pose_collision_free(fx2, fy2, fth2)

                if c1 != c2:
                    free_pose = (fx1, fy1, fth1) if not c1 else (fx2, fy2, fth2)
                    self.add_node(free_pose)
                    if len(self.nodes) >= target_count:
                        break

    def connect_node(
        self,
        node: PRMNode,
        max_edges: Optional[int] = None,
        add_to_graph: bool = True,
    ) -> List[PRMEdge]:
        """
        Connects a single PRMNode to the roadmap.
        Uses node_quadtree to sample the nearest 4*self.k_neighbors candidates instead
        of sorting the entire set of nodes.
        Terminates as soon as target_k valid collision-free edges are generated.
        """
        target_k = max_edges if max_edges is not None else self.k_neighbors
        if len(self.nodes) == 0 or target_k <= 0:
            return []

        # Sample the nearest 4*k (+1 to account for node itself if in tree)
        sample_count = min(len(self.nodes) + 1, 4 * target_k + 1)
        nearest_items = self.node_quadtree.nearest_neighbors((node.pose[0], node.pose[1]), sample_count)

        edges: List[PRMEdge] = []
        for item in nearest_items:
            target_node: PRMNode = item.obj
            if target_node.id == node.id:
                continue

            d = math.hypot(node.pose[0] - target_node.pose[0], node.pose[1] - target_node.pose[1])
            if d > self.connection_radius:
                continue

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
                    target_id=target_node.id,
                    path=path,
                    cost=path.total_length,
                    line_strip=strip,
                )
                if add_to_graph and node.id in self.adj_list:
                    self.adj_list[node.id].append(edge)
                    if self.visualizer is not None:
                        self.visualizer.add_edge(edge, strip)

                edges.append(edge)
                if len(edges) >= target_k:
                    break

        return edges

    def generate_roadmap(self) -> None:
        """
        Constructs the roadmap once before simulation starts:
        1. Samples random collision-free configurations (first 1/sample_dist_ratio uniform, rest Gaussian near obstacles).
        2. Finds candidate neighbor pairs within connection_radius using node_quadtree.
        3. Connects neighbors with collision-free Reeds-Shepp paths using connect_node.
        """
        self.nodes.clear()
        self.adj_list.clear()
        self.node_quadtree.clear()
        if self.visualizer is not None:
            self.visualizer.clear()

        # Step 1: Sample vertices
        print(f'PRM sampling {self.num_samples} vertices...')

        # First half: uniform random strategy
        self.sample_uniform_nodes(target_count=int(self.num_samples/self.sample_dist_ratio))

        # Second half: Gaussian sampler near obstacles
        self.sample_gaussian_nodes(target_count=self.num_samples)

        # Step 2 & 3: Connect k-nearest neighbors with Reeds-Shepp paths
        print(f'PRM connecting {len(self.nodes)} nodes...')
        for i, node in enumerate(self.nodes):
            self.connect_node(node, add_to_graph=True)
            if (i + 1) % max(1, int(self.num_samples / 100)) < 1:
                if self.visualizer is not None:
                    self.visualizer.update_visualization()
            if (i + 1) % max(1, int(self.num_samples / 10)) < 1:
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
        if goal_region is None:
            return None
        if len(self.nodes) == 0:
            self.generate_roadmap()

        # Temporary start node with out-of-bounds positive ID so it cannot be
        # accessed via negative indexing (e.g. self.nodes[-1]) or modify self.nodes / self.adj_list.
        start_node = PRMNode(id=len(self.nodes), pose=start_pose)
        valid_connections = self.connect_node(start_node, max_edges=max(10, self.k_neighbors), add_to_graph=False)

        if len(valid_connections) > 0:
            valid_connections.sort(key=lambda edge: edge.cost)
            start_edge = valid_connections[0]
            start_path = start_edge.path
            first_node = self.nodes[start_edge.target_id]
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
        if not current.is_goal and goal.is_goal and goal.data is not None:
            dist_to_goal, goal_vert = goal.data.get_minimum_distance(current.data.pose[0], current.data.pose[1])
            gx, gy = goal_vert
            cx, cy, ctheta = current.data.pose
            max_dist_to_goal = float(np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2))
            xy_progress = abs(max_dist_to_goal - dist_to_goal) / max_dist_to_goal
            angle_to_goal = np.atan2(gy - cx, gx - cy)
            # prefer pointing at goal when far away and being parallel with goal when close
            heading_heur = (1 - xy_progress) * angle_to_goal + xy_progress * (np.sin(abs(ctheta)))
            result = dist_to_goal + heading_heur
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
                    cost = edge.cost
                    waypoints = edge.path.waypoints()
                    segments = edge.path.segments
                    for i in range(1, len(waypoints)):
                        cost += abs(waypoints[i].yaw - waypoints[i - 1].yaw)/(2*len(waypoints))
                    for i in range(1, len(segments)):
                        cost *= 1.5 if segments[i].direction != segments[i - 1].direction else 1
                    self.visualizer.update_cost_display(edge, edge.cost)
                return edge.cost
        return np.inf

    def is_goal_reached(self, current: AStarData, goal: AStarData):
        if not current.is_goal and goal.is_goal and goal.data is not None:
            return goal.data.contains(current.data.pose[0], current.data.pose[1])
        return False

FT_MOVE = 0
FT_EXT = 1

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

        self.prm = PRMPlanner(obs_quadtree, self.body, (0, 0, 25, 25), self.turn_radius, 0.1, 200, 10, 16, 8)

        # Motion execution queue
        self._path_queue = []
        self._waypoint_queue: List[rsplan.primitives.Waypoint] = []
        self._step_dist = 0
        self.current_goal: Optional[shr.ExtRegion] = None
        self.current_action = FT_MOVE

    def check_collision(self, obs_quadtree: qt.RectQuadTree) -> bool:
        """Tests whether the current body pose collides with the obstacle field."""
        return vp.check_collision(self.body, obs_quadtree)

    def queue_paths(self, paths: List[rsplan.primitives.Path], clear_queue = True) -> None:
        """Enqueues waypoints from a sequence of rsplan Paths for navigation."""
        if clear_queue:
            self._path_queue.clear()
            self._waypoint_queue.clear()
        for path in paths:
            self._path_queue.append(path)
        if len(self._path_queue) > 0:
            self._waypoint_queue = self._path_queue[0].waypoints()

    def replan(self):
        """
        1. Find the nearest fire it's possible to make it to in time and make it the goal
            a. For each nearest burning neighbor
                i. if extinguish timer < burn timer and in range, set goal = neighbor and break
                ii. elif extinguish timer < burn timer, calculate A* path
            b. sort path candidates by total length
            c. if shortest path length / speed < burn timer - extinguish timer, set goal and queue path
        2. If in extinguish range of goal, set action to extinguishing, else set action to moving
        2. If there's no goal, drive to the middle of the map
        :return: None
        """
        checked_regions = {}
        best = None
        for cell_tup in shr.nearest_neighbors(self.x, self.y, shr.OBS_BURNING, 100):
            region = shr.get_ext_region(cell_tup[1], cell_tup[2])
            if region in checked_regions:
                continue
            if (region.get_timers()[0] > region.get_timers()[1] and
                region.get_minimum_distance(self.x, self.y)[0] < ui.meters_to_cells(10)):
                best = region
                checked_regions[region] = []
                break
            elif region.get_timers()[0] > region.get_timers()[1]:
                checked_regions[region] = self.prm.query((self.x, self.y, self.theta), region)
                if best is None or (sum([p.total_length for p in checked_regions[region]])
                                    < sum([p.total_length for p in checked_regions[best]])):
                    best = region

        if best is not None:
            print(f'Firetruck pathing to {best.obs_origin} ({best.get_timers()[0]:0.2f}s remaining)')
            path = checked_regions[best]
            if path is not None and len(path) == 0: # empty path list means we're there
                self.current_goal = best
                shr.set_current_goal(best)
                self.current_action = FT_EXT
                return
            elif path is not None and len(path) > 0:
                print(f'Firetruck found path to {best}')
                self.current_goal = best
                shr.set_current_goal(best)
                self.queue_paths(path)
                self.current_action = FT_MOVE
                return
            else:
                self.current_goal = None
                shr.set_current_goal(None)
                print(f'Firetruck pathing failed')
        # TODO go to middle
        self.current_goal = None
        shr.set_current_goal(None)

    def try_extinguish(self):
        """
        1. Check if goal is in range and extinguish timer < burn timer
            a. true: extinguish and return
        2. For each nearest burning neighbor, check if in range and extinguish timer < burn timer
            a. true: extinguish and return
        3. return false
        :return: True if extinguishing this step, False otherwise
        """
        if self.current_goal is not None:
            if (self.current_goal.get_minimum_distance(self.x, self.y)[0] < ui.meters_to_cells(10) and
                self.current_goal.get_timers()[0] > self.current_goal.get_timers()[1]):
                    shr.extinguishing_from(self)
                    return True
            for cell_tup in shr.nearest_neighbors(self.x, self.y, shr.OBS_BURNING, 8):
                region = shr.get_ext_region(cell_tup[1], cell_tup[2])
                if (region.get_timers()[0] > region.get_timers()[1] and
                    region.get_minimum_distance(self.x, self.y)[0] < ui.meters_to_cells(10)):
                    self.current_goal = region
                    shr.set_current_goal(region)
                    shr.extinguishing_from(self)
                    return True
        return False

    def step(self, dt: float) -> None:
        """
        1. If extinguishing, try_extinguish()
            a. return if try_extinguish() returns true
        2. If no moves planned or goal unset or goal is middle of map or goal extinguish timer < burn timer, replan
        3. If no moves planned, return
        4. Handle movement according to _path_queue
        """
        if self.current_action == FT_EXT and self.try_extinguish():
            return

        if len(self._path_queue) == 0 or self.current_goal is None or self.current_goal.get_timers()[0] < self.current_goal.get_timers()[1]:
            self.replan()

        if self.current_action == FT_EXT and self.try_extinguish(): # check again after replanning
            return

        if dt <= 0 or self.speed <= 0 or not self._waypoint_queue or len(self._path_queue) == 0 or self.current_action != FT_MOVE:
            return

        path = self._path_queue[0]
        path_len = abs(path.total_length)
        if path_len == 0 or len(self._waypoint_queue) == 0:
            return

        distance_to_cover = dt * self.speed
        num_intervals = max(len(self._waypoint_queue) - 1, 1)
        distance_per_waypoint = path_len / num_intervals
        if distance_per_waypoint == 0:
            return

        rem = self._step_dist % distance_per_waypoint
        dist_to_next = distance_per_waypoint - rem if rem > 1e-9 else distance_per_waypoint

        while distance_to_cover >= dist_to_next - 1e-9:
            distance_to_cover = max(0.0, distance_to_cover - dist_to_next)
            self._step_dist += dist_to_next
            if self._step_dist >= path_len - 1e-9:
                self._step_dist -= path_len
                if self._step_dist < 0:
                    self._step_dist = 0.0
                last_wp = self._waypoint_queue[-1]
                self._path_queue.pop(0)
                if not self._path_queue:
                    self._waypoint_queue = []
                    self.set_pose(last_wp.x, last_wp.y, last_wp.yaw)
                    return
                path = self._path_queue[0]
                self._waypoint_queue = path.waypoints()
                if not self._waypoint_queue:
                    return
                path_len = abs(path.total_length)
                if path_len == 0:
                    return
                num_intervals = max(len(self._waypoint_queue) - 1, 1)
                distance_per_waypoint = path_len / num_intervals
                if distance_per_waypoint == 0:
                    return

            path_progress = self._step_dist / path_len
            idx = min(int(path_progress * num_intervals), len(self._waypoint_queue) - 1)
            wp = self._waypoint_queue[idx]
            self.set_pose(wp.x, wp.y, wp.yaw)
            dist_to_next = distance_per_waypoint

        self._step_dist += distance_to_cover
        path_progress = self._step_dist / path_len
        waypoint_progress = (self._step_dist % distance_per_waypoint) / distance_per_waypoint if distance_per_waypoint > 0 else 0.0

        idx = min(int(path_progress * num_intervals), len(self._waypoint_queue) - 1)
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
