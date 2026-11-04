import valet.obstacle_field as obs
import valet.grid_sim as ui
import rsplan as rs

class Firetruck(obs.DrawableGroup):
    def __init__(self):
        obs.DrawableGroup.__init__(self)
        self.body = obs.DrawableRect(width=ui.meters_to_cells(2.2), height=ui.meters_to_cells(4.9), fill_color=(0.9, 0, 0), parent=self)
        self.add(self.body)
        self.wheelbase_offset = ui.meters_to_cells(3)/2. # distance from body center to rear axle
        self.body.origin = (0.5 - self.wheelbase_offset/ui.meters_to_cells(4.9), 0.5)

    # PRM implementation
    #   sampler
    #   rsplan + check_collision = local planner
    #   sampler + local planner = generate map
    #   map + local planner + A* = query step
    # global planner
    #   obs_quadtree + query heuristic
    # visualization
    #   query heuristic
    #       path
    #       color by value
    #   PRM
    #       nodes
    #       valid edges (straight or RS?)