import valet.obstacle_field as obs
import valet.grid_sim as ui
import valet.planner as plan

def main() -> None:
    obs.FIELD_HEIGHT = 12
    obs.FIELD_WIDTH = 12
    obs._cell_width = 25
    obs._cell_height = 25
    _root_window, _frame, _canvas = ui.create_window()
    obs.draw_grid()
    obs.populate_cells(0.1, True, _canvas)

    # start and parking spots
    obs.clear_cell(0, 0)
    obs.clear_cell(0, 1)

    obs.clear_cell(8, 11)
    obs.clear_cell(9, 11)
    obs.add_to_cell(6, 11, 1)
    obs.add_to_cell(7, 11, 1)
    obs.add_to_cell(10, 11, 1)
    obs.add_to_cell(11, 11, 1)

    goal_rect = obs.DrawableRect(9, 11.5, 2, 1, fill_color=(0, 1, 0, 0.5), parent=_canvas)
    _canvas.add_drawable(goal_rect)

    # car = Vehicle(meters_to_cells(5.2), meters_to_cells(1.8))
    robot = plan.Car()
    _canvas.add_drawable(robot)

    def step_time_callback():
        global counter, last_x, last_y
        robot.run_queued_motions(ui.time_rate / ui.fps)
        if robot.check_collision(obs.obs_quadtree):
            robot.body.fill_color = (1.0, 0, 0)
            ui._current_time_rate = 0
            print(f'robot crashed')
        robot.update_visualization()
        ui.update_preview_pose()
        _canvas.redraw()

    ui.step_time_callback = step_time_callback
    ui.set_active_vehicle(robot)
    ui.clear_visualization_callback = robot.clear_visualizations

    obs.draw_grid()
    _frame.pack()

    def run_astar():
        print(f'running A* from {robot.x, robot.y, robot.theta}')
        parking_space = (8.0, 11.0, 10.0, 12.0)
        # 2x2 expanded goal region covering the parking space and adjacent driving aisle
        astar_goal_region = (8.0, 10.0, 10.0, 12.0)
        start_node = plan.MPSearchNode(None, robot.x, robot.y, robot.theta, 1, robot.body)
        path = robot.astar(start_node, astar_goal_region)
        counter = 0
        if path is None:
            print(f'No path found')
            return
        path = list(path)
        robot.optimal_path_line_strips = [
            getattr(path_segment, 'line_strip', None)
            for path_segment in path
            if getattr(path_segment, 'line_strip', None) is not None
        ]
        for path_segment in path:
            print(f'{counter}: {path_segment}')
            if path_segment.via:
                robot.queue_motion(path_segment.via)
            counter += 1

        end_node = path[-1] if len(path) > 0 else start_node
        local_mps = getattr(robot, '_cached_local_plan', None)
        if local_mps is None:
            local_mps = robot.local_planner(end_node, parking_space, None, obs.obs_quadtree)
        if local_mps is not None:
            for mp in local_mps:
                print(f'{counter} (local): {mp}')
                robot.queue_motion(mp)
                strip = getattr(mp, 'world_line_strip', None)
                if strip is not None:
                    robot.optimal_path_line_strips.append(strip)
                    if _canvas is not None:
                        _canvas.add_drawable(strip)
                counter += 1
        else:
            print('No collision-free Reeds-Shepp path found by local planner')

        robot.update_visualization()
        if _canvas is not None and getattr(_canvas, 'context_created', False):
            _canvas.display()

    ui.run_astar_callback = run_astar
    _root_window.mainloop()