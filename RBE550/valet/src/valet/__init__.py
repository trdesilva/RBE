from valet import obstacle_field
from valet.obstacle_field import (
    DrawableGroup,
    DrawableRect,
    ObstacleFieldWidget,
    TF2DOwner,
    create_window,
    draw_grid,
    populate_cells,
)
from valet import planner
from valet.planner import (
    InputFrame,
    MotionPrimitive,
    Robot,
    StateFrame,
    Vehicle,
    VehicleState,
    cells_to_meters,
    meters_to_cells,
)

__all__ = [
    "obstacle_field",
    "ObstacleFieldWidget",
    "TF2DOwner",
    "DrawableRect",
    "DrawableGroup",
    "create_window",
    "draw_grid",
    "populate_cells",
    "main",
    "planner",
    "InputFrame",
    "MotionPrimitive",
    "Robot",
    "StateFrame",
    "Vehicle",
    "VehicleState",
    "cells_to_meters",
    "meters_to_cells",
]


def main() -> None:
    root_window, frame, widget = create_window()
    draw_grid(widget)
    populate_cells(0.7, True, widget)
    frame.pack()
    root_window.mainloop()
