from valet import obstacle_field
from valet.obstacle_field import ObstacleFieldWidget, create_window, draw_grid, populate_cells

__all__ = [
    "obstacle_field",
    "ObstacleFieldWidget",
    "create_window",
    "draw_grid",
    "populate_cells",
    "main",
]


def main() -> None:
    root_window, frame, widget = create_window()
    draw_grid(widget)
    populate_cells(0.7, True, widget)
    frame.pack()
    root_window.mainloop()
