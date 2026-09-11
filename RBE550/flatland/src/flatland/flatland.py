from turtles import obstacle_field as obs

EMPTY = 0
WALL = 1
JUNK = 2
GOAL = 3
HERO = 4
ENEMY = 5

GAME_OVER = -1

class Mover:
    def __init__(self, x, y, cell_type):
        self.x = x
        self.y = y
        self.cell_type = cell_type

    def move(self, dx, dy):
        if (abs(dx) == 1 and abs(dy) == 0) or (abs(dx) == 0 and abs(dy) == 1):
            self.x += dx
            self.y += dy
        else:
            print(f'Illegal move by type {self.cell_type} at ({self.x}, {self.y}): ({dx}, {dy})')

    def collide(self, with_type, other = None):
        return # no-op for base class

class Enemy(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY)

    def collide(self, with_type, other = None):
        if with_type == HERO:
            print(f'Enemy killed hero at ({self.x}, {self.y})!')
            if other is not None:
                other.collide(self.cell_type)
        elif with_type == WALL or with_type == JUNK or with_type == ENEMY:
            print(f'Enemy junked at ({self.x}, {self.y})!')
            self.cell_type = JUNK
            if other is not None:
                other.collide(self.cell_type)

class Hero(Mover):
    def __init__(self, x, y):
        super().__init__(x, y, HERO)

    def collide(self, with_type, other = None):
        if with_type == EMPTY:
            return None
        if with_type == HERO:
            print(f'Hero collided with self? ({self.x}, {self.y})')
        if with_type == GOAL:
            print(f'Hero reached goal at ({self.x}, {self.y})!')
        else:
            print(f'Hero hit obstacle at ({self.x}, {self.y})!')
        return GAME_OVER

def get_fill_color(value):
    if value == EMPTY:
        return 'white'
    if value == WALL:
        return 'black'
    if value == JUNK:
        return 'blue'
    if value == GOAL:
        return 'green'
    if value == HERO:
        return "#00ffff"
    if value == ENEMY:
        return 'red'
    return 'magenta'

if __name__ == "__main__":
    obs.FIELD_HEIGHT = 64
    obs.FIELD_WIDTH = 64
    obs._fill_func = get_fill_color
    root_window, frame, canvas = obs.create_window()
    obs.draw_grid(canvas)
    obs.populate_cells(0.5, True, canvas)
    obs.update_cell(0, 0, HERO)
    obs.draw_grid(canvas)
    frame.pack()
    root_window.mainloop()