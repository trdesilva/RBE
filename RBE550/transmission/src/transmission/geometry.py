import numpy as np
import matplotlib.pyplot as plt

class Cylinder:
    def __init__(self, origin, radius, height, direction):
        self.origin = np.array(origin)
        self.height = height
        self.radius = radius
        d_norm = np.linalg.norm(direction)
        self.direction = np.array(direction)/d_norm if d_norm > 0 else np.array((0, 0, 1))

    def __repr__(self):
        return f'{self.radius}x{self.height} @ {self.origin}, {self.direction}'