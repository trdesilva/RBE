import numpy as np
import matplotlib.pyplot as plt

class Cylinder:
    def __init__(self, origin, height, radius, direction):
        self.origin = origin
        self.height = height
        self.radius = radius
        self.direction = direction/np.linalg.norm(direction)