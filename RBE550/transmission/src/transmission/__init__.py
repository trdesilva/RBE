import matplotlib.pyplot as plt

from geometry import Cylinder
from collision import check_cylinders
import numpy as np

def main() -> None:
    c1 = Cylinder((0, 0, 0), 1, 2, (0, 0, 1))
    c2 = Cylinder((2.01, 0, 0), 1, 2, (0, 0, 1))
    fig = plt.figure()
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2, fig)}')

    c2.origin = np.array((1.99, 0, 0))
    fig = plt.figure()
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2, fig)}')

    c2.origin = np.array((2.99, 0, 0))
    c2.direction = np.array((-1, 0, 0))
    fig = plt.figure()
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2, fig)}')

    c2.origin = np.array((1.4, 1.4, 2))
    c2.direction = np.array((-1., 0., 1.))
    c2.direction /= np.linalg.norm(c2.direction)
    fig = plt.figure()
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2, fig)}')

    c2.origin = np.array((.707, 0, 2.708))
    c2.direction = np.array((-1., 0., 1.))
    c2.direction /= np.linalg.norm(c2.direction)
    fig = plt.figure()
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2, fig)}')
    plt.show(block=True)

if __name__ == '__main__':
    main()
