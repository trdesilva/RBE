import matplotlib.pyplot as plt
import numpy as np

from geometry import Cylinder
from collision import check_cylinders, check_hole


def test_check_cylinders() -> None:
    print("--- Running check_cylinders tests ---")
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


def test_check_hole(show_figures: bool = True) -> None:
    print("\n--- Running check_hole tests ---")
    c_hole = Cylinder((0, 0, 0), 1.0, 2.0, (0, 0, 1))

    cases = [
        ('1. Coaxial Clean Pass', Cylinder((0, 0, -2), 0.5, 6, (0, 0, 1)), True),
        ('2. Coaxial Oversized', Cylinder((0, 0, -2), 1.2, 6, (0, 0, 1)), False),
        ('3. Parallel Offset Pass', Cylinder((0.3, 0, -2), 0.4, 6, (0, 0, 1)), True),
        ('4. Parallel Offset Collide', Cylinder((0.6, 0, -2), 0.5, 6, (0, 0, 1)), False),
        ('5. Tilted Clean Pass', Cylinder((0, 0, -2), 0.3, 6, (0.1, 0, 1)), True),
        ('6. Tilted Colliding', Cylinder((0, 0, -2), 0.8, 6, (0.1, 0, 1)), False),
        ('7. Short Cylinder Before Hole', Cylinder((0, 0, -5), 0.8, 1, (0.1, 0, 1)), True),
        ('8. 3D Skew Clean Pass', Cylinder((0.2, 0.1, -2), 0.2, 7, (0.1, 0.05, 1)), True),
        ('9. 3D Skew Colliding', Cylinder((0.2, 0.1, -2), 0.7, 7, (0.1, 0.05, 1)), False),
    ]

    for name, c_obj, expected in cases:
        fig = plt.figure() if show_figures else None
        res = check_hole(c_obj, c_hole, fig)
        status = 'PASS' if res == expected else 'FAIL'
        print(f'[{status}] {name}: got {res} (expected {expected}) | obj: {c_obj}')


def main() -> None:
    test_check_cylinders()
    test_check_hole(show_figures=True)
    plt.show(block=True)


if __name__ == '__main__':
    main()
