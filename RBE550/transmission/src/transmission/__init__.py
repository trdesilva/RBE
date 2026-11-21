from geometry import Cylinder
from collision import check_cylinders
import numpy as np

def main() -> None:
    c1 = Cylinder((0, 0, 0), 1, 2, (0, 0, 1))
    c2 = Cylinder((2.01, 0, 0), 1, 2, (0, 0, 1))
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2)}')
    c2.origin = np.array((1.99, 0, 0))
    print(f'c1 ({c1}) collides with c2 ({c2}): {check_cylinders(c1, c2)}')
    # TODO need more test cases

if __name__ == '__main__':
    main()
