import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
from geometry import Cylinder

def check_cylinders(c1: Cylinder, c2: Cylinder, fig: plt.Figure|None = None):
    """
    Implementation of method from "COLLISION DETECTION OF CYLINDRICAL RIGID BODIES USING LINE GEOMETRY" by John S.
    Ketchel and Pierre M. Larochelle. DOI: 10.1109/ROBOT.2006.1641925
    (No-paywall early draft, but it's missing the _tp2_search section:
    http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf )
    :param c1:
    :param c2:
    :return: True if c1 collides with c2, False otherwise
    """
    ax = None
    try:
        if fig is not None:
            ax = c1.render(fig, alpha=0.8)
            c2.render(ax, alpha=0.8)
        s = c1.height*c1.direction # axis of c1
        w = c2.height*c2.direction # axis of c2
        n = np.cross(s, w) # common normal of c1 and c2
        if np.linalg.norm(n) == 0:
            # c1 and c2 are parallel
            return np.linalg.norm(c2.origin - c1.origin) <= c1.radius + c2.radius
        t1 = np.dot(np.cross((c2.origin - c1.origin), w), n)/np.dot(n, n) # distance parameter along s for where n meets
        t2 = np.dot(np.cross((c2.origin - c1.origin), s), n)/np.dot(n, n) # distance parameter along w for where n meets
        p = c1.origin + t1*s # point where n meets s
        q = c2.origin + t2*w # point where n meets w
        if np.linalg.norm(p - q) > c1.radius + c2.radius:
            # the axes never get close enough to collide regardless of cylinder length
            return False

        if ax is not None:
            _plot_vectors(np.column_stack((t1*s, t2*w, n)), np.column_stack((c1.origin, c2.origin, p)), ax)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            # common normal intersects both cylinders
            return np.linalg.norm(p - q) <= c1.radius + c2.radius
        elif 0 <= t1 <= 1 or 0 <= t2 <= 1:
            # common normal intersects one cylinder
            return _on_test(c1, s, c2, w, t2) if 0 <= t1 <= 1 else _on_test(c2, w, c1, s, t1)
        else:
            # common normal intersects neither cylinder
            return _off_test(c1, s, p, c2, w, q)
    finally:
        if ax is not None:
            ax.set_aspect('equal')

def _on_test(c1, s, c2, w, t2):
    tp2 = c2.origin if t2 <= 0.5 else c2.origin + w # closest point on c2's axis to the common normal
    p1 = c1.origin - np.dot(c1.origin - tp2, c1.direction)*c1.direction # closest point on c1's axis to tp2
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        t3 = np.dot(p1 - c1.origin, c1.direction) # distance param along s from c1.origin to p1
        if t3 < 0 or 1 < t3:
            tp1 = c1.origin if t3 <= 0 else c1.origin + s # closest point on c1's axis to tp2
            if np.linalg.norm(tp1 - p1) > c2.radius:
                return False

        tp2_prime, p1_prime, t3_prime = _tp2_search(c1, c2, tp2)
        if 0 <= t3_prime <= 1:
            # tp2 is between the top and bottom of c1
            return np.linalg.norm(tp2_prime - p1_prime) <= c1.radius
        if 0 <= t3 <= 1:
            tp1 = c1.origin if t3_prime <= 0.5 else c1.origin + s
            return _end_test(c1, s, tp1, c2, w, tp2_prime)
        return False
    else:
        # normal distance is greater than sum of radii, so no collision possible
        return False

def _tp2_search(c1, c2, tp2):
    theta = 0
    inc = np.pi/2
    p1_prime = c1.origin
    t3_prime = 0
    # use Rodrigues' rotation formula to test different points around the edge of c2
    n = np.cross(c1.direction, c2.direction)
    tp2_delta = c2.radius*n/np.linalg.norm(n) # center to edge of c2 face along common normal with c1
    delta_ax_cross = np.cross(c2.direction, tp2_delta)
    result = (tp2 + tp2_delta, p1_prime, t3_prime)
    while inc > 0.01:
        d_min = np.inf
        counter = 0
        while True:
            tp2_prime = tp2 + (c2.direction * np.dot(c2.direction, tp2_delta) + # rotate tp2_delta by theta
                               np.cross(np.cos(theta) * delta_ax_cross, c2.direction) +
                               np.sin(theta) * delta_ax_cross)
            p1_prime = c1.origin - (np.dot((c1.origin - tp2_prime), c1.direction) * c1.direction)
            d = np.linalg.norm(p1_prime - tp2_prime)
            if d < d_min:
                d_min = d
                t3_prime = np.dot(p1_prime - c1.origin, c1.direction)
                result = (tp2_prime, p1_prime, t3_prime)
                if 0 <= t3_prime <= 1 and d < c1.radius:
                    # collision guaranteed, stop searching
                    return result
                theta += inc
                counter += 1
            else:
                break
        if counter == 0:
            theta -= inc
        else:
            theta -= 2*inc
        inc /= 2
    return result

def _end_test(c1, s, tp1, c2, w, tp2):
    # check if intersection of planes parallel to cylinder ends which pass through test points is inside the end faces
    m = np.cross(s, w) # direction of intersection
    xyz0 = np.linalg.solve(np.vstack((s, w, m)), # some point along the intersection
                           np.array((np.dot(s, tp1), np.dot(w, tp2), 0.)))
    t4_roots = np.roots((m[0] ** 2 + m[1] ** 2 + m[2] ** 2, # distance parameter where intersection line meets c1 edge
                         2*m[0]*(xyz0[0] - tp1[0]) + 2*m[1]*(xyz0[1] - tp1[1]) + 2*m[2]*(xyz0[2] - tp1[2]),
                         (xyz0[0] - tp1[0]) ** 2 + (xyz0[1] - tp1[1]) ** 2 + (xyz0[2] - tp1[2]) ** 2 - c1.radius ** 2))
    if np.iscomplex(t4_roots[0]):
        # intersection line doesn't pass through the end face of c1
        return False
    t5_roots = np.roots((m[0] ** 2 + m[1] ** 2 + m[2] ** 2, # distance parameter where intersection line meets c2 edge
                         2 * m[0] * (xyz0[0] - tp2[0]) + 2 * m[1] * (xyz0[1] - tp2[1]) + 2 * m[2] * (xyz0[2] - tp2[2]),
                         (xyz0[0] - tp2[0]) ** 2 + (xyz0[1] - tp2[1]) ** 2 + (xyz0[2] - tp2[2]) ** 2 - c2.radius ** 2))
    if np.iscomplex(t5_roots[0]):
        # intersection line doesn't pass through the end face of c1
        return False
    # check if the parts of the intersection line that are inside each face overlap
    return (t4_roots[0] <= t5_roots[0] <= t4_roots[1] or t4_roots[0] <= t5_roots[1] <= t4_roots[1] or
            t5_roots[0] <= t4_roots[0] <= t5_roots[1] or t5_roots[0] <= t4_roots[1] <= t5_roots[1])

def _off_test(c1, s, c2, w, is_retry = False):
    # do _on_tests with axis endpoints (TODO: this seems like it could be folded into _on_test, but it's in the paper)
    tp2 = c2.origin
    p1 = c1.origin - np.dot(c1.origin - tp2, c1.direction)*c1.direction
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        # this end is close enough to warrant more detailed checking
        collides = _on_test(c1, s, c2, w, 0)
        if collides:
            return True
    tp2 = c2.origin + w
    p1 = c1.origin - np.dot(c1.origin - tp2, c1.direction)*c1.direction
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        collides = _on_test(c1, s, c2, w, 1)
        if collides:
            return True

    if not is_retry:
        # switch the cylinders
        return _off_test(c2, w, c1, s, True)
    return False

def _plot_vectors(vectors: np.typing.NDArray, origins:np.typing.NDArray, ax: plt3.Axes3D):
    ax.quiver(*origins.reshape((3, -1)), *vectors.reshape((3, -1)))
