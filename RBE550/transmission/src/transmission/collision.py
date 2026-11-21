import numpy as np
from geometry import Cylinder

def check_cylinders(c1: Cylinder, c2: Cylinder):
    """
    Implementation of method from "COLLISION DETECTION OF CYLINDRICAL RIGID BODIES USING LINE GEOMETRY" by John S.
    Ketchel and Pierre M. Larochelle.
    (http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf)
    :param c1:
    :param c2:
    :return: True if c1 collides with c2, False otherwise
    """
    s = c1.height*c1.direction # axis of c1
    w = c2.height*c2.direction # axis of c2
    n = np.cross(s, w) # common normal of c1 and c2
    t1 = np.dot(np.cross((c2.origin - c1.origin), w), n)/np.dot(n, n) # distance parameter along s for where n meets
    t2 = np.dot(np.cross((c2.origin - c1.origin), s), n)/np.dot(n, n) # distance parameter along w for where n meets
    p = c1.origin + t1*s # point where n meets s
    q = c2.origin + t2*w # point where n meets w
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        # common normal intersects both cylinders
        return np.linalg.norm(p - q) <= c1.radius + c2.radius
    elif 0 <= t1 <= 1 or 0 <= t2 <= 1:
        # common normal intersects one cylinder
        return _on_test(c1, s, p, c2, w, q, t2) if 0 <= t1 <= 1 else _on_test(c2, w, q, c1, s, p, t1)
    else:
        # common normal intersects neither cylinder
        return _off_test(c1, s, p, c2, w, q)

def _on_test(c1, s, p, c2, w, q, t2):
    tp2 = c2.origin if t2 <= 0 else c2.origin + w # closest point on c2's axis to the common normal
    p1 = c1.origin - (np.dot((c1.origin - tp2), s) * s) # closest point on c1's axis to tp2
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        t3 = np.dot(p1 - c1.origin, s)/np.dot(s, s) # distance param along s from c1.origin to p1
        if t3 < 0 or 1 < t3:
            tp1 = c1.origin if t3 < 0 else c1.origin + s # closest point on c1's axis to tp2
            if np.linalg.norm(tp1 - p1) > c2.radius:
                return False
        n = (p - q)/np.linalg.norm(p - q)
        tp2_prime = tp2 + c2.radius*n
        p1_prime = c1.origin - (np.dot((c1.origin - tp2_prime), s) * s)
        t3_prime = np.dot(p1_prime - c1.origin, s) / np.dot(s, s)
        if 0 <= t3_prime <= 1:
            return np.linalg.norm(tp2_prime - p1_prime) <= c1.radius
        if 0 <= t3 <= 1:
            tp1 = c1.origin if t3_prime < 0 else c1.origin + s
            return _end_test(c1, s, tp1, c2, w, tp2_prime)
        return False
    else:
        # normal distance is greater than sum of radii, so no collision possible
        return False

def _end_test(c1, s, tp1, c2, w, tp2):
    # check if intersection of planes parallel to cylinder ends which pass through test points is inside the end faces
    m = np.cross(s, w) # direction of intersection
    xyz0 = np.linalg.solve(np.ndarray((s, w, m)), # some point along the intersection
                           np.ndarray((np.dot(s, tp1), np.dot(w, tp2), 0)))
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

def _off_test(c1, s, p, c2, w, q, is_retry = False):
    # do _on_tests with axis endpoints (TODO: this doesn't seem like it needs to be a separate step to me, but it's in the paper)
    tp2 = c2.origin
    p1 = c1.origin - (np.dot((c1.origin - tp2), s) * s)
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        # this end is close enough to warrant more detailed checking
        collides = _on_test(c1, s, p1, c2, w, tp2, 0)
        if collides:
            return True
    tp2 = c2.origin + w
    p1 = c1.origin - (np.dot((c1.origin - tp2), s) * s)
    if np.linalg.norm(tp2 - p1) <= c1.radius + c2.radius:
        collides = _on_test(c1, s, p1, c2, w, tp2, 1)
        if collides:
            return True

    if not is_retry:
        # switch the cylinders
        return _off_test(c2, w, q, c1, s, p, True)
    return False