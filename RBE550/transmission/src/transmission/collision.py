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
        # intersection line doesn't pass through the end face of c2
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

def check_hole(c_obj: Cylinder, c_hole: Cylinder, fig: plt.Figure|None = None) -> bool:
    """
    Analytically determines whether a solid cylindrical object passes through a
    cylindrical hole without colliding with the radial boundary (the circular rims or walls) of the hole.

    :param c_obj: Cylinder representing the solid cylindrical object
    :param c_hole: Cylinder representing the cylindrical hole in another object
    :param fig: Optional matplotlib Figure to render visualization
    :return: True if the object does not collide with the edge of the hole, False otherwise
    """
    ax = None
    try:
        if fig is not None:
            ax = c_hole.render(fig, alpha=0.25, color='gray')
            c_obj.render(ax, alpha=0.7, color='cyan')

        d_obj = c_obj.direction / np.linalg.norm(c_obj.direction)
        d_hole = c_hole.direction / np.linalg.norm(c_hole.direction)
        s = c_obj.height * d_obj
        w = c_hole.height * d_hole
        m = np.cross(s, w)

        # Case 1: Parallel axes
        if np.linalg.norm(m) == 0:
            diff = c_obj.origin - c_hole.origin
            d_perp = np.linalg.norm(diff - np.dot(diff, d_hole) * d_hole)
            # For the object to pass cleanly through, it must fit inside the hole's radius
            result = (d_perp + c_obj.radius) <= c_hole.radius
            if ax is not None:
                ax.set_title(f"Parallel: {'Passes through (No Collision)' if result else 'Collision with hole edge'}")
            return result

        # Case 2: Object axis perpendicular to hole axis (parallel to hole end planes)
        dot_d = np.dot(d_obj, d_hole)
        if abs(dot_d) < 1e-9:
            if ax is not None:
                ax.set_title("Perpendicular axis: Collision")
            return False

        # Two planes of interest: bottom and top planes of the hole cylinder
        O_h_bot = c_hole.origin
        O_h_top = c_hole.origin + w

        # Parametric intersection of object axis line with hole top/bottom planes
        lam_bot = np.dot(O_h_bot - c_obj.origin, d_hole) / dot_d
        lam_top = np.dot(O_h_top - c_obj.origin, d_hole) / dot_d

        I_bot = c_obj.origin + lam_bot * d_obj
        I_top = c_obj.origin + lam_top * d_obj

        # Vectors from plane origins to object axis intersection points
        v_bot = I_bot - O_h_bot
        v_top = I_top - O_h_top

        norm_v_bot = np.linalg.norm(v_bot)
        norm_v_top = np.linalg.norm(v_top)

        # Fallback orthogonal unit vector for concentric case (axis through hole center)
        ref = np.array([1.0, 0.0, 0.0]) if abs(d_hole[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_ref = np.cross(d_hole, ref)
        u_ref /= np.linalg.norm(u_ref)

        tp_bot = O_h_bot + (c_hole.radius * v_bot / norm_v_bot if norm_v_bot > 1e-9 else c_hole.radius * u_ref)
        tp_top = O_h_top + (c_hole.radius * v_top / norm_v_top if norm_v_top > 1e-9 else c_hole.radius * u_ref)

        test_points = [
            ('bottom', tp_bot, O_h_bot),
            ('top', tp_top, O_h_top)
        ]

        collides = False
        for name, tp, O_hp in test_points:
            # Parametric t parameter for object slice plane passing through tp
            t = np.dot(tp - c_obj.origin, d_obj) / c_obj.height

            if 0.0 <= t <= 1.0:
                O_slice = c_obj.origin + t * s

                # Intersection line of hole plane (normal w, through O_hp)
                # and object slice plane (normal s, through tp)
                A_mat = np.vstack((s, w, m))
                rhs = np.array((np.dot(s, tp), np.dot(w, O_hp), 0.0))
                xyz0 = np.linalg.solve(A_mat, rhs)

                # Distance function from line L(tau) = xyz0 + tau*m to O_slice
                # Subtract object radius squared: ||xyz0 + tau*m - O_slice||^2 - r_obj^2 = 0
                quad_A = np.dot(m, m)
                quad_B = 2.0 * np.dot(m, xyz0 - O_slice)
                quad_C = np.dot(xyz0 - O_slice, xyz0 - O_slice) - c_obj.radius ** 2

                disc = quad_B ** 2 - 4.0 * quad_A * quad_C

                if ax is not None:
                    # Plot test point on the edge
                    ax.scatter([tp[0]], [tp[1]], [tp[2]], color='red' if disc >= 0 else 'lime', s=60, edgecolors='black')
                    # Plot the intersection line segment near the hole edge
                    m_norm = m / np.linalg.norm(m)
                    line_pts = tp + np.outer(np.linspace(-c_hole.radius, c_hole.radius, 10), m_norm)
                    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color='orange', linestyle='--', linewidth=1.5)

                if disc >= 0.0:
                    collides = True
                    break
            else:
                if ax is not None:
                    # Test point slice plane does not intersect the finite object cylinder
                    ax.scatter([tp[0]], [tp[1]], [tp[2]], color='gray', s=30, alpha=0.5)

        result = not collides
        if ax is not None:
            status_text = 'Passes through (No Collision)' if result else 'Collision with hole edge'
            ax.set_title(status_text)
        return result
    finally:
        if ax is not None:
            ax.set_aspect('equal')
