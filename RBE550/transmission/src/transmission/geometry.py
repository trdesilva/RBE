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
        return f'{self.radius}r x {self.height}h @ {self.origin}, {self.direction}'

    def render(self, fig, resolution=50, caps=True, **kwargs):
        """
        Renders the Cylinder as a 3D surface in the given matplotlib figure or axes.

        :param fig: A matplotlib Figure or Axes3D instance.
        :param resolution: Number of points along the circumference.
        :param caps: Whether to render the end caps of the cylinder.
        :param kwargs: Additional keyword arguments passed to plot_surface (e.g., color, alpha).
        :return: The 3D axes (Axes3D) containing the rendered cylinder.
        """
        if hasattr(fig, 'plot_surface'):
            ax = fig
        elif hasattr(fig, 'axes'):
            ax = None
            for a in fig.axes:
                if hasattr(a, 'plot_surface'):
                    ax = a
                    break
            if ax is None:
                ax = fig.add_subplot(111, projection='3d')
        else:
            raise TypeError(f"Expected a matplotlib Figure or Axes3D, got {type(fig)}")

        d = self.direction
        # Choose a reference vector not parallel to d to construct an orthonormal basis
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, ref)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)

        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(0, self.height, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Lateral surface
        x = self.origin[0] + z_grid * d[0] + self.radius * (np.cos(theta_grid) * u[0] + np.sin(theta_grid) * v[0])
        y = self.origin[1] + z_grid * d[1] + self.radius * (np.cos(theta_grid) * u[1] + np.sin(theta_grid) * v[1])
        z_coords = self.origin[2] + z_grid * d[2] + self.radius * (np.cos(theta_grid) * u[2] + np.sin(theta_grid) * v[2])

        surf = ax.plot_surface(x, y, z_coords, **kwargs)

        if caps:
            cap_kwargs = kwargs.copy()
            if 'color' not in cap_kwargs and 'facecolor' not in cap_kwargs and 'facecolors' not in cap_kwargs:
                fc = surf.get_facecolor()
                if len(fc) > 0:
                    cap_kwargs['color'] = fc[0]

            r_vals = np.linspace(0, self.radius, 10)
            r_grid, theta_cap = np.meshgrid(r_vals, theta)

            # Bottom cap at origin
            x_bot = self.origin[0] + r_grid * (np.cos(theta_cap) * u[0] + np.sin(theta_cap) * v[0])
            y_bot = self.origin[1] + r_grid * (np.cos(theta_cap) * u[1] + np.sin(theta_cap) * v[1])
            z_bot = self.origin[2] + r_grid * (np.cos(theta_cap) * u[2] + np.sin(theta_cap) * v[2])
            ax.plot_surface(x_bot, y_bot, z_bot, **cap_kwargs)

            # Top cap at origin + height * direction
            top_origin = self.origin + self.height * d
            x_top = top_origin[0] + r_grid * (np.cos(theta_cap) * u[0] + np.sin(theta_cap) * v[0])
            y_top = top_origin[1] + r_grid * (np.cos(theta_cap) * u[1] + np.sin(theta_cap) * v[1])
            z_top = top_origin[2] + r_grid * (np.cos(theta_cap) * u[2] + np.sin(theta_cap) * v[2])
            ax.plot_surface(x_top, y_top, z_top, **cap_kwargs)

        return ax

    plot = render
