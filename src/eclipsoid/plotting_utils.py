from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from jaxoplanet.starry.core.basis import A1, poly_basis, U
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.core.polynomials import Pijk

from .bounds import compute_projected_ellipse

@partial(jax.jit, static_argnums=(0))
def ortho_grid(res: int, f: float = 0.0, theta: float = 0.0):
    x, y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, res), jnp.linspace(-1.0, 1.0, res))
    xp = x * jnp.cos(theta) + y * jnp.sin(theta)
    yp = -x * jnp.sin(theta) + y * jnp.cos(theta)
    x = xp
    y = yp/(1. - f) # scale y to account for flattening
    z = jnp.sqrt(1.0 - x**2 - y**2)
    y = y + 0.0 * z  # propagate nans
    x = jnp.ravel(x)[None, :]
    y = jnp.ravel(y)[None, :]
    z = jnp.ravel(z)[None, :]
    lat = 0.5 * jnp.pi - jnp.arccos(y)
    lon = jnp.arctan2(x, z)
    return (lat, lon), (x, y, z)


def render_oblate_surface(res, theta,  oblateness, prolateness, surface):
    """Render the surface of a body with an oblate/prolate shape

    Args:
        res (int): resolution
        theta (float): phase angle in radians
        oblateness (float): oblateness (flattening) of the body (0 = sphere, 1 = pancake)
        prolateness (float): prolateness of the body (0 = sphere, -1 = cigar)
        surface (_type_): _description_

    Returns:
        array: image of the surface map evaluated on the projected UNIT object surface (orthographic projection)
        (note: see the starry/eclipsoid paper for details; for 0 oblateness and obliquity the unit object surface is the unit circle)
    """
    r_eq, b_proj, theta_proj = compute_projected_ellipse(1.0, oblateness, prolateness, theta, surface.obl, surface.inc-jnp.pi/2)
    b_proj = b_proj/r_eq
    _, xyz = ortho_grid(res, f=(1.0-b_proj), theta=-theta_proj)
    pT = poly_basis(surface.deg)(*xyz)
    Ry = left_project(surface.ydeg, surface.inc, surface.obl, theta, theta_proj, surface.y.todense())
    A1Ry = A1(surface.ydeg).todense() @ Ry
    p_y = Pijk.from_dense(A1Ry, degree=surface.ydeg)
    u = jnp.array([1, *surface.u])
    p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)
    p = (p_y * p_u).todense()
    return jnp.reshape(pT @ p * surface.amplitude, (res, res))

def lon_lat_lines(n: int = 6, pts: int = 100, radius: float = 1.0, oblateness: float = 0.0, prolateness: float = 0.0):
    assert isinstance(n, int) or len(n) == 2

    if isinstance(n, int):
        n = (n, 2 * n)

    n_lat, n_lon = n

    sqrt_radius = radius

    _theta = np.linspace(0, 2 * np.pi, pts)
    _phi = np.linspace(0, np.pi, n_lat + 1)
    lat = np.array(
        [
            (r * np.cos(_theta), r * (1. - prolateness) * np.sin(_theta), np.ones_like(_theta) * h * (1. - oblateness))
            for (h, r) in zip(
                sqrt_radius * np.cos(_phi), sqrt_radius * np.sin(_phi), strict=False
            )
        ]
    )

    _theta = np.linspace(0, np.pi, pts // 2)
    _phi = np.linspace(0, 2 * np.pi, n_lon + 1)[0:-1]
    radii = np.sin(_theta)
    lon = np.array(
        [
            (
                sqrt_radius * radii * np.cos(p),
                sqrt_radius * (1. - prolateness) * radii * np.sin(p),
                sqrt_radius * (1. - oblateness) * np.cos(_theta),
            )
            for p in _phi
        ]
    )

    return lat, lon

def rotation(inc, obl, theta):
    obl = np.array(obl)
    u = [np.cos(obl), np.sin(obl), 0]
    u /= np.linalg.norm(u)
    u *= -(inc - np.pi / 2)

    R = Rotation.from_rotvec(u)
    R *= Rotation.from_rotvec([0, 0, obl])
    R *= Rotation.from_rotvec([np.pi / 2, 0, 0])
    R *= Rotation.from_rotvec([0, 0, -theta])
    return R


def rotate_lines(lines, inc, obl, theta):
    inc = np.array(inc)
    obl = np.array(obl)
    theta = np.array(theta)
    R = rotation(inc, obl, theta)

    rotated_lines = np.array([R.apply(l.T) for l in lines]).T
    rotated_lines = np.swapaxes(rotated_lines.T, -1, 1)

    return rotated_lines


def plot_lines(lines, axis=(0, 1), ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    # hide lines behind
    other_axis = list(set(axis).symmetric_difference([0, 1, 2]))[0]
    behind = lines[:, other_axis, :] < 0
    _xyzs = lines.copy().swapaxes(1, 2)
    _xyzs[behind, :] = np.nan
    _xyzs = _xyzs.swapaxes(1, 2)

    for i, j in _xyzs[:, axis, :]:
        ax.plot(i, j, **kwargs)


def graticule(
    inc: float,
    obl: float,
    theta: float = 0.0,
    oblateness: float = 0.0,
    prolateness: float = 0.0,
    pts: int = 100,
    white_contour=True,
    radius: float = 1.0,
    n=6,
    ax=None,
    **kwargs,
):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    kwargs.setdefault("c", kwargs.pop("color", "k"))
    kwargs.setdefault("lw", kwargs.pop("linewidth", 1))
    kwargs.setdefault("alpha", 0.3)

    # plot lines
    r_eq, b_proj, theta_proj = compute_projected_ellipse(radius, oblateness, prolateness, theta, obl, inc-jnp.pi/2)
    b_proj = b_proj/r_eq
    
    lat, lon = lon_lat_lines(pts=pts, radius=radius, n=n, oblateness=oblateness, prolateness=prolateness)
    lat = rotate_lines(lat, inc, obl, theta)
    plot_lines(lat, ax=ax,**kwargs)
    lon = rotate_lines(lon, inc, obl, theta)
    plot_lines(lon, ax=ax, **kwargs)
    theta = np.linspace(0, 2 * np.pi, 2 * pts)

    # contour
    sqrt_radius = r_eq
    x = sqrt_radius * np.cos(theta) * np.cos(-theta_proj) - sqrt_radius * b_proj * np.sin(theta) * np.sin(-theta_proj)
    y = sqrt_radius * np.cos(theta) * np.sin(-theta_proj) + sqrt_radius * b_proj * np.sin(theta) * np.cos(-theta_proj)
    
    ax.plot(x, y, **kwargs)
    if white_contour:
        ax.plot(x, y, c="w", lw=3)
