import numpy as np

from jaxoplanet.starry.surface import Surface
from eclipsoid.plotting_utils import graticule
from .bounds import compute_projected_ellipse
from jaxoplanet.starry.ylm import Ylm
from eclipsoid.plotting_utils import render_oblate_surface


def show_surface(
    ylm_surface_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    radius: float = None,
    oblateness: float = 0.0,
    prolateness: float = 0.0,
    **kwargs,
):
    """Show map of a

    Args:
        ylm_surface_body (Surface, SurfaceBody, or EllipsoidalBody): Map or Body with a map
        theta (float, optional): Rotation angle of the map wrt its rotation axis.
        Defaults to 0.0.
        res (int, optional): Resolution of the map render. Defaults to 400.
        n (int, optional): number of latitude and longitude lines to show.
        Defaults to 6.
        ax (matplotlib.pyplot.Axes, optional): plot axes. Defaults to None.
        white_contour (bool, optional): Whether to surround the map by a white border
        (to hide border pixel aliasing). Defaults to True.
        radius (float, optional): Radius of the body. Defaults to None.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    if hasattr(ylm_surface_body, "surface"):
        surface = ylm_surface_body.surface
        if ylm_surface_body.radius is not None:
            radius = ylm_surface_body.radius.magnitude
        else:
            radius = 1.0 if radius is None else radius
        n = int(np.ceil(n * np.cbrt(radius)))
    if hasattr(ylm_surface_body, "oblateness"):
        surface = Surface(y=ylm_surface_body.ylm)
        if ylm_surface_body.oblateness is not None:
            oblateness = ylm_surface_body.oblateness.magnitude
        else:
            oblateness = 0.0
            
        if ylm_surface_body.prolateness is not None:
            prolateness = ylm_surface_body.prolateness.magnitude
        else:
            prolateness = 0.0
             
    else:
        surface = ylm_surface_body
        radius = 1.0 if radius is None else radius
        oblateness = 0.0 if oblateness is None else oblateness
        prolateness = 0.0 if prolateness is None else prolateness

    r_eq, b_proj, theta_proj = compute_projected_ellipse(radius, oblateness, prolateness, theta, surface.obl, surface.inc-np.pi/2)
    ax.imshow(
        render_oblate_surface(
            res,
            theta=theta,
            oblateness=oblateness,
            prolateness=prolateness,
            surface=surface,
        ),
        origin="lower",
        **kwargs,
        extent=(-r_eq, r_eq, -r_eq, r_eq),
    )
    if n is not None:
        graticule(
            surface.inc,
            surface.obl,
            theta,
            oblateness=oblateness,
            prolateness=prolateness,
            radius=radius,
            n=n,
            white_contour=white_contour,
            ax=ax,
        )
    ax.axis(False)

def animate_system(system, t):
    raise NotImplementedError