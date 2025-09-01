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
            radius = ylm_surface_body.radius
        else:
            radius = 1.0 if radius is None else radius
        n = int(np.ceil(n * np.cbrt(radius)))
    if hasattr(ylm_surface_body, "oblateness"):
        surface = Surface(y=ylm_surface_body.ylm)
        if ylm_surface_body.oblateness is not None:
            oblateness = ylm_surface_body.oblateness
        else:
            oblateness = 0.0
            
        if ylm_surface_body.prolateness is not None:
            prolateness = ylm_surface_body.prolateness
        else:
            prolateness = 0.0
             
    else:
        surface = ylm_surface_body
        radius = 1.0 if radius is None else radius
        oblateness = 0.0 if oblateness is None else oblateness
        prolateness = 0.0 if prolateness is None else prolateness

    r_eq, b_proj, theta_proj = compute_projected_ellipse(radius, oblateness, prolateness, theta, surface.obl, surface.inc-np.pi/2)
    im = ax.imshow(
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
    return im


def show_surface_anim(
    ylm_surface_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    radius: float = None,
    oblateness: float = 0.0,
    prolateness: float = 0.0,
    animate: bool = False,
    interval: int = 100,
    repeat: bool = True,
    colorbar: bool = False,
    **kwargs,
):
    """
    Show map of a body. If `animate=True` and `theta` is an array, animate the body's rotation.
    
    Args:
        ylm_surface_body: Surface, SurfaceBody, or EllipsoidalBody.
        theta: Single angle (float) or array of angles (for animation).
        res: Resolution of the map render.
        n: Number of latitude/longitude lines.
        ax: matplotlib axis.
        white_contour: Whether to draw white border.
        radius: Radius of the body.
        oblateness: Flattening along y.
        prolateness: Elongation along z axis.
        animate: Animate if True and theta is array.
        interval: Delay between frames in ms.
        repeat: Whether animation should loop.
        **kwargs: Passed to `imshow`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if hasattr(ylm_surface_body, "surface"):
        surface = ylm_surface_body.surface
        radius = (
            ylm_surface_body.radius
            if ylm_surface_body.radius is not None
            else (1.0 if radius is None else radius)
        )
        n = int(np.ceil(n * np.cbrt(radius)))
    elif hasattr(ylm_surface_body, "oblateness"):
        surface = Surface(y=ylm_surface_body.ylm)
        oblateness = (
            ylm_surface_body.oblateness
            if ylm_surface_body.oblateness is not None
            else 0.0
        )
        prolateness = (
            ylm_surface_body.prolateness
            if ylm_surface_body.prolateness is not None
            else 0.0
        )
    else:
        surface = ylm_surface_body
        radius = 1.0 if radius is None else radius
        oblateness = 0.0 if oblateness is None else oblateness
        prolateness = 0.0 if prolateness is None else prolateness

    # Handle array-like theta
    theta_array = np.atleast_1d(theta)

    def render_frame(theta_i):
        ax.clear()
        r_major, r_minor, _ = compute_projected_ellipse(
            radius,
            oblateness,
            prolateness,
            theta_i,
            surface.obl,
            surface.inc - np.pi / 2,
        )
        im = ax.imshow(
            render_oblate_surface(
                res,
                theta=theta_i,
                oblateness=oblateness,
                prolateness=prolateness,
                surface=surface,
            ),
            origin="lower",
            **kwargs,
            extent=(-r_major, r_major, -r_major, r_major),
        )
        if n is not None:
            graticule(
                surface.inc,
                surface.obl,
                theta_i,
                oblateness=oblateness,
                prolateness=prolateness,
                radius=radius,
                n=n,
                white_contour=white_contour,
                ax=ax,
            )
        ax.axis(False)
        ax.set_xlim(-r_minor+prolateness-oblateness, r_minor-prolateness+oblateness)
        ax.set_ylim(-r_minor+prolateness-oblateness, r_minor-prolateness+oblateness)
        if colorbar:
            plt.colorbar(im, ax=ax)

    if animate and theta_array.size > 1:
        anim = FuncAnimation(
            fig,
            lambda i: render_frame(theta_array[i]),
            frames=len(theta_array),
            interval=interval,
            repeat=repeat,
        )
        return anim  # Return the animation object so caller can display or save it
        
    else:
        render_frame(theta_array[0])


def animate_system(system, t):
    """ System animation function (rewrite from system.show in starry)

    Args:
        system (EclipsoidSystem): A system in eclipsoid
        t (Array): time array


    """
    # Render the maps & get the orbital positions
    
    #render the primary map
    
    #render the secondary maps
    
    
    # Convert to units of the primary radius


    # Set up the plot

    # Render the first frame
    
    # Animation


    raise NotImplementedError("Animation not implemented yet.")