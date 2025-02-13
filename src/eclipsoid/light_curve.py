import jax
import jax.experimental
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxoplanet import orbits

from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.light_curves import rT
from jaxoplanet.starry.light_curves import surface_light_curve as circular_surface_light_curve
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.surface import Surface
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet import units
import jpu.numpy as jnpu

from collections.abc import Callable
from jaxoplanet.types import Array, Quantity
from typing import Any, Optional, Union
from jaxoplanet.units import quantity_input, unit_registry as ureg

from .bounds import compute_bounds, compute_projected_ellipse
from .solution import sT, pT, q_integral, solution_vector
from .ellipsoid import EclipsoidSystem
from jax.tree_util import Partial
from functools import partial
import scipy

    
compute_bounds_oblate = jax.jit(jax.vmap(compute_bounds, in_axes=(None, 0,0,None)))

def greens_basis_transform(u):
    U0 = jnp.array([1, *u])
    p_u = (U0 @ U(len(u)))
    lmax = np.floor(np.sqrt(len(p_u))).astype(int)-1
    A2i = scipy.sparse.linalg.inv(A2_inv(lmax))
    A2i = jax.experimental.sparse.BCOO.from_scipy_sparse(A2i)
    g_u =  A2i @ p_u
    return g_u / (- p_u @ rT(lmax))

def limb_dark_oblate_lightcurve(orbit, u, oblateness, obliquity):
    """Compute a simple limb darkened light curve where the planet is oblate (not rotating) and neither the star nor planet have a time variable surface map"""

    b = 1.-oblateness
    #convert to r_eq for computing intersection points
    @vectorize
    def impl(time):
        t = time
        r_eq = (jnpu.sqrt(orbit.radius**2/b) / orbit.central_radius).magnitude
        #TODO: check this works with central_radius != 0
        xo, yo, zo = orbit.relative_position(t)
        #hack to prevent eclipse when planet is behind star
        r_eq = jnp.where(jnp.less_equal(zo.magnitude, 0.0), 0.0, r_eq)
        #hacks to get it to work with both keplerian and transit orbit classes
        #keplerian system wants to return a vector (one for each planet)
        #transit orbit wants to return a scalar
        # TODO implement autobatching over multiple planets where keplerian orbit given
        xo_rot, yo_rot = xo.magnitude*jnp.cos(obliquity)-yo.magnitude*jnp.sin(obliquity), xo.magnitude*jnp.sin(obliquity)+yo.magnitude*jnp.cos(obliquity)
        xis, phis = compute_bounds(jnp.squeeze(b),jnp.squeeze(xo_rot),jnp.squeeze(yo_rot),jnp.squeeze(r_eq))
        g_u = greens_basis_transform(u)
        ns = np.arange(len(g_u))
        lcs = jnp.zeros((len(g_u)))
        zeros = lambda phi1, phi2, b, xo, yo, ro: 0.
        for n in ns:
            cond = g_u[n]!=0
            pT_func = Partial(pT, n=n)
            lcs = lcs.at[n].set(
                jax.lax.cond(cond, pT_func, zeros, phis[0],phis[1], jnp.squeeze(b),xo_rot,yo_rot,jnp.squeeze(r_eq)))
        lmax = np.floor(np.sqrt(len(g_u))).astype(int)-1
        lcs = jnp.array(lcs+q_integral(lmax, xis)).T@g_u
        return lcs
    return impl

""" 
Functions for computing light curves taken from jaxoplanet repo and modified to work with ellipsoidal planet sT
"""

def eclipsoid_light_curve(system: EclipsoidSystem, order: int = 30
) -> Callable[[Quantity], tuple[Optional[Array], Optional[Array]]]:
    
    central_bodies_lc = jax.vmap(
        surface_light_curve, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)
    )
    @partial(system.surface_vmap, in_axes=(0, 0, 0, 0, 0, 0, None))
    def compute_body_light_curve(surface, radius, oblateness, prolateness, x, y, z, time):
        if surface is None:
            return 0.0
        else:
            theta = surface.rotational_phase(time.magnitude)
            return ellipsoidal_surface_light_curve(
                surface,
                (system.central.radius / radius).magnitude,
                oblateness.magnitude,
                prolateness.magnitude,
                (x / radius).magnitude,
                (y / radius).magnitude,
                (z / radius).magnitude,
                theta,
                order,
            )
            
    @quantity_input(time=ureg.day)
    @vectorize
    def light_curve_impl(time: Quantity) -> Array:
        if system.central_surface is None:
            central_light_curves = jnp.array([0.0])
        else:
            theta = system.central_surface.rotational_phase(time.magnitude)
            central_radius = system.central.radius
            central_phase_curve = surface_light_curve(
                system.central_surface, theta=theta, order=order
            )
            if len(system.bodies) > 0:
                xos, yos, zos = system.relative_position(time)
                n = len(xos.magnitude)
                central_light_curves = central_bodies_lc(
                    system.central_surface,
                    (system.radius / central_radius).magnitude,
                    system.oblateness.magnitude,
                    system.prolateness.magnitude,
                    (xos / central_radius).magnitude,
                    (yos / central_radius).magnitude,
                    (zos / central_radius).magnitude,
                    system.surface_vmap(lambda surface: surface.inc)(),
                    system.surface_vmap(lambda surface: surface.obl)(),
                    system.surface_vmap(lambda surface: surface.rotational_phase(time.magnitude))(),
                    theta,
                    order,
                )

                if n > 1 and central_light_curves is not None:
                    central_light_curves = central_light_curves.sum(
                        0
                    ) - central_phase_curve * (n - 1)
                    central_light_curves = jnp.expand_dims(central_light_curves, 0)

                body_light_curves = compute_body_light_curve(
                    system.radius, system.oblateness, system.prolateness, -xos, -yos, -zos, time
                )

                return jnp.hstack([central_light_curves, body_light_curves])
            else:
                return jnp.array([central_phase_curve])
    return light_curve_impl

def ellipsoidal_surface_light_curve(surface: Surface, 
                                    r: float = None,
                                    oblateness: float = 0.0,
                                    prolateness:  float = 0.0,
                                    x:  float = None, 
                                    y:  float = None, 
                                    z:  float = None, 
                                    theta: float = 0.0,
                                    order: int = 20):
    r_eq, b_proj, theta_proj = compute_projected_ellipse(r, oblateness, prolateness, theta+surface.phase, surface.obl, surface.inc-jnp.pi/2)
    ellipse_area_factor = (r_eq*b_proj)/r**2
    return circular_surface_light_curve(surface,r, x, y, z, theta)*ellipse_area_factor # TODO: multiply by projection factor for ellipsoidal area

def surface_light_curve(surface: Surface, 
                                    r: float = None,
                                    oblateness: float = 0.0,
                                    prolateness:  float = 0.0,
                                    x:  float = None, 
                                    y:  float = None, 
                                    z:  float = None,
                                    body_inc: float = 0.0,
                                    body_obl: float = 0.0,
                                    body_theta: float = 0.0,
                                    theta: float = 0.0,
                                    order: int = 20):
    
    
    rT_deg = rT(surface.deg)
    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z
    r_eq, b_proj, theta_proj = 0, 0, 0
    sT = jnp.zeros(surface.deg)
    xis, phis = jnp.zeros(2), jnp.zeros(2)
    xo_rot, yo_rot = 0, 0
    
    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        design_matrix_p = rT_deg
    # occulting body
    else:
        r_eq, b_proj, theta_proj = compute_projected_ellipse(r, oblateness, prolateness, body_theta, body_obl, body_inc-jnp.pi/2)
        #numerical stability for oblateness of 0
        b_proj = b_proj/r_eq
        b = jnp.sqrt(jnp.square(x) + jnp.square(y))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r_eq), jnp.less_equal(z, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(x, y)

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 0.0, r)
        close_to_1 = jnp.isclose(b_proj, 1.0, rtol=10*jnp.finfo(jnp.float64).eps)
        
        #fixing numerical stability issues for oblateness of 0
        b_proj = jnp.where(close_to_1, 1.0, b_proj)
        theta_proj = jnp.where(close_to_1, jnp.pi/2, theta_proj)
        sT = solution_vector(surface.deg)(b_proj, theta_proj, x, y, r_eq)
        if surface.deg > 0:
            A2 = scipy.sparse.linalg.inv(A2_inv(surface.deg))
            A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([[1]])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    if surface.ydeg == 0:
        rotated_y = surface.y.todense()
    else:
        rotated_y = left_project(
            surface.ydeg,
            surface.inc,
            surface.obl,
            theta + surface.phase,
            theta_proj,
            surface.y.todense(),
        )

    # limb darkening
    if surface.udeg == 0:
        p_u = Pijk.from_dense(jnp.array([1]))
    else:
        u = jnp.array([1, *surface.u])
        p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)

    # surface map * limb darkening map
    A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)
    p_y = p_y * p_u

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    return surface.amplitude * (p_y.tosparse() @ design_matrix_p) * norm