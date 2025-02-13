import jax
import jax.experimental
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxoplanet import orbits

from jaxoplanet.starry.basis import U, A2_inv
from jaxoplanet.starry.light_curves import rT
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet import units
import jpu.numpy as jnpu

from ..bounds import compute_bounds
from ..solution import pT, sT, q_integral
from jax.tree_util import Partial
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

def oblate_lightcurve_dict(params,t):
    """_summary_

    Args:
        params (Dict): dictionary containing parameters for the transit model including:
            u: quadratic limb darkening coefficients
            period: period in days
            radius: equatorial radius of the planet in units of stellar radius
            t0: time of transit in days
            bo: impact parameter
            f: oblateness coefficient
            duration: duration of transit in days
        t (Array): _description_
    """
    b = 1-params['f']
    orbit = orbits.TransitOrbit(period=params['period'], time_transit=params['t0'], radius_ratio=params['radius']*jnp.sqrt(b), impact_param=params['bo'], duration=params['duration'])
    
    @vectorize
    def impl(time):
        t = time
        r_eq = (jnpu.sqrt(orbit.radius**2/b) / orbit.central_radius).magnitude
        #TODO: check this works with central_radius != 0
        xo, yo = orbit.relative_position(t)[0].magnitude,orbit.relative_position(t)[1].magnitude
        #hacks to get it to work with both keplerian and transit orbit classes
        #keplerian system wants to return a vector (one for each planet)
        #transit orbit wants to return a scalar
        # TODO implement autobatching over multiple planets where keplerian orbit given
        xo_rot, yo_rot = xo*jnp.cos(params['theta'])-yo*jnp.sin(params['theta']), xo*jnp.sin(params['theta'])+yo*jnp.cos(params['theta'])
        xis, phis = compute_bounds(jnp.squeeze(b),jnp.squeeze(xo_rot),jnp.squeeze(yo_rot),jnp.squeeze(r_eq))
        g_u = greens_basis_transform(params['u'])
        ns = np.arange(len(g_u))
        lcs = jnp.zeros((len(g_u)))
        zeros = lambda phi1, phi2, b, xo, yo, ro: 0.
        for n in ns:
            cond = g_u[n]!=0
            pT_vec = Partial(pT, n=n)
            lcs = lcs.at[n].set(
                jax.lax.cond(cond, pT_vec, zeros, phis[0],phis[1], jnp.squeeze(b),xo_rot,yo_rot,jnp.squeeze(r_eq)))
        lmax = np.floor(np.sqrt(len(g_u))).astype(int)-1
        lcs = jnp.array(lcs+q_integral(lmax, xis)).T@g_u
        return lcs
    return impl(t)

def oblate_lightcurve_numerical(orbit, u):
    obliquity = orbit.body_obliquity.magnitude
    oblateness = orbit.body_oblateness.magnitude
    b = 1.-oblateness
    #convert to r_eq for computing intersection points
    @vectorize
    def impl(time):
        t = time
        r_eq = (jnpu.sqrt(orbit.radius**2/b) / orbit.central_radius).magnitude
        #TODO: check this works with central_radius != 0
        xo, yo = orbit.relative_position(t)[0].magnitude,orbit.relative_position(t)[1].magnitude
        #hacks to get it to work with both keplerian and transit orbit classes
        #keplerian system wants to return a vector (one for each planet)
        #transit orbit wants to return a scalar
        # TODO implement autobatching over multiple planets where keplerian orbit given
        xo_rot, yo_rot = xo*jnp.cos(obliquity)-yo*jnp.sin(obliquity), xo*jnp.sin(obliquity)+yo*jnp.cos(obliquity)
        xis, phis = compute_bounds(jnp.squeeze(b),jnp.squeeze(xo_rot),jnp.squeeze(yo_rot),jnp.squeeze(r_eq))
        g_u = greens_basis_transform(u)
        ns = np.arange(len(g_u))
        lcs = jnp.zeros((len(g_u)))
        zeros = lambda phi1, phi2, xi1, xi2, b, xo, yo, ro: 0.
        for n in ns:
            cond = g_u[n]!=0
            sT_vec = Partial(sT, n=n)
            lcs = lcs.at[n].set(
                jax.lax.cond(cond, sT_vec, zeros, phis[0],phis[1], xis[0],xis[1], jnp.squeeze(b),xo_rot,yo_rot,jnp.squeeze(r_eq)))
        lcs = jnp.array(lcs).T@g_u
        return lcs
    return impl

#sT_vec = jax.jit(jax.vmap(sT,in_axes=(0, 0, 0,0,None,0,0,None,None)), static_argnums=8)

def legacy_greens_basis_transform(u):
    
    """ Returns the star's flux in Green's basis
    given quadratic limb darkening coefficients"""
    assert u.shape[0]==2
    g = jnp.zeros(9)
    g = g.at[0].set(1-u[0]-2*u[1])
    g = g.at[2].set(u[0] + 2*u[1])
    g = g.at[4].set(u[1]/3)
    g = g.at[8].set(u[1])
    #I don't know why the normalization constant is negative
    return g/(-jnp.pi*(1-u[0]/3-u[1]/6))

def legacy_oblate_lightcurve(params,t):
    """_summary_

    Args:
        params (Dict): dictionary containing parameters for the transit model including:
            u: quadratic limb darkening coefficients
            period: period in days
            radius: equatorial radius of the planet in units of stellar radius
            t0: time of transit in days
            bo: impact parameter
            f: oblateness coefficient
            duration: duration of transit in days
        t (Array): _description_
    """
    b = 1-params['f']
    orbit = orbits.TransitOrbit(period=params['period'], time_transit=params['t0'], radius_ratio=params['radius']*jnp.sqrt(b), impact_param=params['bo'], duration=params['duration'])
    xo, yo = orbit.relative_position(t)[0].magnitude,orbit.relative_position(t)[1].magnitude
    
    xo_rot, yo_rot = xo*jnp.cos(params['theta'])-yo*jnp.sin(params['theta']), xo*jnp.sin(params['theta'])+yo*jnp.cos(params['theta'])
    xis, phis = compute_bounds_oblate(b,xo_rot,yo_rot,params['radius'])
    g = greens_basis_transform(params['u'])
    ns = np.arange(len(g))
    lcs = jnp.zeros((len(g),len(t)))
    
    zeros = lambda phi1, phi2, xi1, xi2, b, xo, yo, ro: jnp.zeros(len(xo))
    for n in ns:
        cond = g[n]!=0
        sT_vec = jax.jit(jax.vmap(Partial(sT, n=n),in_axes=(0, 0, 0,0,None,0,0,None)))
        lcs = lcs.at[n].set(
            jax.lax.cond(cond, sT_vec, zeros, phis[:,0],phis[:,1], xis[:,0],xis[:,1], b,xo_rot,yo_rot,params['radius']))
        
    lcs = jnp.array(lcs).T@g
    
    return lcs

