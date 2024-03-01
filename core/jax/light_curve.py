import jax.numpy as jnp
from .utils import gauss_quad
import jax
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet import light_curves, orbits

from .bounds import compute_bounds
from .solution import sT
from jax.tree_util import Partial

def greens_basis_transform(u):
    
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

compute_bounds_vec = jax.jit(jax.vmap(compute_bounds, in_axes=(None, 0,0,None)))
#sT_vec = jax.jit(jax.vmap(sT,in_axes=(0, 0, 0,0,None,0,0,None,None)), static_argnums=8)

def oblate_lightcurve(params,t):
    """_summary_

    Args:
        params (Dict): dictionary containing parameters for the transit model including:
            u: quadratic limb darkening coefficients
            period: period in days
            radius: radius of the planet in units of stellar radius
            bo: impact parameter
            f: oblateness coefficient
        t (Array): _description_
    """
    b = 1-params['f']
    orbit = orbits.TransitOrbit(period=params['period'], radius=params['radius'], impact_param=params['bo'], duration=params['duration'])
    xo, yo = orbit.relative_position(t)[0].magnitude,orbit.relative_position(t)[1].magnitude
    
    xo_rot, yo_rot = xo*jnp.cos(params['theta'])-yo*jnp.sin(params['theta']), xo*jnp.sin(params['theta'])+yo*jnp.cos(params['theta'])
    xis, phis = compute_bounds_vec(b,xo_rot,yo_rot,params['radius'])
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