import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxoplanet import orbits
from jaxoplanet.orbits.keplerian import Central, Body, System, OrbitalBody
from jaxoplanet import units
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg
from jaxoplanet.experimental.starry.basis import U0, A2_inv
from jaxoplanet.experimental.starry.light_curves import rT

from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union
import jpu.numpy as jnpu

from .bounds import compute_bounds
from .solution import sT
from jax.tree_util import Partial
import scipy

class OblateBody(Body):
    """ An oblate body with a given oblateness and obliquity """
    
    f: Optional[Quantity] = units.field(
    default=None, units=ureg.dimensionless
    )
    
    theta: Optional[Quantity] = units.field(
    default=0, units=ureg.radian
    )
    def __check_init__(self) -> None:
        if ((self.f is None) or (self.theta is None)):
            raise ValueError("Oblateness and obliquity must be both defined")
    
class EclipsoidSystem(System):
    
    def __init__(
        self,
        central: Optional[Central] = None,
        *,
        bodies: Iterable[Union[OblateBody, OrbitalBody]] = (),
    ):
        self.central = Central() if central is None else central
        self._body_stack = ObjectStack(
            *(
                b if isinstance(b, OrbitalBody) else OrbitalBody(self.central, b)
                for b in bodies
            )
        )
    
    def add_body(
        self,
        body: Optional[OblateBody] = None,
        central: Optional[Central] = None,
        **kwargs: Any,
    ) -> "EclipsoidSystem":
        body_: Optional[Union[OblateBody, OrbitalBody]] = body
        if body_ is None:
            body_ = OblateBody(**kwargs)
        if central is not None:
            body_ = OrbitalBody(central, body_)
        return EclipsoidSystem(central=self.central, bodies=self.bodies + (body_,))

    @property
    def body_oblateness(self) -> Quantity:
        return self.body_vmap(lambda body: body.f)()
    
    @property
    def body_obliquity(self) -> Quantity:
        return self.body_vmap(lambda body: body.theta)()

class OblateTransitOrbit(orbits.TransitOrbit):
    
    f: Optional[Quantity] = units.field(
    default=None, units=ureg.dimensionless
    )
    
    theta: Optional[Quantity] = units.field(
    default=0, units=ureg.radian
    )
    
    def __init__(
        self,
        *,
        period: Quantity,
        duration: Optional[Quantity] = None,
        speed: Optional[Quantity] = None,
        time_transit: Optional[Quantity] = None,
        impact_param: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
        f: Optional[Quantity] = None,
        theta: Optional[Quantity] = None,
    ):
        super().__init__(period=period, radius=radius, impact_param=impact_param, duration=duration, speed=speed, time_transit=time_transit)
        self.f = f
        self.theta = theta
        
    @property
    def body_oblateness(self) -> Quantity:
        return self.f
    
    @property
    def body_obliquity(self) -> Quantity:
        return self.theta
    
compute_bounds_vec = jax.jit(jax.vmap(compute_bounds, in_axes=(None, 0,0,None)))

def greens_basis_transform(u):
    U = jnp.array([1, *u])
    p_u = (U @ U0(len(u)))
    lmax = jnp.floor(jnp.sqrt(len(p_u))).astype(int)-1
    g_u =  scipy.sparse.linalg.inv(A2_inv(lmax)) @ p_u
    return g_u / (- p_u @ rT(lmax))

def oblate_lightcurve(orbit, u, t):
    obliquity = orbit.body_obliquity.magnitude
    oblateness = orbit.body_oblateness.magnitude
    b = 1.-oblateness
    r = orbit.radius / orbit.central_radius
    xo, yo = orbit.relative_position(t)[0].magnitude,orbit.relative_position(t)[1].magnitude
    xo_rot, yo_rot = xo*jnp.cos(obliquity)-yo*jnp.sin(obliquity), xo*jnp.sin(obliquity)+yo*jnp.cos(obliquity)
    xis, phis = compute_bounds_vec(b,xo_rot,yo_rot,r.magnitude)
    g_u = greens_basis_transform(u)
    ns = np.arange(len(g_u))
    lcs = jnp.zeros((len(g_u),len(t)))
    zeros = lambda phi1, phi2, xi1, xi2, b, xo, yo, ro: jnp.zeros(len(xo))
    for n in ns:
        cond = g_u[n]!=0
        sT_vec = jax.jit(jax.vmap(Partial(sT, n=n),in_axes=(0, 0, 0,0,None,0,0,None)))
        lcs = lcs.at[n].set(
            jax.lax.cond(cond, sT_vec, zeros, phis[:,0],phis[:,1], xis[:,0],xis[:,1], b,xo_rot,yo_rot,r.magnitude))

    lcs = jnp.array(lcs).T@g_u
    return lcs
    
    
def ellipsoid_lightcurve(orbit, t):
    raise NotImplementedError("Ellipsoid lightcurve not implemented yet")
    

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