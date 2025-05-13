from eclipsoid.ellipsoid import EclipsoidSystem, EllipsoidalBody
from eclipsoid.light_curve import eclipsoid_light_curve

from jaxoplanet.starry.surface import Surface
from jaxoplanet.orbits.keplerian import System, Central
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.light_curves import light_curve

import astropy.units as u
import jax.numpy as jnp
import jax
import pytest

@pytest.mark.parametrize(("params", "t"), [
({
    'm_star': 1.37843833,
    'r_star': 1.458,
    'm_planet': (1.157*u.M_jup).to(u.M_sun).value,
    'period':1.7,
    't0': 0.0,
    'inc': jnp.radians(88.0),
    'oblateness': 0.0,
    'prolateness': 0.0,
    'u': jnp.array([0.01,0.01]),
    'rp': 0.017,
}, jnp.linspace(-1.0, 1.0, 1000))])
def test_eclipsoid_spherical(params, t):
    """ Test eclipsoid light curve against jaxoplanet light curve
    for 0 oblateness and 0 prolateness on a tidally locked exoplanet"""
    #putting it all together for NRS1
    central_surface = Surface(inc=jnp.pi/2, obl=0.0, period=jnp.inf, u=params['u'], amplitude=1.0)
    central = Central(radius=params['r_star'], mass=params['m_star'])
    ylm = Ylm.from_dense(jnp.array([1.0]))
    body_surface = Surface(y=ylm, inc=params['inc'], obl=0.0, period=params['period'], amplitude=0.0, normalize=False, phase=-2*jnp.pi*params['t0']/params['period']+jnp.pi)
    eclipsoid_system = EclipsoidSystem(central, central_surface).add_body(radius=params['rp'], mass=params['m_planet'], period=params['period'], surface=body_surface, inclination=params['inc'],
                                                                    oblateness=params['oblateness'], prolateness=params['prolateness'], time_transit=params['t0'])
    #add one since exoplanet/jaxoplanet convention is to return 0 as the mean
    sphere_system = SurfaceSystem(central, central_surface).add_body(radius=params['rp'], mass=params['m_planet'], period=params['period'], surface=body_surface, inclination=params['inc'], time_transit=params['t0'])
    
    lc = light_curve(sphere_system)(t).sum(axis=1)
    
    eclipsoid_lc = eclipsoid_light_curve(eclipsoid_system)(t).sum(axis=1)
    
    assert jnp.allclose(lc, eclipsoid_lc, rtol=1e-7)
    
@pytest.mark.parametrize(("params", "t"), [
({
    'm_star': 1.37843833,
    'r_star': 1.458,
    'm_planet': (1.157*u.M_jup).to(u.M_sun).value,
    'period':1.7,
    't0': 0.0,
    'inc': jnp.radians(88.0),
    'oblateness': 1e-3,
    'prolateness': 0.0, #for a oblate but not prolate planet
    'u': jnp.array([0.01,0.01]),
    'rp': 0.017,
}, jnp.linspace(-1.0, 1.0, 1000)),
({
    'm_star': 1.37843833,
    'r_star': 1.458,
    'm_planet': (1.157*u.M_jup).to(u.M_sun).value,
    'period':1.7,
    't0': 0.0,
    'inc': jnp.radians(88.0),
    'oblateness': 1e-3,
    'prolateness': -1e-3, #for a prolate and slightly oblate planet
    'u': jnp.array([0.01,0.01]),
    'rp': 0.017,
}, jnp.linspace(-1.0, 1.0, 1000)),
({
    'm_star': 1.37843833,
    'r_star': 1.458,
    'm_planet': (1.157*u.M_jup).to(u.M_sun).value,
    'period':1.7,
    't0': 0.0,
    'inc': jnp.radians(88.0),
    'oblateness': 0.0,
    'prolateness': -0.1, #for a prolate but not oblate planet
    'u': jnp.array([0.01,0.01]),
    'rp': 0.017,
}, jnp.linspace(-1.0, 1.0, 1000))])
def test_eclipsoid_grads(params, t):
    """Test eclipsoid light curve gradients to make sure they aren't nan for a tidally locked exoplanet"""
    def func(params, t):
        central_surface = Surface(inc=jnp.pi/2, obl=0.0, period=jnp.inf, u=params['u'], amplitude=1.0)
        central = Central(radius=params['r_star'], mass=params['m_star'])
        ylm = Ylm.from_dense(jnp.array([1.0]))
        body_surface = Surface(y=ylm, inc=params['inc'], obl=0.0, period=params['period'], amplitude=0.0, normalize=False, phase=-2*jnp.pi*params['t0']/params['period']+jnp.pi)
        eclipsoid_system = EclipsoidSystem(central, central_surface).add_body(radius=params['rp'], mass=params['m_planet'], period=params['period'], surface=body_surface, inclination=params['inc'],
                                                                        oblateness=params['oblateness'], prolateness=params['prolateness'], time_transit=params['t0'])
        return eclipsoid_light_curve(eclipsoid_system)(t).sum(axis=1)
    grad_func = jax.jit(jax.jacrev(func))
    grads = grad_func(params, t)
    for n, key in enumerate(grads.keys()):
        assert jnp.all(jnp.isfinite(grads[key])), f"Gradient {key} is not finite"
    