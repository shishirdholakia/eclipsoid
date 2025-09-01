from eclipsoid.ellipsoid import EclipsoidSystem, EllipsoidalBody
from eclipsoid.light_curve import eclipsoid_light_curve
from eclipsoid.utils import scipy_surface_min_intensity, mollweide_grid
from eclipsoid.utils import fibonacci_grid, surface_min_intensity

from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.light_curves import light_curve

import astropy.units as u
import jax.numpy as jnp
import jax
import pytest

ylm_surface = jnp.array([1.0,  0.,  8.08972083e-01,  0.,
        0., 0.,  1.51928272e-02,  0.,
        1.48219417e-02, 0., 0., 0.,
       -2.59072086e-01,  0.,  3.82095585e-03, 0.,
        0.,  0.,  0.,  0.,
        1.40144823e-02,  0., -2.77728498e-03,  0.,
       -4.30075310e-04])

surface = Surface(y=Ylm.from_dense(ylm_surface), inc=jnp.radians(90), obl=jnp.radians(0), period=1.0, amplitude=1.0, normalize=True)

@pytest.mark.parametrize(("surface"), [
    surface
])
def test_surface_min_intensity(surface):
    """ Test that the surface_min_intensity function returns the same result as the scipy version"""
    oversample = 4
    lmax = 4
    (jax_lat, jax_lon), jax_min = surface_min_intensity(surface, oversample, lmax)
    (scipy_lat, scipy_lon), scipy_min = scipy_surface_min_intensity(surface, oversample, lmax)
    assert jnp.allclose(jax_lat, scipy_lat, rtol=1e-7) & jnp.allclose(jax_lon, scipy_lon, rtol=1e-7) & jnp.allclose(jax_min, scipy_min, rtol=1e-7)