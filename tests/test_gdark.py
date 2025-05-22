import jax.numpy as jnp
from eclipsoid.gdark import gdark_to_ylms
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry import Surface

import numpy as np
from astropy import units as u
from pathlib import Path

# Get the current script's folder
base_dir = Path(__file__).parent

import pytest

def test_gdark_to_ylms():
    wav = 770 * u.nm.to(u.m)
    omega = 0.2
    beta = 0.22
    tpole = 7700

    # Call the function
    ylm_coeffs = gdark_to_ylms(wav, omega, beta, tpole)
    ylm = Ylm.from_dense(ylm_coeffs)
    
    surface = Surface(y=ylm, inc=jnp.pi/2, obl=0,normalize=True)

    starry_ylm = np.load(base_dir / 'starry_gdark_map.npy')
    print(starry_ylm)
    print(surface.y.todense())
    # Check that the coefficients are real numbers
    assert jnp.allclose(surface.y.todense(), starry_ylm, rtol=0, atol=1e-5)