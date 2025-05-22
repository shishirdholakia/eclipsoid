from scipy.constants import h, c
from scipy.constants import k as kb

import jax.numpy as jnp
from jaxoplanet.experimental import calc_poly_coeffs
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.starry.core.rotation import dot_rotation_matrix

from jax.scipy.special import lpmn



def planck(l, t):
    return 2.*h*c**2/l**5/(jnp.exp(h*c/(l*kb*t)) - 1.)

def f(omega):
    return 1- 2/(omega**2 + 2)


def f_eff(omega, i_star):
    inc = jnp.radians(i_star)
    return 1 - jnp.sqrt((1-f(omega))**2 * jnp.cos(inc)**2 + jnp.sin(inc)**2)


def flux_oblate(z, wav, omega, beta, tpole):
    a= z**2
    b = (1.-f(omega))
    
    temp = tpole * b**(2*beta) * ((-a*b**2 + (a-1)*(-(omega**2)*(a*b**2 - a + 1)**(3./2.) + 1.)**2.) / (-a*b**2 + a - 1)**3)**(beta/2)
    return planck(wav, temp)

def flux_spherical(z, wav, omega, beta, tpole):
    
    temp = tpole * ((omega**4 - 2*omega**2)*(1-z**2) + 1.)**(beta/2)
    return planck(wav, temp)

def legendre_polynomials(fdeg, z):
    """Compute Legendre polynomials P_0 to P_fdeg evaluated at z."""
    z = jnp.atleast_1d(z)
    P = [jnp.ones_like(z)]                         # P_0(z) = 1
    if fdeg == 0:
        return jnp.stack(P, axis=1)

    P.append(z)                                    # P_1(z) = z
    for l in range(1, fdeg):
        P_lm1 = P[-1]
        P_lm2 = P[-2]
        P_l = ((2 * l + 1) * z * P_lm1 - l * P_lm2) / (l + 1)
        P.append(P_l)

    return jnp.stack(P, axis=1)  # shape: (len(z), fdeg+1)


def gdark_to_ylms(wav, omega, beta, tpole, degree=4):
    """
    Calculate gravity darkening for a pole-on star as a "limb darkening" effect,
    then convert to spherical harmonics and rotate to the equator-on frame
    
    Parameters
    ----------
    wav : array-like
        The wavelength in meters.
    omega : float
        The angular velocity in radians per second.
    beta : float
        The exponent for the temperature distribution.
    tpole : float
        The temperature at the pole in Kelvin.
    
    Returns
    -------
    array-like
        The spherical harmonics coefficients for the gravity darkening effect.
    """
    smoothing=0.0
    eps4=1e-9
    npts = 4 * (degree + 1) ** 2
    z = jnp.linspace(-1, 1, npts)

    # Compute matrix B
    l_arr = jnp.arange(degree + 1)
    P = legendre_polynomials(degree, z)  # shape (npts, fdeg+1)
    B = P * jnp.sqrt(2 * l_arr + 1)    # each column scaled

    # Solve (B^T B + eps4 * I) A = B^T
    BtB = B.T @ B
    regularized = BtB + eps4 * jnp.eye(degree + 1)
    A = jnp.linalg.solve(regularized, B.T)

    # Compute smoothing factor S
    idx = l_arr * (l_arr + 1)
    S = jnp.exp(-0.5 * idx * smoothing ** 2)

    SHT = S[:, None] * A  # final transform matrix
    
    gdark_profile = flux_oblate(z, wav, omega, beta, tpole)
    ylm = jnp.zeros((degree + 1) ** 2)
    ylm_subarray = jnp.dot(SHT, gdark_profile)
    ylm = ylm.at[idx].set(ylm_subarray)
    return dot_rotation_matrix(degree, 1.0, 0.0, 0.0, jnp.pi/2)(ylm)

    
    