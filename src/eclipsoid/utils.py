import numpy as np
import jax.numpy as jnp
import jax
from numpy.polynomial.legendre import leggauss

import numpy as np
from scipy.optimize import minimize

from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.utils import y1d_to_2d, y2d_to_1d, C
from functools import partial



def gauss_quad(f, a, b, n):
    """
    Computes the definite integral of a function using Gaussian quadrature with n points.
    :param f: function to integrate
    :param a: lower limit of integration
    :param b: upper limit of integration
    :param n: number of quadrature points
    :return: definite integral of f from a to b
    """
    x, w = leggauss(n)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    return 0.5 * (b - a) * jnp.sum(w * f(x))

def poly(x):
    return jnp.sin(x)**4

  
@jax.custom_jvp
def zero_safe_sqrt(x):
    return jnp.sqrt(x)

@zero_safe_sqrt.defjvp
def zero_safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = jnp.sqrt(x)
    cond = jnp.less_equal(x, 10 * jnp.finfo(jax.dtypes.result_type(x)).eps)
    val_where = jnp.where(cond, jnp.ones_like(x), x)
    denom = val_where**0.5
    tangent_out = 0.5 * x_dot / denom
    return primal_out, tangent_out  # Return only primal and tangent


@jax.custom_jvp
def zero_safe_arctan2(x, y):
    return jnp.arctan2(x, y)


@zero_safe_arctan2.defjvp
def zero_safe_arctan2_jvp(primals, tangents):
    (x, y) = primals
    (x_dot, y_dot) = tangents
    primal_out = zero_safe_arctan2(x, y)
    tol = 10 * jnp.finfo(jax.dtypes.result_type(x)).eps
    cond_x = jnp.logical_and(x > -tol, x < tol)
    cond_y = jnp.logical_and(y > -tol, y < tol)
    cond = jnp.logical_and(cond_x, cond_y)
    denom = jnp.where(cond, jnp.ones_like(x), x**2 + y**2)
    tangent_out = (y * x_dot - x * y_dot) / denom
    return primal_out, tangent_out


@jax.custom_jvp
def zero_safe_power(x, y):
    return x**y


@zero_safe_power.defjvp
def zero_safe_power_jvp(primals, tangents):
    (x, y) = primals
    (x_dot, _) = tangents
    primal_out = zero_safe_power(x, y)
    tol = 10 * jnp.finfo(jax.dtypes.result_type(x)).eps
    cond_x = jnp.logical_and(x > -tol, x < tol)
    cond_y = jnp.less(y, 1.0)
    cond = jnp.logical_and(cond_x, cond_y)
    denom = jnp.where(cond, jnp.ones_like(x), x)
    tangent_out = y * primal_out * x_dot / denom
    return primal_out, tangent_out


#########################################################################

# Utility function to ensure surface maps are physical 
# (non-negative intensity everywhere)

#########################################################################

def mollweide_grid(oversample, lmax):
    """Create an approximately uniform grid on the sphere using the Mollweide projection."""
    npts = oversample * lmax ** 2
    nlat = int(jnp.sqrt(npts))
    nlon = int(npts / nlat)
    lats = jnp.linspace(-jnp.pi / 2 + 1e-3, jnp.pi / 2 - 1e-3, nlat)
    lons = jnp.linspace(0, 2 * jnp.pi, nlon, endpoint=False)
    grid = [(lat, lon) for lat in lats for lon in lons]
    return grid

def scipy_surface_min_intensity(surface: Surface, oversample: int = 4, lmax: int = 4):
    """Find global minimum intensity on the surface."""
    grid = mollweide_grid(oversample, lmax)

    @jax.jit
    def objective(coord):
        lat, lon = coord
        return surface.intensity(lat, lon)

    min_val = jnp.inf
    min_coord = None
    
    for coord in grid:
        #try:
            res = minimize(objective, np.array(coord), method="BFGS")
            if res.success and res.fun < min_val:
                min_val = res.fun
                min_coord = res.x
        #except Exception as e:
            #print(f"Minimization failed at {coord} with error: {e}")
            #continue

    return min_coord, min_val


# -------- Grid (equal-area-ish Fibonacci; tiny + deterministic) --------
def mollweide_grid(oversample: int, lmax: int):
    n = oversample * (lmax ** 2)
    i = jnp.arange(n)
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0
    theta = jnp.arccos(1.0 - 2.0 * (i + 0.5) / n)        # colatitude ∈ [0, π]
    lon = (2.0 * jnp.pi) * ((i / phi) % 1.0)             # [0, 2π)
    lat = (jnp.pi / 2.0) - theta                         # latitude ∈ [-π/2, π/2]
    return jnp.stack([lat, lon], axis=-1)                # (n, 2)

# -------- Helpers to keep angles in-range (differentiable) -------------
def _wrap_lon(lon):
    # Wrap to [0, 2π)
    two_pi = 2.0 * jnp.pi
    return lon - two_pi * jnp.floor(lon / two_pi)

def _clip_lat(lat, eps=1e-6):
    # Clip to (-pi/2, pi/2) by tiny epsilon to avoid singularities
    return jnp.clip(lat, -jnp.pi/2 + eps, jnp.pi/2 - eps)

def _project_coord(x):
    lat = _clip_lat(x[0])
    lon = _wrap_lon(x[1])
    return jnp.array([lat, lon])


# -------- Fixed-iteration damped Newton in 2D (lat, lon) --------------
@partial(jax.jit, static_argnames=("oversample", "lmax", "newton_iters", "damping", "step"))
def surface_min_intensity(surface, oversample: int, lmax: int,
                         newton_iters: int = 12, damping: float = 1e-3, step: float = 1.0,
                         tau_softmin: float = None):
    """
    Fully JAX, end-to-end differentiable approximate global min:
      1) seed from tiny equal-area grid
      2) run fixed M Newton steps from each seed in parallel
      3) take global min across seeds

    Args:
      surface: jaxoplanet Surface-like object with .intensity(lat, lon)
      oversample, lmax: define N = oversample * lmax^2 seeds
      newton_iters: fixed Newton iterations per seed (no line search)
      damping: Levenberg-Marquardt diagonal added to Hessian
      step: Newton step scaling (e.g., 1.0 or 0.5)
      tau_softmin: if not None, additionally return soft-min value with temperature tau_softmin

    Returns:
      (lat_min, lon_min), min_val, (optional soft_min_val)
    """

    # Scalar objective taking a 2-vector x = [lat, lon]
    def f(x):
        lat, lon = x
        return surface.intensity(lat, lon)

    # Grad & Hessian w.r.t. x
    grad_f = jax.grad(f)
    hess_f = jax.hessian(f)

    def newton_one_seed(x0):
        def body(_, x):
            g = grad_f(x)                                # (2,)
            H = hess_f(x)                                # (2,2)
            H_damped = H + damping * jnp.eye(2, dtype=x.dtype)

            # Solve H p = g  (Newton step uses p = H^{-1} g)
            p = jnp.linalg.solve(H_damped, g)
            x_new = x - step * p

            # keep angles valid
            return _project_coord(x_new)

        xT = jax.lax.fori_loop(0, newton_iters, body, _project_coord(x0))
        return xT, f(xT)

    # Seeds
    seeds = mollweide_grid(oversample, lmax)  # (N,2)

    # Run Newton from all seeds in parallel, no Python loops, fixed iteration count
    xs, vals = jax.vmap(newton_one_seed)(seeds)  # xs: (N,2), vals: (N,)

    # Take global min (subgradient is OK almost everywhere; for extra smoothness use softmin)
    i_min = jnp.argmin(vals)
    x_min = xs[i_min]
    v_min = vals[i_min]

    if tau_softmin is None:
        return x_min, v_min
    else:
        # Smooth surrogate for min (good for stable gradients in penalties)
        # softmin_tau(z) = -tau * log( sum_i exp(-z_i / tau) )
        # As tau -> 0, softmin -> min
        z = vals
        m = jnp.min(z)
        soft = -tau_softmin * (jnp.log(jnp.sum(jnp.exp(-(z - m)/tau_softmin))) + m / tau_softmin)
        return x_min, v_min, soft


### Spherical Harmonic Transform Utilities with S2FFT

def ylm_to_pixels(ylm_map, lmax):
    import s2fft
    ylm_2d = y1d_to_2d(lmax, ylm_map)@C(lmax)
    pixels = s2fft.inverse_jax(ylm_2d, lmax+1, reality=True)
    return pixels

def pixels_to_ylm(pixel_map, lmax):
    import s2fft
    ylm_2d = s2fft.forward_jax(pixel_map, 3, reality=True)
    return ylm_2d@C(lmax).T


if __name__=="__main__":
    print(gauss_quad(poly,-1.,2.,10)-1/32*(36 - 8*np.sin(2) - 7*np.sin(4) + np.sin(8)))