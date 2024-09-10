import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.integrate import quad
from .utils import gauss_quad, zero_safe_sqrt
from jax.tree_util import Partial
import jax
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxoplanet.types import Array
from collections.abc import Callable
from functools import partial
from .bounds import compute_bounds

def G(n):
    """
    Return the anti-exterior derivative of the nth term of the Green's basis.
    This is a two-dimensional (Gx, Gy) vector of functions of x and y.
    
    """
    # Get the mu, nu indices
    l = np.floor(np.sqrt(n)).astype(int)
    m = n - l * l - l
    mu = l - m
    nu = l + m

    # NOTE: The abs prevents NaNs when the argument of the sqrt is
    # zero but floating point error causes it to be ~ -eps.
    z = lambda x, y: zero_safe_sqrt(jnp.maximum(1e-24, jnp.abs(1 - x ** 2 - y ** 2)))
    
    if nu % 2 == 0:
        
        G = (lambda x, y: 0, lambda x, y: x ** (0.5 * (mu + 2)) * y ** (0.5 * nu))
    
    elif (l == 1) and (m == 0):

        def G0(x, y):
            z_ = z(x, y)
            return jnp.where(z_ > 1 - 1e-8, -0.5 * y, (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * (-y))
        
        def G1(x, y):
            z_ = z(x, y)
            
            return jnp.where(z_ > 1 - 1e-8, 0.5 * x, (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * x)

        G = (G0, G1)

    elif (mu == 1) and (l % 2 == 0):
        
        G = (lambda x, y: x ** (l - 2) * z(x, y) ** 3, lambda x, y: 0)
    
    elif (mu == 1) and (l % 2 != 0):
        
        G = (lambda x, y: x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0)
    
    else:
        
        G = (
            lambda x, y: 0,
            lambda x, y: x ** (0.5 * (mu - 3))
            * y ** (0.5 * (nu - 1))
            * z(x, y) ** 3,
        )
        
    return G

    

def primitive(x,y,dx,dy,theta1,theta2, n):

    def func(theta):
        Gx, Gy = G(n)
        return Gx(x(theta), y(theta)) * dx(theta) + Gy(x(theta), y(theta)) * dy(theta)
    
    return gauss_quad(func, theta1, theta2, n=30)
    
    
def qT(xi1, xi2, n):
    """Compute the tT integral numerically from its integral definition."""
    
    x = lambda xi: jnp.cos(xi)
    y = lambda xi: jnp.sin(xi)
        
    dx = lambda xi: -jnp.sin(xi)
    dy = lambda xi: jnp.cos(xi)
    res = primitive(x, y, dx, dy, xi1, xi2, n)
    return res

def pT(phi1, phi2, b, xo, yo, ro, n):
    """Compute the pT integral numerically from its integral definition."""
    
    x = lambda phi: ro * jnp.cos(phi) + xo
    y = lambda phi: ro * b * jnp.sin(phi) + yo
    
    dx = lambda phi: -ro * jnp.sin(phi)
    dy = lambda phi: b * ro * jnp.cos(phi)
    res = primitive(x, y, dx, dy, phi1, phi2, n)
    return res


def sT(phi1, phi2, xi1, xi2, b, xo, yo, ro, n):
    """The solution vector for occultations, computed via Green's theorem."""
    return pT(phi1, phi2, b, xo, yo, ro, n) + qT(xi1, xi2, n)


####################
# MUCH FASTER SOLUTION
####################


def delta(lam: Array) -> Array:
    return lam[1] - lam[0]

def q_integral(l_max: int, lam: Array) -> Array:
    c = jnp.cos(lam)
    s = jnp.sin(lam)
    h = {
        (0, 0): delta(lam),
        (1, 0): delta(s),
        (0, 1): -delta(c),
        (1, 1): -delta(c**2)/2
    }

    def get(u: int, v: int) -> Array:
        if (u, v) in h:
            return h[(u, v)]
        if (u < 2) and (v >=2):
            term = c**(u+1)*s**(v-1)
            comp = (- delta(term) + (v-1)*get(u, v-2))
        else:
            term = c**(u-1)*s**(v+1)
            comp = (delta(term) + (u-1)*get(u-2, v))
        comp /= u + v
        h[(u, v)] = comp
        return comp

    U = []
    for l in range(l_max + 1):  # noqa
        for m in range(-l, l + 1):
            if l == 1 and m == 0:
                U.append((delta(lam)) / 3)
                continue
            mu = l - m
            nu = l + m
            if (mu % 2) == 0 and (nu % 2) == 0:
                u = mu // 2 + 2
                v = nu // 2
                U.append(get(u, v))
            else:
                U.append(0)

    return jnp.stack(U)


def solution_vector(l_max: int) -> Callable[[Array, Array, Array, Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),(),(),(),()->({n_max})")
    def impl(b:Array, theta: Array, xo:Array, yo:Array, ro:Array) -> Array:
        xo_rot, yo_rot = xo*jnp.cos(theta)-yo*jnp.sin(theta), xo*jnp.sin(theta)+yo*jnp.cos(theta)
        xis, phis = compute_bounds(b,xo_rot,yo_rot,ro)
        Q = q_integral(l_max, xis)
        P = []
        for l in range(l_max + 1):  # noqa
            for m in range(-l, l + 1):
                n = l**2 + l + m
                P.append(pT(phis[0], phis[1], b, xo_rot, yo_rot, ro, n=n))
        P = jnp.stack(P)
                
        return -(Q + P)

    return impl