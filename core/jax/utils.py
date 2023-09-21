import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss


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

if __name__=="__main__":
    print(gauss_quad(poly,-1.,2.,10)-1/32*(36 - 8*np.sin(2) - 7*np.sin(4) + np.sin(8)))