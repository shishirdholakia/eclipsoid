import numpy as np
import jax.numpy as jnp
import jax
from numpy.polynomial.legendre import leggauss


import numpy as np

from jax import jit
from jax import lax
from jax._src import dtypes
from jax._src import core
from jax._src.numpy.lax_numpy import (
    arange, argmin, array, asarray, atleast_1d, concatenate, convolve,
    diag, dot, finfo, full, ones, outer, roll, trim_zeros,
    trim_zeros_tol, vander, zeros)
from jax._src.numpy.ufuncs import maximum, true_divide, sqrt
from jax._src.numpy.reductions import all
from jax._src.numpy import linalg
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes, promote_dtypes_inexact, _where)
from jax._src.typing import Array, ArrayLike

from jax import pure_callback
from typing import Tuple


import jax.lax.linalg as lax_linalg
from jax import custom_jvp

from jax import lax
from jax.numpy.linalg import solve
from functools import partial

# -----------------------------------------------------------------------------
# Functions related to a generalized solution to a polynomial's roots that
# supports higher order derivatives
# -----------------------------------------------------------------------------

#FROM https://github.com/google/jax/issues/2748#issuecomment-1179511268

@custom_jvp
def eig_jvp(a):
    w, vl, vr = lax_linalg.eig(a)
    return w, vr


@eig_jvp.defjvp
def eig_jvp_rule(primals, tangents):
    a, = primals
    da, = tangents

    w, v = eig_jvp(a)

    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
    Fmat = (jnp.reciprocal(eye + w[..., jnp.newaxis, :] - w[..., jnp.newaxis])
            - eye)
    dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                  precision=lax.Precision.HIGHEST)
    vinv_da_v = dot(solve(v, da), v)
    du = dot(v, jnp.multiply(Fmat, vinv_da_v))
    corrections = (jnp.conj(v) * du).sum(-2, keepdims=True)
    dv = du - v * corrections
    dw = jnp.diagonal(vinv_da_v, axis1=-2, axis2=-1)
    return (w, v), (dw, dv)


# Taken verbatim from 
# https://github.com/facebookresearch/fmmax/blob/8c7491a23e902644614c1761b6fbefa472786567/src/fmmax/utils.py

EPS_EIG = 1e-6

def matrix_adjoint(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the adjoint for a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))


@jax.custom_vjp
def eig(matrix: jnp.ndarray, eps: float = EPS_EIG) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme used in [2021
    Colburn]. The method effectively applies a Lorentzian broadening to a term
    containing the inverse difference of eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392
    [2021 Coluburn] https://www.nature.com/articles/s42005-021-00568-6

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps
    return eig_jvp(matrix)




def _eig_fwd(
    matrix: jnp.ndarray,
    eps: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = eig_jvp(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps = res
    grad_eigenvalues, grad_eigenvectors = grads

    # Compute the F-matrix, from equation 5 of [2021 Colburn]. This applies a
    # Lorentzian broadening to the matrix `f = 1 / (eigenvalues[i] - eigenvalues[j])`.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    f_broadened = (eigenvalues_i - eigenvalues_j) / (
        (eigenvalues_i - eigenvalues_j) ** 2 + eps
    )

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_H = matrix_adjoint(eigenvectors)
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_H @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_H @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_H @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_H
    grad_matrix = jnp.linalg.solve(eigenvectors_H, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps`.
    return grad_matrix, None


eig.defvjp(_eig_fwd, _eig_bwd)


# TAKEN FROM JAX SOURCE CODE FOR jnp.roots
def _roots_no_zeros(p: Array) -> Array:
  # build companion matrix and find its eigenvalues (the roots)
  if p.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p.dtype))
  A = diag(ones((p.size - 2,), p.dtype), -1)
  A = A.at[0, :].set(-p[1:] / p[0])
  eigvals, eigvecs = eig(A)
  return eigvals


def _roots_with_zeros(p: Array, num_leading_zeros: int) -> Array:
  # Avoid lapack errors when p is all zero
  p = _where(len(p) == num_leading_zeros, 1.0, p)
  # Roll any leading zeros to the end & compute the roots
  roots = _roots_no_zeros(roll(p, -num_leading_zeros))
  # Sort zero roots to the end.
  roots = lax.sort_key_val(roots == 0, roots)[1]
  # Set roots associated with num_leading_zeros to NaN
  return _where(arange(roots.size) < roots.size - num_leading_zeros, roots, complex(1.j, 1.j))



def roots(p: ArrayLike, *, strip_zeros: bool = True) -> Array:
  check_arraylike("roots", p)
  p_arr = atleast_1d(*promote_dtypes_inexact(p))
  if p_arr.ndim != 1:
    raise ValueError("Input must be a rank-1 array.")
  if p_arr.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p_arr.dtype))
  num_leading_zeros = _where(all(p_arr == 0), len(p_arr), argmin(p_arr == 0))

  if strip_zeros:
    num_leading_zeros = core.concrete_or_error(int, num_leading_zeros,
      "The error occurred in the jnp.roots() function. To use this within a "
      "JIT-compiled context, pass strip_zeros=False, but be aware that leading zeros "
      "will be result in some returned roots being set to NaN.")
    return _roots_no_zeros(p_arr[num_leading_zeros:])
  else:
    return _roots_with_zeros(p_arr, num_leading_zeros)



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

if __name__=="__main__":
    print(gauss_quad(poly,-1.,2.,10)-1/32*(36 - 8*np.sin(2) - 7*np.sin(4) + np.sin(8)))