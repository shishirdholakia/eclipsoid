import jax.numpy as jnp
import jax
from .utils import zero_safe_sqrt, zero_safe_arctan2, zero_safe_power


def coeffs_zhu(b, xo, yo, a):
    A = a**2-b**2
    B = 4*b**2*xo-4*1.j*a**2*yo
    C = -2*(a**2+b**2-2*a**2*b**2+2*b**2*xo**2+2*a**2*yo**2)
    D = 4*b**2*xo+4*1.j*a**2*yo
    E = a**2-b**2
    return jnp.array([A, B, C, D, E])

def coeffs(b, xo, yo, ro):
    """
    Polynomial coefficients A, B, C, D and E coded up as a python function.
    """
    A = (b**4 - 2*b**2 + 1)/(4*yo**2)
    B = (-b**4*xo + b**2*xo)/yo**2
    C = (-b**4*ro**2 + 3*b**4*xo**2 + b**2*ro**2 - b**2*xo**2 + b**2*yo**2 + b**2 + yo**2 - 1)/(2*yo**2)
    D = (b**4*ro**2*xo - b**4*xo**3 - b**2*xo*yo**2 - b**2*xo)/yo**2
    E = (b**4*ro**4 - 2*b**4*ro**2*xo**2 + b**4*xo**4 - 2*b**2*ro**2*yo**2 - 2*b**2*ro**2 + 2*b**2*xo**2*yo**2 + 2*b**2*xo**2 + yo**4 - 2*yo**2 + 1)/(4*yo**2)
    return jnp.array([A, B, C, D, E])

def compute_bounds(b, xo, yo, ro):
    
    coeff = coeffs(b, xo, yo, ro)
    x_roots=jnp.roots(coeff,strip_zeros=False)
    y_roots = (-b**2*ro**2 + b**2*(x_roots - xo)**2 - x_roots**2 + yo**2 + 1)/(2*yo)
    reals = jnp.sum(jnp.abs(x_roots.imag)<1e-5)
    
    def no_ints(x_roots, y_roots, b, xo, yo, ro):
        in_star = jnp.hypot(xo,yo)<1
        xi = jnp.array([2*jnp.pi,0])
        phi = jnp.where(in_star,jnp.array([0,2*jnp.pi]),jnp.array([0,0]))
        return xi, phi
    def two_ints(x_roots, y_roots, b, xo, yo, ro):
        #sort by value of complex part
        real_sorted_args = jnp.argsort(jnp.abs(x_roots.imag))
        
        #remove last two values, leaving only real roots
        x_real = x_roots[real_sorted_args][:-2].real #size 2 array
        y_real = y_roots[real_sorted_args][:-2].real #size 2 array
        
        xi = jnp.sort(zero_safe_arctan2(y_real,x_real))
        
        between = jnp.logical_and(zero_safe_arctan2(-yo,-xo)>xi[0], zero_safe_arctan2(-yo,-xo)<xi[1])
        xi = jnp.where(
        #if
        between, 
        #then
        jnp.array([xi[1],xi[0]]),
        #else
        jnp.array([xi[0]+2*jnp.pi,xi[1]])
                )
        
        phi = jnp.sort(zero_safe_arctan2(zero_safe_sqrt(jnp.abs(ro**2-(x_real-xo)**2)),x_real-xo)*jnp.sign(zero_safe_arctan2(y_real-yo,x_real-xo)))
        #ALGORITHM TO FIND CORRECT SEGMENT FOR INTEGRATION
        #FIND MIDDLE POINT ON ELLIPSE PARAMETRIZED BY PHI
        #IF THAT POINT IS IN CIRCLE, RIGHT BOUNDS
        #IF NOT, SWITCH
        midpoint = jnp.hypot(ro*jnp.cos(jnp.mean(phi)) + xo, ro*b*jnp.sin(jnp.mean(phi))+yo)
        phi = jnp.where(
            #if
            midpoint<1.0,
            #then
            jnp.array([phi[0],phi[1]]),
            #else
            jnp.array([phi[1],2*jnp.pi+phi[0]])
        )
        return xi, phi
    def four_ints(x_roots, y_roots, b, xo, yo, ro):
        return jnp.array([jnp.inf, jnp.inf]), jnp.array([jnp.inf, jnp.inf])

    
    xi, phi = jax.lax.cond(reals==0,
                           no_ints,
                           lambda x_roots, y_roots, b, xo, yo, ro: 
                               jax.lax.cond(reals==2, two_ints, four_ints, x_roots, y_roots, b, xo, yo, ro)
                           , x_roots, y_roots, b, xo, yo, ro)
    return xi, phi

