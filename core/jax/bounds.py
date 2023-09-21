import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse


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

def intersection_points(b, xo, yo, ro):
    coeff = coeffs(b, xo, yo, ro)
    r=jnp.roots(coeff,)
    x_real = r.real[jnp.abs(r.imag)<1e-5]
    y_real = (-b**2*ro**2 + b**2*(x_real - xo)**2 - x_real**2 + yo**2 + 1)/(2*yo)
    return x_real, y_real

def compute_bounds(b, xo, yo, ro):
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape[0]==0:
        if jnp.hypot(xo,yo)<1:
            #occultor entirely inside star
            xi = jnp.array([2*jnp.pi,0])
            phi = jnp.array([0,2*jnp.pi])
            #force midpoint to be inside the star
            midpoint = 0
        else:
            #occultor entirely outside star
            xi = jnp.array([2*jnp.pi,0])
            phi = jnp.array([0,0])
            #force midpoint to be inside the star
            midpoint = 2
    elif x_real.shape[0]==2:
        
        xi = jnp.sort(jnp.arctan2(y_real,x_real))
        xi = jnp.where(
        #if
        xi[0]<jnp.arctan2(-yo,-xo)<xi[1], 
        #then
        jnp.array([xi[1],xi[0]]),
        #else
        jnp.array([xi[0]+2*jnp.pi,xi[1]])
                )
    
        phi = jnp.sort(jnp.arctan2(jnp.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*jnp.sign(jnp.arctan2(y_real-yo,x_real-xo)))
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
    else:
        raise NotImplementedError("jax0planet doesn't yet support 4 intersection points. Reduce r_occultor to << r_occulted")
    
    return xi, phi

def compute_bounds_under_planet(b, xo, yo, ro):
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape[0]==0:
        if jnp.hypot(xo,yo)<1:
            #occultor entirely inside star
            xi = jnp.array([0,0])
            phi = jnp.array([0,2*jnp.pi])
        else:
            #occultor entirely outside star
            xi = jnp.array([0,0])
            phi = jnp.array([0,0])
    elif x_real.shape[0]==2:
        xi = jnp.sort(jnp.arctan2(y_real,x_real))
        xi = jnp.where(
        #if xi contains vector pointing to planet center
        xi[0]<jnp.arctan2(-yo,-xo)<xi[1], 
        #then
        jnp.array([xi[1],xi[0]+2*jnp.pi]),
        #else
        jnp.array([xi[0],xi[1]]),
                )
    
        phi = jnp.sort(jnp.arctan2(jnp.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*jnp.sign(jnp.arctan2(y_real-yo,x_real-xo)))
        phi_inters=jnp.arctan2(-yo,-xo)
        phi = jnp.where(
            #if
            phi[0] < phi_inters < phi[1],
            #then
            jnp.array([phi[0],phi[1]]),
            #else
            jnp.array([phi[1],2*jnp.pi+phi[0]])
        )
    else:
        raise NotImplementedError("jax0planet doesn't yet support 4 intersection points. Reduce r_occultor to << r_occulted")
    
    return xi, phi




