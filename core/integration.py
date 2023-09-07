import jax.numpy as jnp


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
    r=jnp.roots(coeff)
    x_real = r.real[jnp.abs(r.imag)<1e-5]
    y_real = (-b**2*ro**2 + b**2*(x_real - xo)**2 - x_real**2 + yo**2 + 1)/(2*yo)
    return x_real, y_real

def compute_bounds(b, xo, yo, ro):
    x_real, y_real = intersection_points(b, xo, yo, ro)
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
    phi_inters=jnp.arctan2(-yo,-xo)
    phi = jnp.where(
        #if
        phi[0] < phi_inters < phi[1],
        #then
        jnp.array([phi[0],phi[1]]),
        #else
        jnp.array([phi[1],2*jnp.pi+phi[0]])
    )
    
    return xi, phi

