import jax.numpy as jnp
import jax
from .utils import zero_safe_sqrt, zero_safe_arctan2, zero_safe_power, roots

def rotate_y(angle):
    """Rotation matrix around the y-axis by angle theta"""
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])

def rotate_z(angle):
    """Rotation matrix around the z-axis by angle obl"""
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

def rotate_x(angle):
    """Rotation matrix around the x-axis by angle inc"""
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])

def compute_projected_ellipse(r, f1, f2, theta, obl, inc):
    
    # Define the semi-axes of the ellipsoid
    a = r  # x-axis radius
    b = r * (1 - f1)  # y-axis radius
    c = r * (1 - f2)  # z-axis radius
    
    # Create the diagonal matrix of the ellipsoid
    ellipsoid_matrix = jnp.diag(jnp.array([a**2, b**2, c**2]))
    
    # Perform the rotations in sequence
    rotation_matrix = rotate_x(inc) @ rotate_z(obl) @ rotate_y(theta)
    
    # Transform the ellipsoid matrix
    transformed_matrix = rotation_matrix @ ellipsoid_matrix @ rotation_matrix.T
    
    # Projection to 2D (z''' axis is pointing towards the observer)
    # We only care about the x''' and y''' components in 2D projection
    projected_matrix = transformed_matrix[:2, :2]
    
    # Eigenvalues and eigenvectors for the projected ellipse
    eigenvalues, eigenvectors = jnp.linalg.eigh(projected_matrix)
    
    # The square roots of the eigenvalues give the lengths of the semi-axes
    semi_minor_axis = zero_safe_sqrt(eigenvalues[0])  # smallest eigenvalue
    semi_major_axis = zero_safe_sqrt(eigenvalues[1])  # largest eigenvalue
    
    # Angle of the semi-minor axis with respect to the x-axis
    angle_major_axis = jnp.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    
    return semi_major_axis, semi_minor_axis, angle_major_axis

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
    A, B, C, D and E all have a denominator of yo^2 which has been factored out
    for numerical stability at yo -> 0.
    """
    A = (b**4 - 2*b**2 + 1)/4
    B = (-b**4*xo + b**2*xo)
    C = (-b**4*ro**2 + 3*b**4*xo**2 + b**2*ro**2 - b**2*xo**2 + b**2*yo**2 + b**2 + yo**2 - 1)/2
    D = (b**4*ro**2*xo - b**4*xo**3 - b**2*xo*yo**2 - b**2*xo)
    E = (b**4*ro**4 - 2*b**4*ro**2*xo**2 + b**4*xo**4 - 2*b**2*ro**2*yo**2 - 2*b**2*ro**2 + 2*b**2*xo**2*yo**2 + 2*b**2*xo**2 + yo**4 - 2*yo**2 + 1)/4
    return jnp.array([A, B, C, D, E])

def compute_bounds(b, xo, yo, ro):
    
    coeff = jnp.array(coeffs(b, xo, yo, ro),dtype=complex)
    x_roots=roots(coeff,strip_zeros=False)
    #plug into the ellipse to avoid the +/- ambiguity with sqrt(1-x_roots**2)
    #but how to fix the precision issue for yo->0?
    y_roots = jnp.where(jnp.abs(yo)<1e-6,zero_safe_sqrt(1-x_roots**2),(-b**2*ro**2 + b**2*(x_roots - xo)**2 - x_roots**2 + yo**2 + 1)/(2*yo))
    reals = jnp.sum(jnp.abs(x_roots.imag)<1e-10)
    
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

def compute_bounds_near_zero(b, xo, yo, ro):

    ###
    # IF yo is close to 0, use a quadratic solver to patch the singularities in the quartic solver at yo=0
    ###
    A0 = 1 - (b**2)
    B0 = - (2 * b**2) * xo
    C0 = b**2*ro**2 - b**2 * xo**2 + yo - 1
    discriminant = B0**2 - 4 * A0 * C0
    quad_roots = -jnp.array([(-B0 + zero_safe_sqrt(discriminant)) / (2 * A0),
                            (-B0 - zero_safe_sqrt(discriminant)) / (2 * A0)])

    #construct a size-4 array containing the real root if it exists, or pad the array with 1j
    x_quad_root = jnp.tile(jnp.where(jnp.abs(quad_roots)<1.0, quad_roots, jnp.array([2j]).repeat(2)), 2)
    y_quad_root = jnp.where(jnp.abs(x_quad_root)<1.0, jnp.array([zero_safe_sqrt(1 - quad_roots**2), -zero_safe_sqrt(1 - quad_roots**2)]).flatten(),jnp.array([2j]).repeat(4))
    
    ###
    # Use the quartic solver to find the roots of the quartic equation
    ###
    coeff = jnp.array(coeffs(b, xo, yo, ro),dtype=complex)
    x_roots=jnp.where(jnp.abs(yo)>1e-5, roots(coeff,strip_zeros=False), x_quad_root)
    #plug into the ellipse to avoid the +/- ambiguity with sqrt(1-x_roots**2)
    #but how to fix the precision issue for yo->0?
    y_roots = jnp.where(jnp.abs(yo)>1e-5, (-b**2*ro**2 + b**2*(x_roots - xo)**2 - x_roots**2 + yo**2 + 1)/(2*yo), y_quad_root)
    #y_roots = jnp.where(jnp.abs(yo)<1e-6,zero_safe_sqrt(1-x_roots**2),(-b**2*ro**2 + b**2*(x_roots - xo)**2 - x_roots**2 + yo**2 + 1)/(2*yo))
    reals = jnp.sum(jnp.abs(x_roots.imag)<1e-10)
    
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

