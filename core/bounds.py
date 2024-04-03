import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse


def coeffs_zhu(b, xo, yo, a):
    A = a**2-b**2
    B = 4*b**2*xo-4*1.j*a**2*yo
    C = -2*(a**2+b**2-2*a**2*b**2+2*b**2*xo**2+2*a**2*yo**2)
    D = 4*b**2*xo+4*1.j*a**2*yo
    E = a**2-b**2
    return np.array([A, B, C, D, E])

def coeffs(b, xo, yo, ro):
    """
    Polynomial coefficients A, B, C, D and E coded up as a python function.
    """
    A = (b**4 - 2*b**2 + 1)/(4*yo**2)
    B = (-b**4*xo + b**2*xo)/yo**2
    C = (-b**4*ro**2 + 3*b**4*xo**2 + b**2*ro**2 - b**2*xo**2 + b**2*yo**2 + b**2 + yo**2 - 1)/(2*yo**2)
    D = (b**4*ro**2*xo - b**4*xo**3 - b**2*xo*yo**2 - b**2*xo)/yo**2
    E = (b**4*ro**4 - 2*b**4*ro**2*xo**2 + b**4*xo**4 - 2*b**2*ro**2*yo**2 - 2*b**2*ro**2 + 2*b**2*xo**2*yo**2 + 2*b**2*xo**2 + yo**4 - 2*yo**2 + 1)/(4*yo**2)
    return np.array([A, B, C, D, E])

def intersection_points(b, xo, yo, ro):
    coeff = coeffs(b, xo, yo, ro)
    r=np.roots(coeff)
    x_real = r.real[np.abs(r.imag)<1e-5]
    y_real = (-b**2*ro**2 + b**2*(x_real - xo)**2 - x_real**2 + yo**2 + 1)/(2*yo)
    return x_real, y_real

def compute_bounds(b, xo, yo, ro):
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape[0]==0:
        if np.hypot(xo,yo)<1:
            #occultor entirely inside star
            xi = np.array([2*np.pi,0])
            phi = np.array([0,2*np.pi])
            #force midpoint to be inside the star
            midpoint = 0
        else:
            #occultor entirely outside star
            xi = np.array([2*np.pi,0])
            phi = np.array([0,0])
            #force midpoint to be inside the star
            midpoint = 2
    elif x_real.shape[0]==2:
        
        xi = np.sort(np.arctan2(y_real,x_real))
        xi = np.where(
        #if
        xi[0]<np.arctan2(-yo,-xo)<xi[1], 
        #then
        np.array([xi[1],xi[0]]),
        #else
        np.array([xi[0]+2*np.pi,xi[1]])
                )
    
        phi = np.sort(np.arctan2(np.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*np.sign(np.arctan2(y_real-yo,x_real-xo)))
        #ALGORITHM TO FIND CORRECT SEGMENT FOR INTEGRATION
        #FIND MIDDLE POINT ON ELLIPSE PARAMETRIZED BY PHI
        #IF THAT POINT IS IN CIRCLE, RIGHT BOUNDS
        #IF NOT, SWITCH
        midpoint = np.hypot(ro*np.cos(np.mean(phi)) + xo, ro*b*np.sin(np.mean(phi))+yo)
        phi = np.where(
            #if
            midpoint<1.0,
            #then
            np.array([phi[0],phi[1]]),
            #else
            np.array([phi[1],2*np.pi+phi[0]])
        )
    else:
        raise NotImplementedError("jax0planet doesn't yet support 4 intersection points. Reduce r_occultor to << r_occulted")
    
    return xi, phi

def compute_bounds_under_planet(b, xo, yo, ro):
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape[0]==0:
        if np.hypot(xo,yo)<1:
            #occultor entirely inside star
            xi = np.array([0,0])
            phi = np.array([0,2*np.pi])
        else:
            #occultor entirely outside star
            xi = np.array([0,0])
            phi = np.array([0,0])
    elif x_real.shape[0]==2:
        xi = np.sort(np.arctan2(y_real,x_real))
        xi = np.where(
        #if xi contains vector pointing to planet center
        xi[0]<np.arctan2(-yo,-xo)<xi[1], 
        #then
        np.array([xi[1],xi[0]+2*np.pi]),
        #else
        np.array([xi[0],xi[1]]),
                )
    
        phi = np.sort(np.arctan2(np.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*np.sign(np.arctan2(y_real-yo,x_real-xo)))
        phi_inters=np.arctan2(-yo,-xo)
        phi = np.where(
            #if
            phi[0] < phi_inters < phi[1],
            #then
            np.array([phi[0],phi[1]]),
            #else
            np.array([phi[1],2*np.pi+phi[0]])
        )
    else:
        raise NotImplementedError("jax0planet doesn't yet support 4 intersection points. Reduce r_occultor to << r_occulted")
    
    return xi, phi





##############################################################
# PLOTTING FUNCTIONS
##############################################################

def draw_oblate_under_planet(b, xo, yo, ro):
    # Set up the figure
    #theta in degrees
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(min(-1.01, xo - ro - 0.01), max(1.01, xo + ro + 0.01));
    ax.set_ylim(min(-1.01, yo - ro - 0.01), max(1.01, yo + ro + 0.01));
    ax.set_aspect(1);
    ax.axis('off');

    # Draw the star
    occulted = Circle((0, 0), 1, fill=False, color='k')
    occulted_fill = Circle((0, 0), 1, fill=True, color='k',alpha=0.03)
    ax.add_artist(occulted)
    ax.add_artist(occulted_fill)
    
    # Draw the planet, multiply semi major and semi minor axes by 2 to get major and minor axes
    occultor = Ellipse((xo, yo), ro*2,ro*b*2, fill=False, color='r')
    occultor_fill = Ellipse((xo, yo), ro*2,ro*b*2, fill=True, color='r',alpha=0.03)
    ax.add_artist(occultor_fill)
    ax.add_artist(occultor)

    ax.plot(0, 0, 'ko')
    ax.plot(xo, yo, 'ro')
    
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape == (0,): #if there are no intersections
        xi = (0,0) #we want xi to be from 0 to 2 pi
    else: 
        xi = np.sort(np.arctan2(y_real,x_real))

    ax.plot(x_real,y_real, 'ko')

    #if xi0 to xi1 contains the unit vector alpha
    #we DO want to perform that integral, instead switch to xi1 to xi0 
    if (xi[0]<np.arctan2(-yo,-xo)<xi[1]):
        xi_grid = np.linspace(xi[1],xi[0]+2*np.pi,1000)
    else:
        xi_grid = np.linspace(xi[0],xi[1],1000)
        
    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(xi_grid):
        x[i] = np.cos(v)
        y[i] = np.sin(v)
        
    for i in np.arange(0,len(x),len(x)//10)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="k"),
            size=20,
        )
    #bold the Q integral region
    ax.plot(x, y, color='k', lw=2,zorder=-1); 

    ax.plot([0,np.cos(xi[0])],[0,np.sin(xi[0])], 'k-', alpha=0.3)
    ax.plot([0,np.cos(xi[1])],[0,np.sin(xi[1])], 'k-', alpha=0.3)

    #add the circle bounding the planet to help parametrize the angle phi
    anomaly = Circle((xo, yo), ro, fill=False, color='r', alpha=0.3)
    ax.add_artist(anomaly)

    #horizontal line along the major axis
    ax.plot([xo-ro,xo+ro],[yo,yo],'r--',alpha=0.3)

    #arctan of y *on the circle circumscribing the occcultor ellipse* to x on the ellipse (drops straight down)
    if x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)<1: #if there are no intersections and the planet is in star
        phi = (0,0) #phi from 0 to 2 pi (2 pi will be added later)
    elif x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)>=1: #if no intersections and planet is outside star
        phi = (0,2*np.pi) #phi from 2pi to 2pi (we don't want to integrate boundary of planet)
    else:
        phi = np.sort(np.arctan2(np.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*np.sign(np.arctan2(y_real-yo,x_real-xo)))

    #plot the phi angle (parametrized like eccentric anomaly)
    ax.plot([xo,xo+ro*np.cos(phi[0])],[yo,yo+ro*np.sin(phi[0])], 'r-', alpha=0.3)
    ax.plot([xo,xo+ro*np.cos(phi[1])],[yo,yo+ro*np.sin(phi[1])], 'r-', alpha=0.3)

    #plot the line down to the major axis
    ax.plot([xo+ro*np.cos(phi[0]), xo+ro*np.cos(phi[0])],[yo+ro*np.sin(phi[0]), yo], 'r--', alpha=0.3)
    ax.plot([xo+ro*np.cos(phi[1]), xo+ro*np.cos(phi[1])],[yo+ro*np.sin(phi[1]), yo], 'r--', alpha=0.3)

    phi_inters=np.arctan2(-yo,-xo)


    if phi[0] < phi_inters < phi[1]:
        phi_grid = np.linspace(phi[0],phi[1],1000)
    else:
        #reverse the order of integration so it is always performed counterclockwise
        phi_grid = np.linspace(phi[1],2*np.pi+phi[0],1000)

    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(phi_grid):
        x[i] = ro*np.cos(v) + xo
        y[i] = ro*b*np.sin(v) + yo
        
    #plot arrows to show the direction of the P integral
    for i in np.arange(0,len(x),len(x)//5)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="r"),
            size=20,
        )
    #bold the P integral region
    ax.plot(x, y, color='r', lw=2,zorder=-1);
    return fig, ax

def draw_oblate(b, xo, yo, ro):
    # Set up the figure
    #theta in degrees
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(min(-1.01, xo - ro - 0.01), max(1.01, xo + ro + 0.01));
    ax.set_ylim(min(-1.01, yo - ro - 0.01), max(1.01, yo + ro + 0.01));
    ax.set_aspect(1);
    ax.axis('off');

    # Draw the star
    occulted = Circle((0, 0), 1, fill=False, color='k')
    occulted_fill = Circle((0, 0), 1, fill=True, color='k',alpha=0.03)
    ax.add_artist(occulted)
    ax.add_artist(occulted_fill)
    
    # Draw the planet, multiply semi major and semi minor axes by 2 to get major and minor axes
    occultor = Ellipse((xo, yo), ro*2,ro*b*2, fill=False, color='r')
    occultor_fill = Ellipse((xo, yo), ro*2,ro*b*2, fill=True, color='r',alpha=0.03)
    ax.add_artist(occultor_fill)
    ax.add_artist(occultor)

    ax.plot(0, 0, 'ko')
    ax.plot(xo, yo, 'ro')
    
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape == (0,): #if there are no intersections
        xi = (0,0) #we want xi to be from 0 to 2 pi
    else: 
        xi = np.sort(np.arctan2(y_real,x_real))

    ax.plot(x_real,y_real, 'ko')

    #if xi0 to xi1 contains the unit vector alpha
    #we DONT want to perform that integral, instead switch to xi1 to xi0 
    if (xi[0]<np.arctan2(-yo,-xo)<xi[1]):
        xi_grid = np.linspace(xi[1],xi[0],1000)
    else:
        xi_grid = np.linspace(xi[0]+2*np.pi,xi[1],1000)
        
    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(xi_grid):
        x[i] = np.cos(v)
        y[i] = np.sin(v)
        
    for i in np.arange(0,len(x),len(x)//10)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="k"),
            size=20,
        )
    #bold the Q integral region
    ax.plot(x, y, color='k', lw=2,zorder=-1); 

    ax.plot([0,np.cos(xi[0])],[0,np.sin(xi[0])], 'k-', alpha=0.3)
    ax.plot([0,np.cos(xi[1])],[0,np.sin(xi[1])], 'k-', alpha=0.3)

    #add the circle bounding the planet to help parametrize the angle phi
    anomaly = Circle((xo, yo), ro, fill=False, color='r', alpha=0.3)
    ax.add_artist(anomaly)

    #horizontal line along the major axis
    ax.plot([xo-ro,xo+ro],[yo,yo],'r--',alpha=0.3)

    #arctan of y *on the circle circumscribing the occcultor ellipse* to x on the ellipse (drops straight down)
    if x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)<1: #if there are no intersections and the planet is in star
        phi = (0,0) #phi from 0 to 2 pi (2 pi will be added later)
    elif x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)>=1: #if no intersections and planet is outside star
        phi = (0,2*np.pi) #phi from 2pi to 2pi (we don't want to integrate boundary of planet)
    else:
        phi = np.sort(np.arctan2(np.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*np.sign(np.arctan2(y_real-yo,x_real-xo)))

    #plot the phi angle (parametrized like eccentric anomaly)
    ax.plot([xo,xo+ro*np.cos(phi[0])],[yo,yo+ro*np.sin(phi[0])], 'r-', alpha=0.3)
    ax.plot([xo,xo+ro*np.cos(phi[1])],[yo,yo+ro*np.sin(phi[1])], 'r-', alpha=0.3)

    #plot the line down to the major axis
    ax.plot([xo+ro*np.cos(phi[0]), xo+ro*np.cos(phi[0])],[yo+ro*np.sin(phi[0]), yo], 'r--', alpha=0.3)
    ax.plot([xo+ro*np.cos(phi[1]), xo+ro*np.cos(phi[1])],[yo+ro*np.sin(phi[1]), yo], 'r--', alpha=0.3)

    phi_inters=np.arctan2(-yo,-xo)


    if phi[0] < phi_inters < phi[1]:
        phi_grid = np.linspace(phi[0],phi[1],1000)
    else:
        #reverse the order of integration so it is always performed counterclockwise
        phi_grid = np.linspace(phi[1],2*np.pi+phi[0],1000)

    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(phi_grid):
        x[i] = ro*np.cos(v) + xo
        y[i] = ro*b*np.sin(v) + yo
        
    #plot arrows to show the direction of the P integral
    for i in np.arange(0,len(x),len(x)//5)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="r"),
            size=20,
        )
    #bold the P integral region
    ax.plot(x, y, color='r', lw=2,zorder=-1);
    return fig, ax

if __name__ == "__main__":
    # Draw the star
    fig, ax = draw_oblate(0.7, 0.8, 0.8, 0.4)
    fig.savefig("oblate_star.png", dpi=300)