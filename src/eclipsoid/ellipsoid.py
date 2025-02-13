import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

from jaxoplanet import orbits
from jaxoplanet.orbits.keplerian import Central, Body, System, OrbitalBody

from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.orbit import SurfaceBody, SurfaceSystem
from jaxoplanet import units
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg
from functools import partial

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Union
import jpu.numpy as jnpu


class EllipsoidalBody(Body):
    """ An ellipsoidal body
    Parameters:
    oblateness: flattening parameter along the rotational axis
    prolateness: flattening parameter along the z-axis by default at theta0
    """
    
    oblateness: Optional[Quantity] = units.field(
    default=0.0, units=ureg.dimensionless
    )
    
    prolateness: Optional[Quantity] = units.field(
    default=0.0, units=ureg.dimensionless
    )
    
    surface: Optional[Surface] = None
    
    def __check_init__(self) -> None:
        if ((self.oblateness is None) and (self.prolateness is None)):
            raise ValueError("Either oblateness or prolateness must be specified")
        

class EllipsoidalOrbitalBody(OrbitalBody):
    oblateness: Optional[Quantity] = units.field(units=ureg.dimensionless)
    prolateness: Optional[Quantity] = units.field(units=ureg.dimensionless)
    surface: Optional[Surface] = None
    
    def __init__(
        self,
        central: Central,
        body: Union[Body, EllipsoidalBody]
    ):
        super().__init__(central=central, body=body)
        if isinstance(body, EllipsoidalBody):
            self.oblateness = body.oblateness
            self.prolateness = body.prolateness
            self.surface = body.surface
        else:
            self.oblateness = 0.0
            self.prolateness = 0.0
            self.surface = Surface()
    

class EclipsoidSystem(System):
    """
    Rewrite of jaxoplanet's surface body to optionally handle ellipsoidal bodies
    """
    central_surface: Optional[Surface]
    _body_surface_stack: ObjectStack[Surface]
    
    def __init__(
        self,
        central: Optional[Central] = None,
        central_surface: Optional[Surface] = None,
        *,
        bodies: Iterable[
            tuple[Union[Body, EllipsoidalOrbitalBody, EllipsoidalBody], Optional[Surface]]
        ] = (),
    ):
        self.central = Central() if central is None else central

        if central_surface is None:
            central_surface = Surface()

        self.central_surface = central_surface

        orbital_bodies = []
        body_surfaces = []
        for body, surface in bodies:
            if isinstance(body, EllipsoidalOrbitalBody):
                orbital_bodies.append(body)
                body_surfaces.append(surface)
            else:
                orbital_bodies.append(EllipsoidalOrbitalBody(self.central, body))
                if surface is None:
                    body_surfaces.append(getattr(body, "surface", None))
                else:
                    body_surfaces.append(surface)

        self._body_stack = ObjectStack(*orbital_bodies)
        self._body_surface_stack = ObjectStack(*body_surfaces)

    @property
    def body_surfaces(self) -> tuple[Surface, ...]:
        return self._body_surface_stack.objects

    def add_body(
        self,
        body: Optional[Union[Body, SurfaceBody, EllipsoidalBody]] = None,
        surface: Optional[Surface] = None,
        **kwargs: Any,
    ) -> "EclipsoidSystem":
        if body is None:
            body = EllipsoidalBody(**kwargs)
        if surface is None:
            surface = getattr(body, "surface", None)
        bodies = list(zip(self.bodies, self.body_surfaces)) + [(body, surface)]
        
        return EclipsoidSystem(
            central=self.central,
            central_surface=self.central_surface,
            bodies=bodies,
        )
        
    @property
    def oblateness(self) -> Quantity:
        return self.body_vmap(lambda body: body.oblateness)()
    
    @property
    def prolateness(self) -> Quantity:
        return self.body_vmap(lambda body: body.prolateness)()

    def surface_vmap(
        self,
        func: Callable,
        in_axes: Union[int, None, Sequence[Any]] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._body_surface_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)
    
    
