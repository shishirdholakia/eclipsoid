import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

from jaxoplanet import orbits
from jaxoplanet.orbits.keplerian import Central, Body, System, OrbitalBody
from jaxoplanet import units
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg


from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union
import jpu.numpy as jnpu



class ProlateBody(Body):
    """ An Prolate body with a given Prolateness and obliquity """
    
    f: Optional[Quantity] = units.field(
    default=None, units=ureg.dimensionless
    )
    
    theta: Optional[Quantity] = units.field(
    default=0, units=ureg.radian
    )
    def __check_init__(self) -> None:
        if ((self.f is None) or (self.theta is None)):
            raise ValueError("Prolateness and obliquity must be both defined")

class ProlateOrbitalBody(OrbitalBody):
    
    f: Optional[Quantity] = units.field(
    default=None, units=ureg.dimensionless
    )
    
    theta: Optional[Quantity] = units.field(
    default=0, units=ureg.radian
    )
    
    def __init__(
        self,
        central: Central,
        body: ProlateBody
    ):
        super().__init__(central=central, body=body)
        self.f = body.f
        self.theta = body.theta
        
    @property
    def body_Prolateness(self) -> Quantity:
        return self.f
    
    @property
    def body_obliquity(self) -> Quantity:
        return self.theta
    
class ProlateSystem(System):
    
    def __init__(
        self,
        central: Optional[Central] = None,
        *,
        bodies: Iterable[Union[ProlateBody, ProlateOrbitalBody]] = (),
    ):
        self.central = Central() if central is None else central
        self._body_stack = ObjectStack(
            *(
                b if isinstance(b, ProlateOrbitalBody) else ProlateOrbitalBody(self.central, b)
                for b in bodies
            )
        )
        raise NotImplementedError("ProlateSystem not implemented yet")
    def add_body(
        self,
        body: Optional[ProlateBody] = None,
        central: Optional[Central] = None,
        **kwargs: Any,
    ) -> "ProlateSystem":
        body_: Optional[Union[ProlateBody, ProlateOrbitalBody]] = body
        if body_ is None:
            body_ = ProlateBody(**kwargs)
        if central is not None:
            body_ = OrbitalBody(central, body_)
        return ProlateSystem(central=self.central, bodies=self.bodies + (body_,))

    @property
    def body_Prolateness(self) -> Quantity:
        return self.body_vmap(lambda body: body.f)()
    
    @property
    def body_obliquity(self) -> Quantity:
        return self.body_vmap(lambda body: body.theta)()


class ProlateTransitOrbit(orbits.TransitOrbit):
    
    f: Optional[Quantity] = units.field(
    default=None, units=ureg.dimensionless
    )
    
    theta: Optional[Quantity] = units.field(
    default=0, units=ureg.radian
    )
    
    def __init__(
        self,
        *,
        period: Quantity,
        duration: Optional[Quantity] = None,
        speed: Optional[Quantity] = None,
        time_transit: Optional[Quantity] = None,
        impact_param: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
        f: Optional[Quantity] = None,
        theta: Optional[Quantity] = None,
    ):
        super().__init__(period=period, radius=radius, impact_param=impact_param, duration=duration, speed=speed, time_transit=time_transit)
        self.f = f
        self.theta = theta
        
    @property
    def body_Prolateness(self) -> Quantity:
        return self.f
    
    @property
    def body_obliquity(self) -> Quantity:
        return self.theta