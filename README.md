# eclipsoid
Transit models for ellipsoidal planets in Jax

Eclipsoid is an extension of the [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) package for computing transits, occultations, and rotational light curves of bodies in orbit around each other, such as an exoplanet orbiting a host star. In eclipsoid, we extend jaxoplanet to model these bodies when the orbiting body is oblate or ellipsoidal. This is expected to be the case for any planet with a nonzero rotation rate, and can be a significant effect for long period exoplanets. In our own Solar system, Saturn is about 10% oblate, (i.e 10% squashed at the poles compared to at its equator), and Jupiter about 7%. 

Refer to the jaxoplanet documentation for a primer on how to model light curves; our API largely follows jaxoplanet's. 