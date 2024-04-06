
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import numpyro
from numpyro import distributions as dist
from numpyro import infer

from numpyro_ext import distributions as distx
from numpyro_ext import info, optim

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(2)
#jax.config.update('jax_disable_jit', True)

from jaxoplanet import light_curves, orbits
import arviz as az
import corner

from jax0planet import legacy_oblate_lightcurve, compute_bounds

np.random.seed(11)
period_true = np.random.uniform(5, 20)
t = np.linspace(0.9,1.1,1000)
yerr = 50*1e-6

true_params = {'period':period_true,
               't0':1.0,
                'radius':0.1,
                'bo':0.8,
                'u':jnp.array([0.3, 0.2]),
                'f':0.3,
                'theta':np.radians(35)
} 

print(true_params)

# Compute a limb-darkened light curve using starry
lc_true = oblate_lightcurve(true_params, t-true_params['t0'])

lc = lc_true + yerr*np.random.normal(size=len(t))

def model(t, yerr, y=None):
    # If we wanted to fit for all the parameters, we could use the following,
    # but we'll keep these fixed for simplicity.
    
    #log_duration = numpyro.sample("log_duration", dist.Uniform(jnp.log(0.08), jnp.log(0.2)))
    #b = numpyro.sample("b", dist.Uniform(0.0, 1.0))

    log_jitter = numpyro.sample("log_jitter", dist.Normal(jnp.log(yerr), 1.0))
    log_r = numpyro.sample("log_r", dist.Normal(jnp.log(0.1), 2.0))
    r = numpyro.deterministic("r", jnp.exp(log_r))
    u = numpyro.sample("u", distx.QuadLDParams())
    bo = numpyro.sample("bo", dist.Uniform(0.0,1.))
    f = numpyro.sample("f", dist.Uniform(0.,0.5))
    theta = numpyro.sample("theta", dist.Uniform(0.0,jnp.pi))
    params = {
        'period':period_true,
        't0': 1.0,
        "radius": r,
        'bo':bo,
        'u': u,
        'f':f, 
        'theta':theta      
    }

    numpyro.sample(
        "flux",
        dist.Normal(
            oblate_lightcurve(params, t-true_params['t0']), jnp.sqrt(yerr**2 + jnp.exp(2 * log_jitter))
        ),
        obs=y,
    )
    
init_params = {'period':period_true,
                't0':1.0,
                'log_jitter':jnp.log(yerr),
                'radius':true_params['radius'],
                'log_r':jnp.log(true_params['radius']),
                'u':jnp.array([0.3, 0.2]),
                'f':0.1,
                'bo':0.79,
                'theta':jnp.radians(0)
               
} 

sampler_wn = infer.MCMC(
    infer.NUTS(
        model,
        target_accept_prob=0.9,
        dense_mass=False,
        init_strategy=infer.init_to_median(),
        regularize_mass_matrix=False,
    ),
    num_warmup=1000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
sampler_wn.run(jax.random.PRNGKey(11), t, yerr, lc)

inf_data_wn = az.from_numpyro(sampler_wn)
#az.summary(inf_data_wn, var_names=["log_jitter", "log_r", "u1", "u2", "bo", "f", "theta"])
inf_data_wn.to_netcdf('oblate_planet_NUTS.h5')