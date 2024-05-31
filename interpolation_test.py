import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu') #NOTE: Interpolation fails when run under JAX Metal

# Testing of 1D Interpolation in JAX.
# Will use this to interpolate path curviture values
x_vals = jnp.arange(10)
y_vals = 2 * jnp.arange(10)

#Let's try making a f(x) that returns our interpolated values for a given x

def interp_func(x):
    return jnp.interp(x, x_vals, y_vals, left="extrapolate", right="extrapolate")

x = 1.5
intp_val = interp_func(x)
print(f"interpreted: {intp_val}")