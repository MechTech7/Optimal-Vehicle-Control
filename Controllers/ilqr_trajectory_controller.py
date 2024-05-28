import numpy as np
import jax
import jax.numpy as jnp

#Full State: [ux, uy, r, s, e, delta_phi]
def traj_dyn_func(veh_dyn_func, trajectory_info):
    phi_ref, phi_ref_deriv, kappa_ref, s_dot_ref = trajectory_info

    #TODO: How do I know which kappa_ref I'm comparing to at any timestep?
    
    def dyn_func(full_state):
        ux, uy, r, s, e, delta_phi = full_state
        vehicle_state = full_state[:3]

        beta = jnp.atan2(uy, ux)
        v = jnp.sqrt(ux ** 2 + uy ** 2)
        s_dot = (v * jnp.cos(delta_phi)) #***

    return dyn_func
def ilqr_function(dyn_func, init_state, ):
    pass