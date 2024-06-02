import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

import helper_funcs


# This will create the nominal paths for the vehicle to track
# Paths will be created by rolling out the vehicle simulator and 
jax.config.update('jax_platform_name', 'cpu')

def rollout_path(dyn_model, control_fn, 
                 initial_veh_state, rollout_horizon_m, ds,
                 initial_world_pos = jnp.zeros(3),
                 state_n = 3,
                 control_m = 4):
    #Generates trajectories by rolling out the vehicle dynamics subject to a given control function
    #Returns path trajectory information as well as interpolated functions for path heading and curviture values

    n_steps = int(rollout_horizon_m / ds)

    veh_space_traj = np.zeros((n_steps + 1, state_n))
    controls_traj = np.zeros((n_steps, control_m))

    #Set Initial State
    veh_space_traj[0] = initial_veh_state

    s_vals = jnp.arange(n_steps + 1) * ds
    #Iterate State Trajectory Forward
    for i in range(n_steps):
        controls_traj[i] = control_fn(veh_space_traj[i])
        veh_state_derivs = dyn_model(veh_space_traj[i], controls_traj[i])

        veh_vel_magnitude = np.linalg.norm(veh_space_traj[0:2])
        time_step = ds / veh_vel_magnitude
    
        veh_space_traj[i + 1] = veh_state_derivs * time_step + veh_space_traj[i]
    
    world_xy, world_vels = helper_funcs.s_spaced_states_to_world_positions(initial_world_pos, veh_space_traj, ds)
    
    #Discrete heading and path for trajectory
    psi_traj = jnp.arctan2(-1.0 * world_vels[:, 0], world_vels[:, 1]) #Path Heading: -x_dot / y_dot
    kappa_traj = jnp.gradient(psi_traj, ds)

    #Create Interpolation Functions for psi and kappa
    def interp_psi(s_coord):
        return jnp.interp(s_coord, s_vals, psi_traj, left="extrapolate", right="extrapolate")

    def interp_kappa(s_coord):
        return jnp.interp(s_coord, s_vals, kappa_traj, left="extrapolate", right="extrapolate")


    return veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa

def get_path_space_dyn_func(dyn_func, path_information, ds):
    # Given a dynamics model and path space information, creates a path-space dynamics model for the funtion
    
    veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_information
    @jax.jit
    def veh_and_path_dyn(veh_and_path_state, control_inputs, s_coord):
        v_x, v_y, r, s, e_lat, delta_psi = veh_and_path_state
        
        veh_deriv = dyn_func(jnp.array([v_x, v_y, r]), control_inputs)
        #---------Calculation of state derivatives in Path-Fixed Coordinates-----------

        s_dot = (v_x * jnp.cos(delta_psi) - v_y * jnp.sin(delta_psi)) / (1 - interp_kappa(s_coord) * e_lat)
        e_dot = (v_x * jnp.sin(delta_psi) + v_y * jnp.cos(delta_psi))
        delta_psi_dot = r - interp_kappa(s_coord) * s_dot

        #S_space deriv
        full_deriv = (1 / s_dot) * (jnp.concatenate((veh_deriv, jnp.array([s_dot, e_dot, delta_psi_dot]))))

        #Euler Integrate to Next State
        next_veh_path_state = full_deriv * ds + veh_and_path_state
        return next_veh_path_state

    return veh_and_path_dyn


