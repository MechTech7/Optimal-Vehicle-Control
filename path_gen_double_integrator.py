import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

import helper_funcs

jax.config.update('jax_platform_name', 'cpu')




def rollout_path(s_space_dyn_func, control_fn, 
                 initial_state, rollout_horizon_m, ds,
                 initial_world_pos = jnp.zeros(3),
                 state_n = 3,
                 control_m = 3):
    #Generates trajectories by rolling out the vehicle dynamics subject to a given control function
    #Returns path trajectory information as well as interpolated functions for path heading and curviture values

    # NOTE: s_space_dyn_func is the time derivative function converted into an s derivative function by 
    #       multiplying by 1 / s_dot and integrated using either euler or runge-kutta integration

    n_steps = int(rollout_horizon_m / ds)

    ref_state_traj = np.zeros((n_steps + 1, state_n))
    ref_ctrl_traj = np.zeros((n_steps, control_m))

    #Set Initial State
    ref_state_traj[0] = initial_state

    s_vals = jnp.arange(n_steps + 1) * ds

    #Iterate State Trajectory Forward
    for i in range(n_steps):
        ref_ctrl_traj[i] = control_fn(ref_state_traj[i], s_vals[i])
        ref_state_traj[i + 1] = s_space_dyn_func(ref_state_traj[i], ref_ctrl_traj[i])
    
    world_xy, world_vels = helper_funcs.s_spaced_states_to_world_positions(initial_world_pos, ref_state_traj, ds)
    
    #Discrete heading and path for trajectory
    psi_traj = jnp.arctan2(-1.0 * world_vels[:, 0], world_vels[:, 1]) #Path Heading: -x_dot / y_dot
    kappa_traj = jnp.gradient(psi_traj, ds)

    #Create Interpolation Functions for psi and kappa
    def interp_psi(s_coord):
        return jnp.interp(s_coord, s_vals, psi_traj, left="extrapolate", right="extrapolate")

    def interp_kappa(s_coord):
        return jnp.interp(s_coord, s_vals, kappa_traj, left="extrapolate", right="extrapolate")


    return ref_state_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa




def get_path_space_dyn_func(dyn_func, path_information, ds):
    # Given a dynamics model and path space information, creates a path-space dynamics model for the funtion
    
    ref_state_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_information
    
    def path_dyn(veh_and_path_state, control_inputs, s_coord):
        v_x, v_y, r, s, e_lat, delta_psi = veh_and_path_state
        
        veh_deriv = dyn_func(jnp.array([v_x, v_y, r]), control_inputs)
        #---------Calculation of state derivatives in Path-Fixed Coordinates-----------
        kap = interp_kappa(s_coord)
        s_dot = (v_x * jnp.cos(delta_psi) - v_y * jnp.sin(delta_psi)) / (1 - kap * e_lat)
        e_dot = (v_x * jnp.sin(delta_psi) + v_y * jnp.cos(delta_psi))
        delta_psi_dot = r - kap * s_dot

        #S_space deriv
        full_deriv = (1 / s_dot) * (jnp.concatenate((veh_deriv, jnp.array([s_dot, e_dot, delta_psi_dot]))))

        return full_deriv
    

    next_state_func = helper_funcs.euler_integrated_states(path_dyn, ds)
    return next_state_func


