from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm
import yaml

import matplotlib.pyplot as plt
import cvxpy as cvx

import scp_solver
import helper_funcs
import path_generation
from Models import bicycle_model

# Here, we test use SCP to follow a reference path.  This will help us 
# test how SCP performs when made to optimize a trajectory with dynamics
# that depend on hidden variable s

jax.config.update('jax_platform_name', 'cpu')

MPC_HORIZON = 15
MPC_ITERATIONS = 3

def s_spaceify(f):
    #NOTE: Only for use when using trajectory rollout to generate a path
    def s_spaced_f(z, u):
        s_dot = np.linalg.norm(z[0:2]) #magnitude of vehicle velocity
        return (1 / s_dot) * f(z, u)

    return s_spaced_f

def reference_control_func(z, s):
    t_val = int(int(s) / 5) % 2 == 0
    torque = (t_val) * (-0.1) + (t_val) * (0.3)
    
    return np.array([0.15, 0.0]) #Constrant Steering Trajectory to test


def euler_integrated_states(f, ds):
    # Defines first order euler integration function for given dynamics
    def integrated_next_state(z, u):
        return z + f(z, u) * ds
    
    return integrated_next_state

def runge_kutta_integrated_states(f, ds):
    # Defines classic runge-kutta integration (as used in hw2) for given dynamics
    def integrated_next_state(z, u):
        k1 = ds * f(z, u)
        k2 = ds * f(z + k1 / 2, u)
        k3 = ds * f(z + k2 / 2, u)
        k4 = ds * f(z + k3, u)
        return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrated_next_state

def get_vehicle_params(config_file):
    with open(config_file, "r") as yml_file:
        yml_content = yaml.safe_load(yml_file)
    
    return yml_content


def main():
    #---------Dynamics Model Initialization-------------------
    vehicle_params = get_vehicle_params("/Users/masonllewellyn/VehicleDynamics/my_own_sim/VehicleParams/takumi_params.yml")
    terrain_params = {"mu": 0.8}
    veh_model = bicycle_model.Model(vehicle_params, terrain_params)

    ds = 0.1
    t_dyn_func = veh_model.state_deriv_delta_fxr_control()
    s_space_dyn = s_spaceify(t_dyn_func)
    dyn_func = runge_kutta_integrated_states(s_space_dyn, ds) #euler_integrated_states(s_space_dyn, ds)

    #---------Path Creation and Path Space Model Initialization----------------
    rollout_horizon_m = 25.0
    initial_state = np.array([1.0, 0., 0.])

    path_info = path_generation.rollout_path(dyn_func, reference_control_func,
                                            initial_state, rollout_horizon_m, ds,
                                            control_m=2)
    
    ref_state_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_info
    kappa_traj = np.array(kappa_traj)
    psi_traj = np.array(psi_traj)


    helper_funcs.plot_veh_path(world_xy) 
    helper_funcs.plot_psi_kappa(psi_traj, kappa_traj, ds)
    p_space_dyn = path_generation.get_path_space_dyn_func(t_dyn_func, path_info, ds)

    
    # #-------Path Space Dyn Test
    N = ref_state_traj.shape[0]
    s_coords = np.arange(N) * ds
    n = 6
    m = 3
    
    #-----------Problem Formulation and Solve------------
    control_limits = (np.array([-1.0, -1.0, -0.5]), np.array([1.0, 1.0, 0.5]))
    goal_state = np.zeros((6,)) #Our goal is to have zero path lateral and heading error
    initial_path_state = np.array([1., 0., 0., 0., 0., 0.]) # Initialize path with nonzero forward motion

    m = 2

    Q = np.diag([0., 0., 0., 0., 10., 100.])
    P = 10 * Q
    R = np.eye(m)

    N = ref_state_traj.shape[0] - 1
 
    #Initial Nominal Trajectory

    z_init = None
    u_init = None

    s_coords = np.arange(N) * ds

    z_traj = np.zeros((N + 1, n))
    z_traj[0] = initial_path_state

    for k in tqdm(range(N)):
        opt_traj_len = min(MPC_HORIZON, N - k)
        traj_s_coords = s_coords[k : k + opt_traj_len]

        #----Cut Nominal Trajectory to within bounds of original path-------
        if z_init is not None and u_init is not None:
            z_init = z_init[:opt_traj_len + 1]
            u_init = u_init[:opt_traj_len]

        costs_at_iters, z_opt, u_opt = scp_solver.scp_solve(p_space_dyn, z_traj[k], control_limits, goal_state,
                                                            P, Q, R, ds, opt_traj_len, traj_s_coords=traj_s_coords, nominal_trajectory=z_init, 
                                                            nominal_control=u_init, enforce_feasible_dyn=False, max_iterations=MPC_ITERATIONS)
        
        #---Nominal trajectories will be linearized around in next iteration-------
        u_init = np.concatenate([u_opt[1:], u_opt[-1:]])
        z_init = np.concatenate(
            [z_opt[1:], p_space_dyn(z_opt[-1], z_opt[-1], traj_s_coords[-1]).reshape([1, -1])]
        )

        #-----Propogate Model to Next state------
        z_traj[k + 1] = p_space_dyn(z_traj[k] , u_opt[0], traj_s_coords[0])


    initial_world_pos = np.array([0., 0., 0.])

    #open_loop_xy = helper_funcs.roll_out_controls(dyn_func, initial_state, initial_world_pos, u_opt, ds)

    helper_funcs.plot_reference_traj_and_optimized_traj(path_info, z_traj, np.zeros((1, 2)), 
                                                        ds, initial_world_pos)
    helper_funcs.plot_path_space_errors(z_traj[:, -3:], ds)

if __name__ == "__main__":
    main()