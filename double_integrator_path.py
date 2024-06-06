from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm

import matplotlib.pyplot as plt
import cvxpy as cvx

import scp_solver
import helper_funcs
import path_gen_double_integrator

# Here, we test use SCP to follow a reference path.  This will help us 
# test how SCP performs when made to optimize a trajectory with dynamics
# that depend on hidden variable s

jax.config.update('jax_platform_name', 'cpu')

def double_integrator_dynamics(z, u):
    # Dynamics for a double integrator represented as a unit mass
    # that can be controlled with forces and torques in "vehicle-frame" x, y, and rotation about z
    # z : [v_x, v_y, r]
    # u : [Fx, Fy, tau]

    state_deriv = jnp.array([u[0], u[1], u[2]])
    return state_deriv

def s_space_double_integrator(z, u):
    s_dot = np.linalg.norm(z[0:2]) #magnitude of vehicle velocity

    return (1 / s_dot) * double_integrator_dynamics(z, u)


def reference_control_func(z, s):
    torque = (s < 5.0) * (-0.1) + (s > 5.0) * (0.3)
    return np.array([0.1, 0., torque])


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

def main():
    #---------Dynamics Model Initialization-------------------
    ds = 0.1
    rollout_horizon_m = 10.
    dyn_func = euler_integrated_states(s_space_double_integrator, ds)
    control_limits = (np.array([-1.0, -1.0, -0.5]), np.array([1.0, 1.0, 0.5]))
    initial_state = np.array([1.0, 0., 0.])


    #---------Path Space Model Initialization----------------
    path_info = path_gen_double_integrator.rollout_path(dyn_func, reference_control_func,
                                                        initial_state, rollout_horizon_m, ds)
    
    ref_state_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_info
    kappa_traj = np.array(kappa_traj)
    psi_traj = np.array(psi_traj)


    #helper_funcs.plot_veh_path(world_xy) # NOTE: delta_psi behaves strangely when path points opposite direction to vehicle trajectory
    #helper_funcs.plot_psi_kappa(psi_traj, kappa_traj, ds)
    p_space_dyn = path_gen_double_integrator.get_path_space_dyn_func(double_integrator_dynamics, path_info, ds)

    
    #-------Path Space Dyn Test
    N = ref_state_traj.shape[0]
    s_coords = np.arange(N) * ds
    n = 6
    m = 3

    # z = np.zeros((N + 1, n))
    # u = np.zeros((N, m))

    # #---------Initial, zero-control trajectory-------------
    # z[0] = np.array([1., 0., 0., 0., 0., 0.])
    # for k in range(N):
    #     z[k + 1] = p_space_dyn(z[k], u[k], s_coords[k])

    # initial_world_pos = np.array([0., 0., 0.])
    # helper_funcs.plot_reference_traj_and_optimized_traj(path_info, z, ds, initial_world_pos)

    # p_states = z[:, 2:]
    # helper_funcs.plot_path_space_errors(p_states, ds)
    
    #-----------Problem Formulation and Solve------------
    goal_state = np.zeros((6,)) #Our goal is to have zero path lateral and heading error
    initial_path_state = np.array([1.0, 0., 0., 0., 0., 0.]) # Initialize path with nonzero forward motion

    m = 3

    Q = np.diag([0., 0., 0., 0., 10., 100.])
    P = 10 * Q
    R = np.eye(m)

    N = ref_state_traj.shape[0] - 1

    costs_at_iters, z_opt, u_opt = scp_solver.scp_solve(p_space_dyn, initial_path_state, control_limits, goal_state,
                                                        P, Q, R, ds, N, enforce_feasible_dyn=True, max_iterations=50)
    
    initial_world_pos = np.array([0., 0., 0.])

    open_loop_traj, open_loop_xy = helper_funcs.roll_out_controls(path_info, p_space_dyn, 
                                                  initial_path_state, initial_world_pos, u_opt, ds, random_pert=True)

    helper_funcs.plot_reference_traj_and_optimized_traj(path_info, z_opt, open_loop_xy, 
                                                        ds, initial_world_pos)
    helper_funcs.plot_path_space_errors(open_loop_traj[:, -3:], ds)

if __name__ == "__main__":
    main()