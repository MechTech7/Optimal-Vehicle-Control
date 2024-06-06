from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm

import matplotlib.pyplot as plt
import cvxpy as cvx

jax.config.update('jax_platform_name', 'cpu')

@partial(jax.jit, static_argnums=(0,)) #Cache this affine function anytime f changes
@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def affine_dynamics(f, state, control_input, s_coord):
    #Linearize dynamics about (state, control_input)
    A, B = jax.jacfwd(f, argnums=(0, 1))(state, control_input, s_coord)
    c = -1.0 * (A @ state + B @ control_input) + f(state, control_input, s_coord)
    return A, B, c

def scp_iteration(A, B, c, 
                  initial_state,
                  control_limits,
                  goal_state,
                  prev_state_traj,
                  prev_control_traj,
                  N,
                  tau,
                  P, Q, R, dt):
    n = Q.shape[0]
    m = R.shape[0]

    #-----------Define Variables and Convex Objectives------------
    z_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    terminal_cost = cvx.quad_form(z_cvx[-1] - goal_state, P)
    running_control_cost = sum(cvx.quad_form(u_cvx[k], R) for k in range(N))
    running_state_cost = sum(cvx.quad_form(z_cvx[k] - goal_state, Q) for k in range(N))

    cost = terminal_cost + running_control_cost + running_state_cost
    #----------Dynamics, State, and Control Constraints--------
    lower_control_bound, upper_control_bound = control_limits

    constraints = [z_cvx[0] == initial_state]
    constraints += [z_cvx[k + 1] == A[k] @ z_cvx[k] + B[k] @ u_cvx[k] + c[k] for k in range(N)]

    #Control Constraints
    constraints += [u_cvx[k] <= upper_control_bound for k in range(N)]
    constraints += [u_cvx[k] >= lower_control_bound for k in range(N)]

    #Trust Region Constraints (Shouldn't be necessary for )
    constraints += [cvx.norm(z_cvx[k] - prev_state_traj[k], 2) <= tau for k in range(N + 1)]
    constraints += [cvx.norm(u_cvx[k] - prev_control_traj[k], 2) <= tau for k in range(N)]

    #----------Problem Solve---------------

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()

    if prob.status != "optimal":
        raise RuntimeError(f"SCP Solve Failed with status {prob.status}")
    
    cost_information = (terminal_cost.value, running_state_cost.value, running_control_cost.value)
    return cost_information, z_cvx.value, u_cvx.value


def scp_solve(dyn_func, initial_state, control_limits, goal_state, 
              P, Q, R, ds, N, traj_s_coords = None, nominal_trajectory = None, nominal_control = None,
              max_iterations = 50, break_on_non_converge = False, eps = 1e-2, enforce_feasible_dyn = False):
    #TODO: Add path displacement s
    n = Q.shape[0]
    m = R.shape[0]

    if traj_s_coords is not None:
        if traj_s_coords.shape[0] != N:
            raise RuntimeError("s_coords from trajectory must be of lenght N")
        s_coords = traj_s_coords
    else:
        s_coords = np.arange(N) * ds

    z = np.zeros((N + 1, n))
    u = np.zeros((N, m))

    terminal_costs = []
    running_state_costs = []
    running_control_costs = []

    total_costs = []

    z_traj_prev = np.zeros((N + 1, n))
    u_traj_prev = np.zeros((N, m))

    converged = False

    if nominal_trajectory is not None and nominal_control is not None:
        #If a nominal trajectory is provided, such as within an MPC loop, use that as an initial trajectory
        if nominal_trajectory.shape[0] != N + 1 or nominal_control.shape[0] != N:

            raise RuntimeError("Provided nominal state trajectory must be of length N + 1 and nominal control trajectory must be of length N")
        # if not np.allclose(nominal_trajectory[0], initial_state):
        #     raise RuntimeError("Pedantic I know but the nominal trajectory must start with the initial_state")
        
        z_traj_prev = nominal_trajectory
        u_traj_prev = nominal_control

    else:
        #---------Initial, zero-control trajectory-------------
        z_traj_prev[0] = initial_state
        for k in range(N):
            z_traj_prev[k + 1] = dyn_func(z_traj_prev[k], u_traj_prev[k], s_coords[k])

    for i in range(max_iterations):
        A, B, c = affine_dynamics(dyn_func, z_traj_prev[:-1], u_traj_prev, s_coords)
        A, B, c = np.array(A), np.array(B), np.array(c)

        trust_region_tau = 5.0
        cost_info, z, u = scp_iteration(A, B, c,
                                        initial_state,
                                        control_limits,
                                        goal_state,
                                        z_traj_prev,
                                        u_traj_prev,
                                        N,
                                        trust_region_tau,
                                        P, Q, R, ds)
        
        terminal_cost, r_state_cost, r_ctrl_cost = cost_info

        terminal_costs.append(terminal_cost)
        running_state_costs.append(r_state_cost)
        running_control_costs.append(r_ctrl_cost)

        t_cost = terminal_cost + r_state_cost + r_ctrl_cost
        total_costs.append(t_cost)

        if i > 0 and np.abs(total_costs[i - 1] - t_cost) < eps:
            converged = True
            if enforce_feasible_dyn:
                for i in range(N):
                    z[i + 1] = dyn_func(z[i], u[i], s_coords[i])
            break
    
        #An idea that I had: 
        #   After very solve iteration, run the results through state dynamics to 
        #   ensure that every solution we linearize about is dynamically feasible.
        #   This might prove to be unhelpful but I figured it would be better to have it to test
        if enforce_feasible_dyn:
            for i in range(N):
                z[i + 1] = dyn_func(z[i], u[i], s_coords[i])
        
        z_traj_prev = z
        u_traj_prev = u

    if not converged and break_on_non_converge:
        raise RuntimeError("SCP did not converge. Consider increasing max_iterations or investigating your problem formulation")
    
    costs_at_iters = (total_costs, terminal_costs, running_state_costs, running_control_costs)
    return costs_at_iters, z, u