from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

from Models import bicycle_model
import cvxpy as cvx

jax.config.update('jax_platform_name', 'cpu')

#For simplicity (I don't wanna implement the Ricatti Eq) let's experiment with Sequential Quadratic Programming
#STEPS
# Define Dynamics (done in other file)
# Affinize Dynamics at each timestep
# Define Problem
    # Define Dynamics Constrains
    # Define Costs

#THE SOLUTION WAS A SIMPLE SCIPY INTERPOLATE ALL ALONG! LOL! WILL IMPLEMENT AND TEST
#Example of interpolation from Rajan's Code: 
#   self.psi_rad_interpFcn = scipy.interpolate.interp1d(self.centerline['s_m'], self.centerline['psi_rad'], axis = 0, kind = 'linear', fill_value = 'extrapolate');

@partial(jax.jit, static_argnums=(0,)) #Cache this affine function anytime f changes
@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def affine_dynamics(f, state, control_input, s_coord):
    #Linearize dynamics about (state, control_input)
    A, B = jax.jacfwd(f, argnums=(0, 1))(state, control_input, s_coord)
    c = -1.0 * (A @ state + B @ control_input) + f(state, control_input, s_coord)
    return A, B, c

#TODO: SCP Is currently returning "infeasible" after a couple of iterations. 
# Problem formulation / trust region might be incorrect


def scp_trajectory_opt_step(path_space_dynamics, 
                            control_limits,
                            initial_state, 
                            path_space_states_prev, 
                            path_controls_prev, 
                            N, P, Q, R, ds):
    n = Q.shape[0]
    m = R.shape[0]
    
    #--------Linearize Dynamics About Point-----------
    s_coords = np.arange(N) * ds
    A, B, c = affine_dynamics(path_space_dynamics, path_space_states_prev[:-1], path_controls_prev, s_coords)
    A, B, c = np.array(A), np.array(B), np.array(c)

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    #------------Final and Running Costs---------
    final_cost = cvx.quad_form(s_cvx[-1], P)
    running_control_cost = sum(cvx.quad_form(u_cvx[k], R) for k in range(N))
    running_state_cost = sum(cvx.quad_form(s_cvx[k], Q) for k in range(N))

    cost = final_cost + running_state_cost + running_control_cost

    #-----------Constraints-as-Costs-----------------
    
    #s_cvx[k + 1] = s_cvx[k] + 0.5 * ds * ()
    dynamics_cost = sum(cvx.norm(s_cvx[k + 1] - A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k], 2) for k in range(N))
    #lam = 25.0

    cost += dynamics_cost
    #-----------Constraints-------------

    control_lower_limit = control_limits[0]
    control_upper_limit = control_limits[1]

    constraints = []
    #Initial Condition Constraints
    constraints = [s_cvx[0] == initial_state]

    #Dynamics Constraints
    #constraints += [s_cvx[k + 1] == A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k] for k in range(N)]
    
    #Control Constraints
    constraints += [u_cvx[k] <= control_upper_limit for k in range(N)]
    constraints += [u_cvx[k] >= control_lower_limit for k in range(N)]

    #Trust Region Constraints (Value may need to be tuned)
    # constraints += [cvx.abs()]
    constraints += [cvx.norm(s_cvx[k] - path_space_states_prev[k], "inf") <= 1.0 for k in range(N + 1)]
    constraints += [cvx.norm(u_cvx[k] - path_controls_prev[k], "inf") <= 1.0 for k in range(N)]

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve(solver=cvx.MOSEK)

    if prob.status != "optimal":
        raise RuntimeError(f"SCP Solve Failed with status {prob.status}")
    new_path_states = s_cvx.value
    new_path_controls = u_cvx.value

    # print(f"final state cost: {final_cost.value}")
    # print(f"running state cost: {running_state_cost.value}")
    # print(f"running_control_cost: {running_control_cost.value}")

    return prob.objective.value, new_path_states, new_path_controls, final_cost.value, running_state_cost.value, running_control_cost.value

def plan_trajectory(path_space_dynamics, 
                    path_information,
                    control_limits,
                    ds,
                    P,
                    Q,
                    R,
                    eps = 1e-2,
                    max_iters = 50):
    # Given path information and path-space dynamics
    # Solve for a series of controls that gets the vehicle to follow the trajectory
    # Currently using sequential quadratic programming

    
    veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_information
    N_steps = veh_space_traj.shape[0] - 1

    #Path State Space: [vx, vy, r, s, e, delta_psi]
    #Initial State : Moving straight forward at 1mps
    initial_full_state = np.array([2., 0., 0., 0., 0., 0.])


    #Initialize Dynamically Feasible Zero-Control Trajectory
    n = Q.shape[0]
    m = R.shape[0]

    path_controls = np.zeros((N_steps, m))
    path_space_states = np.zeros((N_steps + 1, n))

    #Warm-Starting with a fixed steering angle
    #path_controls[:, 0] = 0.1
    s_coords = np.arange(N_steps) * ds
    path_space_states[0] = initial_full_state
    for i in range(N_steps):
        path_space_states[i + 1] = path_space_dynamics(path_space_states[i], path_controls[i], s_coords[i])

    
    A, B, c = affine_dynamics(path_space_dynamics, path_space_states[:-1], path_controls, s_coords)
    A, B, c = np.array(A), np.array(B), np.array(c)

    iteration_costs = []

    running_control_costs = []
    running_state_costs = []
    running_terminal_costs = []

    converged = False
    for i in tqdm(range(max_iters)):
        J, new_path_states, new_controls, final_cost, running_st_cost, running_ctrl_cost  = scp_trajectory_opt_step(path_space_dynamics, control_limits, initial_full_state,
                                                                path_space_states, path_controls, N_steps, P, Q, R, ds)

        iteration_costs.append(J)
        running_control_costs.append(running_ctrl_cost)
        running_state_costs.append(running_st_cost)
        running_terminal_costs.append(final_cost)

        if np.linalg.norm(new_controls - path_controls) <= eps:
            converged = True
            path_space_states = new_path_states
            path_controls = new_controls
            break

        path_controls = new_controls
        #Run generated controls through dynamics to ensure that every intialized trajectory is dynamically feasible

        
        
    
    if converged:
        print("-----------Control Converged-----------------------")

    return path_space_states, path_controls, np.array(iteration_costs), np.array(running_state_costs), np.array(running_terminal_costs), np.array(running_control_costs)

    





