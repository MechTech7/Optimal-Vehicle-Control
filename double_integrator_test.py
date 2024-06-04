from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm

import matplotlib.pyplot as plt
import cvxpy as cvx

import scp_solver

# In accordance with Dani's advice to "Strip it down until it works"
# I am implementing Sequential Convex Programming on a double-integrator to ensure that my strategy works.
# I will then attempt to use this double-integrator to track a path in path-fixed coordinates



#STEPS:
# 1. Dynamics: Double Integrator, Objective: Single Goal State, Dynamics Integration: Euler, Problem Integration: Euler
# 2. Dynamics: Double Integrator, Objective: Single Goal State, Dynamics Integration: Runge-Kutta, Problem Integration: Euler
# 3. Dynamics: Double Integrator, Objective: Path-Fixed Coordinates Path, Dynamics Integration: Euler, Problem Integration: Euler
# 4. Dynamics: Double Integrator, Objective: Path-Fixed Coordinates Path, Dynamics Integration: Euler, Problem Integration: Euler

jax.config.update('jax_platform_name', 'cpu')

def double_integrator_dynamics(s, u):
    # Dynamics for a double integrator represented as a unit mass
    # that can be controlled with forces in x and y
    # s : [x, y, x', y']
    # u : [Fx, Fy]

    state_deriv = jnp.array([s[2], s[3], u[0], u[1]])
    return state_deriv

def euler_integrated_states(f, dt):
    # Defines first order euler integration function for given dynamics
    def integrated_next_state(s, u):
        return s + f(s, u) * dt
    
    return integrated_next_state

def runge_kutta_integrated_states(f, dt):
    # Defines classic runge-kutta integration (as used in hw2) for given dynamics
    def integrated_next_state(s, u):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrated_next_state

def plot_trajectory(s_traj, goal_state, name="di_opt_trajectory"):
    plt.plot(s_traj[:, 0], s_traj[:, 1])
    plt.scatter([goal_state[0]], [goal_state[1]], color="green") #Scatter Plot w/ one point to show goal state
    plt.savefig(f"double_integrator_test_plots{name}.png", bbox_inches="tight")
    plt.show()

def main():
    dt = 0.1
    di_dyn_func = euler_integrated_states(double_integrator_dynamics, dt)

    goal_state = np.array([1., 1., 0., 0.])
    initial_state = np.array([0., 0., 0., 0.])

    control_limits = (np.array([-1., -1.]), np.array([1., 1.]))

    n = 4
    m = 2

    Q = np.eye(n)
    P = 10 * Q
    R = np.eye(m)

    N = 20
    costs_at_iters, s_opt, u_opt = scp_solver.scp_solve(di_dyn_func, initial_state, control_limits, goal_state,
                                                        P, Q, R, dt, N)
    

    plot_trajectory(s_opt, goal_state)


if __name__ == "__main__":
    main()