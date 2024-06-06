import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt

from Models import bicycle_model
import path_generation
import path_problem_sim
import helper_funcs

# NOTE:
# Current State of Things
#   - Optimization works partially: trajectories approach reference before the trajectory becomes unable to be optimized
#       - Within these failures, I think, lies the reason for the failures and the key to success

# Problems:
#   1. Path optimization fails with "infeasible" and / or inaccurate.  Checking solves before failure shows non-convergance.
#   2. (And this is more general) Path Optimization is very brittle to certain variables
#       - Path length
#       - ds
#       - vehicle state initialization


#TODO: Things to Try
# - Plot running state and control costs
# - Attempt with reference trajectory with smaller steering angle
# - Investigate Problem with larger ds (0.5)
#       - Currently, the convex program fails with this ds however, 
#           a larger ds would reduce the number of steps and make the optmization simpler.
#       - 



jax.config.update('jax_platform_name', 'cpu')

def get_vehicle_params(config_file):
    with open(config_file, "r") as yml_file:
        yml_content = yaml.safe_load(yml_file)
    
    return yml_content

def s_spaceify(f):
    #NOTE: Only for use when using trajectory rollout to generate a path
    def s_spaced_f(z, u):
        s_dot = np.linalg.norm(z[0:2]) #magnitude of vehicle velocity
        return (1 / s_dot) * f(z, u)

    return s_spaced_f

def euler_integrated_states(f, ds):
    # Defines first order euler integration function for given dynamics
    def integrated_next_state(z, u):
        return z + f(z, u) * ds
    
    return integrated_next_state

def main():
    vehicle_params = get_vehicle_params("/Users/masonllewellyn/VehicleDynamics/my_own_sim/VehicleParams/takumi_params.yml")
    terrain_params = {"mu": 0.8}
    veh_model = bicycle_model.Model(vehicle_params, terrain_params)
    dyn_func = veh_model.get_state_deriv_func()

    ds = 0.1

    eu_int_dyn = euler_integrated_states(s_spaceify(dyn_func), 0.1)
    veh_lin_state, ss_control_fn = helper_funcs.steady_state_init_and_control(veh_model, corner_radius_m=16., corner_vel_mps=2.)

    rollout_horizon_m = 25.

    
    path_information = path_generation.rollout_path(eu_int_dyn, ss_control_fn, 
                                                    veh_lin_state, rollout_horizon_m, ds, control_m=4)
    
    veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_information

    helper_funcs.plot_veh_path(world_xy)
    helper_funcs.plot_psi_kappa(psi_traj, kappa_traj, ds)

    # path_space_dyn = path_generation.get_path_space_dyn_func(dyn_model, path_information, ds)
    # control_limits = helper_funcs.control_limits_from_veh_params(veh_model)

    # # #------Aaaaand Showtime!------------
    Q = np.diag([0., 0., 0., 0., 0.01, 0.01]) * (ds / rollout_horizon_m)
    R = np.diag([0.01, 0.01, 0.01, 0.01])
    P = Q
    opt_traj_information = path_problem_sim.plan_trajectory(path_space_dyn, path_information, control_limits, ds, P, Q, R, max_iters=10)
    opt_traj_path_space, opt_traj_controls, iteration_costs, running_state_costs, terminal_costs, running_control_costs = opt_traj_information

    helper_funcs.plot_opt_traj_information(opt_traj_information, ds)
    
    # # # # plt.plot(np.arange(iteration_costs.shape[0]), iteration_costs)
    # # # # plt.show()

    # # # #---------Plot reference path and trajectory overlay------------
    initial_world_pos = np.zeros((3,))
    helper_funcs.plot_reference_traj_and_optimized_traj(path_information, opt_traj_information, ds, initial_world_pos)
    
    # helper_funcs.plot_veh_states(opt_traj_path_space[:, :3], ds)
    # helper_funcs.plot_path_space_errors(opt_traj_path_space[:, 3:], ds)
    helper_funcs.plot_veh_controls(opt_traj_controls, ds)

    print("En los antros de lujo!!!!!!")


if __name__ == "__main__":
    main()
