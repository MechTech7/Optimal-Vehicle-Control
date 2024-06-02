import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt

from Models import bicycle_model
import path_generation
import path_problem_sim
import helper_funcs


#TODO's From NOW:
# - Test Path Generation Function (Done)
#    - Plot world coords from circular path (Done)
#    - Plot Kappa and Psi Values (Done)

# - Define Optimal Control Problem for full-path optimization
# - Test Optimal Controller in path-space coordinates
# -     Optimize entire trajectory using iLQR or SCP

jax.config.update('jax_platform_name', 'cpu')

def get_vehicle_params(config_file):
    with open(config_file, "r") as yml_file:
        yml_content = yaml.safe_load(yml_file)
    
    return yml_content

def main():
    vehicle_params = get_vehicle_params("/Users/masonllewellyn/VehicleDynamics/my_own_sim/VehicleParams/takumi_params.yml")
    terrain_params = {"mu": 0.8}
    veh_model = bicycle_model.Model(vehicle_params, terrain_params)
    dyn_model = veh_model.get_state_deriv_func()

    ds = 0.1

    veh_lin_state, ss_control_fn = helper_funcs.steady_state_init_and_control(veh_model, corner_radius_m=16., corner_vel_mps=2.)

    rollout_horizon_m = 50.

    
    path_information = path_generation.rollout_path(dyn_model, ss_control_fn, 
                                                    veh_lin_state, rollout_horizon_m, ds)
    
    veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = path_information

    path_space_dyn = path_generation.get_path_space_dyn_func(dyn_model, path_information, ds)
    control_limits = helper_funcs.control_limits_from_veh_params(veh_model)

    #------Aaaaand Showtime!------------
    Q = np.diag([0., 0., 0., 0., 0.1, 0.1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    P = 10.0 * Q
    opt_traj_information = path_problem_sim.plan_trajectory(path_space_dyn, path_information, control_limits, ds, P, Q, R, max_iters=0)
    opt_traj_path_space, opt_traj_controls, iteration_costs = opt_traj_information

    
    plt.plot(np.arange(iteration_costs.shape[0]), iteration_costs)
    plt.show()

    #---------Plot reference path and trajectory overlay------------
    initial_world_pos = np.zeros((3,))
    helper_funcs.plot_reference_traj_and_optimized_traj(path_information, opt_traj_information, ds, initial_world_pos)
    
    helper_funcs.plot_veh_states(opt_traj_path_space[:, :3], ds)
    print("En los antros de lujo!!!!!!")


if __name__ == "__main__":
    main()
