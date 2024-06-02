import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def control_limits_from_veh_params(veh_model):
    # Through these control limits, we enforce that the car is front steeing, rear wheel drive, and cannot drive backward
    
    # Returns: [delta_f_min, fxf_min, fxr_min, delta_r_min], [delta_f_max, fxf_max, fxr_max, delta_r_max]

    #-------Braking [fxf] is limited by static friction w/ front tire---------
    #-------NOTE: May put braking on hold for now as its effect is speed dependant
    fzf, fzr = veh_model.axle_normal_forces()
    max_braking_fxf = fzf * veh_model.terrain_params["mu"] 

    #-------Bounds for FxR are determined by max available wheel torque and wheel radius
    max_rear_force = veh_model.vehicle_params["max_tau_kilonm"] * 1e3 * veh_model.vehicle_params["wheel_rad_m"] 

    max_steering_angle_rad = (np.pi / 180) * veh_model.vehicle_params["max_delta_deg"]
    lower_limit = np.array([-1.0 * max_steering_angle_rad, 0.0, 0.0, 0.0]) #Artifically cutting rear power
    upper_limit = np.array([max_steering_angle_rad, 0.0, 0.0, 0.0]) 

    return lower_limit, upper_limit

def s_spaced_states_to_world_positions(initial_position, 
                                 state_trajectory, ds):
    #Converts from vehicle velocities to world coordinate positions and velocities
    n = state_trajectory.shape[0]
    position_trajectory = np.zeros((n + 1, state_trajectory.shape[1]))
    position_trajectory[0] = initial_position

    xy_vels = np.zeros((n, 2)) 

    for i in range(n):
        ux, uy, r = state_trajectory[i]

        #Conversion from vehicle frame velocities to Cartesian Velocities
        sin_th = np.sin(position_trajectory[i, 2])
        cos_th = np.cos(position_trajectory[i, 2])

        
        x_vel = -1.0 * ux * sin_th + uy * cos_th
        y_vel = ux * cos_th + uy * sin_th

        xy_vels[i][0] = x_vel
        xy_vels[i][1] = y_vel

        s_deriv = np.sqrt(ux ** 2 + uy ** 2)
        time_step = ds / s_deriv
        position_trajectory[i + 1] = time_step * np.array([x_vel, y_vel, r]) + position_trajectory[i]
        
    return position_trajectory, xy_vels

def steady_state_init_and_control(veh_model, corner_radius_m, corner_vel_mps):
    l = veh_model.vehicle_params["a_m"] + veh_model.vehicle_params["b_m"]
    ack_steer_angle_rad = l / corner_radius_m

    delta_steady_f = ack_steer_angle_rad 

    veh_lin_state = jnp.array([corner_vel_mps, 0.0, 1.0 * corner_vel_mps / corner_radius_m])
    veh_lin_ctrl = jnp.array([1.0 * delta_steady_f, 0.0, 0.0, 0.0])

    def control_fn(veh_state):
        return veh_lin_ctrl

    return veh_lin_state, control_fn

#--------------------Plotting Functions-------------------------
def plot_veh_path(positions):
    xy_pos = positions[:, :2]

    plt.figure()
    ax = plt.gca()
    ax.plot(xy_pos[:, 0], xy_pos[:, 1])
    ax.set_aspect('equal')  # Set the aspect ratio to be equal
    plt.show()

def plot_psi_kappa(psi_traj, kappa_traj, ds):
    s_coords = np.arange(psi_traj.shape[0]) * ds
    plt.plot(s_coords, psi_traj)
    plt.plot(s_coords, kappa_traj)
    plt.show()

def plot_reference_traj_and_optimized_traj(reference_path_info, optimized_traj_info, 
                                           ds, initial_world_pos):
    #Plots optimized trajectory laid over reference trajectory
    veh_space_traj, world_xy, psi_traj, kappa_traj, interp_psi, interp_kappa = reference_path_info
    opt_traj_full_states, opt_traj_controls, _ = optimized_traj_info

    opt_traj_veh_states = opt_traj_full_states[:, :3]
    opt_traj_world_pos, opt_traj_world_vels = s_spaced_states_to_world_positions(initial_world_pos,
                                                    opt_traj_veh_states, ds)
    
    #-----------Plotting Function---------------
    ref_xy_pos = world_xy[:, :2]

    plt.figure()
    ax = plt.gca()
    ax.plot(ref_xy_pos[:, 0], ref_xy_pos[:, 1], "--") #plot reference trajectory
    ax.plot(opt_traj_world_pos[:, 0], opt_traj_world_pos[:, 1]) #plot optimized trajectory
    
    ax.set_xlabel("x displacenment (m)")
    ax.set_ylabel("y displacenment (m)")
    
    ax.set_aspect('equal')  # Set the aspect ratio to be equal


    plt.show()

def plot_veh_states(traj, ds):
    n = traj.shape[0]
    s_coord = np.arange(n) * ds

    fig, axs = plt.subplots(4, 1)

    combined_vel = np.linalg.norm(traj[:, 0:2], axis=1)
    #Plot experimental data
    axs[0].plot(s_coord, traj[:, 0])
    axs[1].plot(s_coord, traj[:, 1])
    axs[2].plot(s_coord, combined_vel)
    axs[3].plot(s_coord, traj[:, 2])

    axs[0].set_ylabel("ux")
    axs[1].set_ylabel("uy")
    axs[2].set_ylabel("v")
    axs[3].set_ylabel("r")

    plt.show()

def plot_veh_controls(controls_traj, dt):
    n = controls_traj.shape[0]
    t = np.arange(n) * dt

    fig, axs = plt.subplots(4, 1)

    #Plot experimental data
    axs[0].plot(t, controls_traj[:, 0])
    axs[1].plot(t, controls_traj[:, 1])
    axs[2].plot(t, controls_traj[:, 2])
    axs[3].plot(t, controls_traj[:, 3])

    axs[0].set_ylabel("delta_f")
    axs[1].set_ylabel("fxf")
    axs[2].set_ylabel("fxr")
    axs[3].set_ylabel("delta_r")

    plt.show()
