import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

from Models import bicycle_model

"""
    Objective: 
        - Build a simulator that can simulate the motion of the vehicle in environments 
            with varying friction, etc.

    Test:
        - Forward Motion of Vehicle
            - FxR  : 3657 N (Enough to accelerate to ~60mph in ~4.7 seconds)
        - Constant Radius Circle
"""

#State: x, y, theta
class Simulator:
    def __init__(self, vehicle_model, state_n = 3, control_m = 4, dt = 0.02, ds = 0.01) -> None:
        self.vehicle_model = vehicle_model
        self.dt = dt
        self.ds = ds

        self.state_n = state_n
        self.control_m = control_m
        
    def rollout_trajectory(self, initial_state, rollout_horizon, control_func):
        #NOTE: States are in inertial frame
        state_trajectory = np.zeros((rollout_horizon + 1, self.state_n))
        controls_trajectory = np.zeros((rollout_horizon, self.control_m))
        veh_acc_trajectory = np.zeros((rollout_horizon, self.state_n))

        state_trajectory[0] = initial_state

        for i in range(1, rollout_horizon + 1):
            controls_trajectory[i - 1] = control_func(state_trajectory[i - 1]) #np.array([-1.0 * np.pi * 0.05, 0., -1500., 0.])
            
            state_derivs = self.vehicle_model.state_deriv_func(state_trajectory[i - 1], 
                                                               controls_trajectory[i - 1])
            
            veh_acc_trajectory[i - 1] = state_derivs
            #Euler Integration Step (NOTE: Experiment w/ more complex integration strategies)
            state_trajectory[i] = state_derivs * self.dt + state_trajectory[i - 1]
        
        return state_trajectory, controls_trajectory, veh_acc_trajectory
    
    def rollout_s_spaced_trajectory(self, initial_state, rollout_horizon, control_func):
        #Similar to rollout_trajectory however, states are spaced by a constant ds rather than a constant dt
        state_trajectory = np.zeros((rollout_horizon + 1, self.state_n))
        controls_trajectory = np.zeros((rollout_horizon, self.control_m))
        veh_acc_trajectory = np.zeros((rollout_horizon, self.state_n))

        state_trajectory[0] = initial_state

        for i in range(1, rollout_horizon + 1):
            controls_trajectory[i - 1] = control_func(state_trajectory[i - 1]) #np.array([-1.0 * np.pi * 0.05, 0., -1500., 0.])
            
            state_derivs = self.vehicle_model.state_deriv_func(state_trajectory[i - 1], 
                                                               controls_trajectory[i - 1])
            
            veh_acc_trajectory[i - 1] = state_derivs

            s_deriv = np.sqrt(state_trajectory[i - 1][0] ** 2 + state_trajectory[i - 1][1] ** 2)
            #Euler Integration Step (NOTE: Experiment w/ more complex integration strategies)

            time_step = self.ds / s_deriv
            state_trajectory[i] = state_derivs * time_step  + state_trajectory[i - 1]
        
        return state_trajectory, controls_trajectory, veh_acc_trajectory

    def roll_until_equilibrium(self, initial_state, set_control, max_iters = 1000000,  eps=1e-4):
        curr_state = initial_state
        converged = False

        for _ in tqdm(range(max_iters)):
            state_derivs = self.vehicle_model.state_deriv_func(curr_state, set_control)
            next_state = curr_state + state_derivs * self.dt

            if np.linalg.norm(next_state - curr_state) <= eps:
                converged = True
                curr_state = next_state
                break

            curr_state = next_state
        
        if converged != True:
            print("----Failed to Converge------")

        return curr_state

    def states_to_positions(self, initial_position, state_trajectory, acc_trajectory, dt):
        #Given an array of velocities, we integrate into a complete trajectory of positions
        n = state_trajectory.shape[0]
        position_trajectory = np.zeros((n + 1, state_trajectory.shape[1]))
        position_trajectory[0] = initial_position

        xy_vels = np.zeros((n, 2)) #Leaves first xy derivs to 0
        xy_accs = np.zeros((n, 2))

        for i in range(n):
            ux, uy, r = state_trajectory[i]
            ux_dot, uy_dot, r_dot = acc_trajectory[i] if i < n - 1 else np.zeros(acc_trajectory.shape[1])

            #Conversion from vehicle frame velocities to Cartesian Velocities
            sin_th = np.sin(position_trajectory[i, 2])
            cos_th = np.cos(position_trajectory[i, 2])

            x_vel = -1.0 * ux * sin_th + uy * cos_th
            y_vel = ux * cos_th + uy * sin_th

            x_acc = -1.0 * ux_dot * sin_th - ux * cos_th * r + uy_dot * cos_th - uy * sin_th * r
            y_acc = ux_dot * cos_th - ux * sin_th * r + uy_dot * sin_th + uy * cos_th * r


            xy_vels[i][0] = x_vel
            xy_vels[i][1] = y_vel

            xy_accs[i][0] = x_acc
            xy_accs[i][1] = y_acc

            position_trajectory[i + 1] = dt * np.array([x_vel, y_vel, r]) + position_trajectory[i]
        
        return position_trajectory, xy_vels, xy_accs
    
    def s_spaced_states_to_positions(self, initial_position, state_trajectory, acc_trajectory, ds):
        #Given an array of velocities, we integrate into a complete trajectory of positions
        n = state_trajectory.shape[0]
        position_trajectory = np.zeros((n + 1, state_trajectory.shape[1]))
        position_trajectory[0] = initial_position

        xy_vels = np.zeros((n, 2)) #Leaves first xy derivs to 0
        xy_accs = np.zeros((n, 2))

        for i in range(n):
            ux, uy, r = state_trajectory[i]
            ux_dot, uy_dot, r_dot = acc_trajectory[i] if i < n - 1 else np.zeros(acc_trajectory.shape[1])

            #Conversion from vehicle frame velocities to Cartesian Velocities
            sin_th = np.sin(position_trajectory[i, 2])
            cos_th = np.cos(position_trajectory[i, 2])

            x_vel = -1.0 * ux * sin_th + uy * cos_th
            y_vel = ux * cos_th + uy * sin_th

            x_acc = -1.0 * ux_dot * sin_th - ux * cos_th * r + uy_dot * cos_th - uy * sin_th * r
            y_acc = ux_dot * cos_th - ux * sin_th * r + uy_dot * sin_th + uy * cos_th * r


            xy_vels[i][0] = x_vel
            xy_vels[i][1] = y_vel

            xy_accs[i][0] = x_acc
            xy_accs[i][1] = y_acc

            s_deriv = np.sqrt(ux ** 2 + uy ** 2)
            time_step = ds / s_deriv
            position_trajectory[i + 1] = time_step * np.array([x_vel, y_vel, r]) + position_trajectory[i]
        
        return position_trajectory, xy_vels, xy_accs

    
    def positions_to_reference_trajectory(self, position_traj, xy_world_vels, xy_world_accs, dt):
        x_vels = xy_world_vels[:, 0]
        y_vels = xy_world_vels[:, 1]

        phi_ref = np.arctan2(-1.0 * x_vels, y_vels)
        
        # Giving up on the analytically for now: switching to derivatives by difference
        # phi_ref_deriv = x_vels * x_accs / denominator + y_vels * y_accs / denominator
        phi_ref_deriv = np.zeros(phi_ref.shape)

        # Difference calculation to get phi_ref derivative.  For use in curviture calculation kappa
        for i in range(phi_ref.shape[0] - 1):
            phi_ref_deriv[i] = (phi_ref[i + 1] - phi_ref[i]) / dt

        phi_ref_deriv[-1] = phi_ref_deriv[-2]

        s_deriv = np.linalg.norm(xy_world_vels, axis=1)
        kappa_ref = phi_ref_deriv / s_deriv

        return phi_ref, phi_ref_deriv, kappa_ref, s_deriv


def generate_reference_trajectory(veh_model, rollout_horizon_m):
    eq_state, steering_angle = steady_state_rollout(veh_model, corner_vel_mps=2., corner_radius_m=16.)

    def control_function(veh_state):
        return np.array([steering_angle, 0., 0., 0.])
    
    initial_state = np.array([2.0, 0., 0.])
    initial_position = np.array([0., 0., 0.])
    
    ref_sim = Simulator(veh_model)

    roll_n_steps = int(rollout_horizon_m / ref_sim.ds)
    state_traj, controls_traj, veh_acc_traj = ref_sim.rollout_s_spaced_trajectory(initial_state, roll_n_steps, control_function)
    
    world_pos_traj, xy_vels, xy_accs = ref_sim.s_spaced_states_to_positions(initial_position, state_traj, veh_acc_traj, ref_sim.ds)
    
    #Intermediate Plotting of States
    plot_veh_path(world_pos_traj)
    plot_veh_states(state_traj, ref_sim.ds)
    plot_veh_states(veh_acc_traj, ref_sim.ds)
    plot_veh_controls(controls_traj, ref_sim.ds)

    #NOTE: There is an error here: we are sampling path curviture in dt rather than ds
    phi_ref, phi_ref_deriv, kappa_ref, s_deriv = ref_sim.positions_to_reference_trajectory(world_pos_traj, xy_vels, xy_accs, ref_sim.dt)

    return phi_ref, phi_ref_deriv, kappa_ref, world_pos_traj, xy_vels, s_deriv


def steady_state_rollout(veh_model, corner_vel_mps, corner_radius_m):
    #Roll-out sim at an initial steering angle and velocity 
    #(corresponding to an idealized corner radius at a set velocity)
    
    ss_sim = Simulator(veh_model)
    l = veh_model.vehicle_params["a_m"] + veh_model.vehicle_params["b_m"]
    ack_steer_angle_rad = l / corner_radius_m

    delta_steady_f = ack_steer_angle_rad 
    
    veh_lin_state = jnp.array([corner_vel_mps, 0.0, -1.0 * corner_vel_mps / corner_radius_m])
    veh_lin_ctrl = jnp.array([-1.0 * delta_steady_f, 0.0, 0.0, 0.0])

    eq_position = ss_sim.roll_until_equilibrium(veh_lin_state, veh_lin_ctrl)
    return eq_position, delta_steady_f

def plot_veh_path(positions):
    xy_pos = positions[:, :2]

    plt.figure()
    ax = plt.gca()
    ax.plot(xy_pos[:, 0], xy_pos[:, 1])
    ax.set_aspect('equal')  # Set the aspect ratio to be equal
    plt.show()

def plot_veh_states(traj, dt):
    n = traj.shape[0]
    t = np.arange(n) * dt

    fig, axs = plt.subplots(4, 1)

    combined_vel = np.linalg.norm(traj[:, 0:2], axis=1)
    #Plot experimental data
    axs[0].plot(t, traj[:, 0])
    axs[1].plot(t, traj[:, 1])
    axs[2].plot(t, combined_vel)
    axs[3].plot(t, traj[:, 2])

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

def get_vehicle_params(config_file):
    with open(config_file, "r") as yml_file:
        yml_content = yaml.safe_load(yml_file)
    
    return yml_content

def main():
    vehicle_params = get_vehicle_params("/Users/masonllewellyn/VehicleDynamics/my_own_sim/VehicleParams/takumi_params.yml")
    terrain_params = {"mu": 0.8}
    veh_model = bicycle_model.Model(vehicle_params, terrain_params)
    

    rollout_horizon_m = 50
    phi_ref, phi_ref_deriv, kappa_ref, world_pos, world_vels, s_deriv  = generate_reference_trajectory(veh_model, rollout_horizon_m)

    # print(phi_ref_deriv)
    t = np.arange(phi_ref.shape[0]) * 0.01
    plt.plot(t, phi_ref)
    plt.plot(t, phi_ref_deriv)
    plt.show()


if __name__ == "__main__":
    main()