import numpy as np
from scipy import constants
import jax
import jax.numpy as jnp

#TODO: Make model more JAX-esque
#       - Remove "Model" class and create a larget "generator" function that returns a JIT'd state derivative function
jax.config.update('jax_platform_name', 'cpu')

def slip_cond(alpha_f, _, mu, fz_front):
    return -1.0 * fz_front * mu * jnp.sign(alpha_f)

def linear_cond(alpha_f, c_stiff, mu, fz_front):
    tan_a = jnp.tan(alpha_f)
    return -1.0 * c_stiff * tan_a + (c_stiff ** 2 / (3 * mu * fz_front)) * jnp.abs(tan_a) * tan_a - (c_stiff ** 3 / (27 * mu ** 2 * fz_front ** 2)) * (tan_a ** 3)

class Model:
    def __init__(self, 
                 vehicle_params, 
                 terrain_params) -> None:
        self.vehicle_params = vehicle_params
        self.terrain_params = terrain_params

        self.jit_derivs = jax.jit(self.state_derivative)
        self.state_deriv_func = self.get_state_deriv_func()

    def tire_slip_angles(self, ux, uy, r, delta_f, delta_r, a, b):
        alpha_f = jnp.arctan2(uy + a * r, ux) - delta_f
        alpha_r = jnp.arctan2(uy - b * r, ux) - delta_r

        return alpha_f, alpha_r

    def axle_normal_forces(self):
        #Calculates Fz for both front and rear
        #Currently assuming static weight distribution
        a_b_sum = self.vehicle_params["a_m"] + self.vehicle_params["b_m"]
        fzf = constants.g * self.vehicle_params["m_kilog"] * (self.vehicle_params["b_m"] / a_b_sum)
        fzr = constants.g * self.vehicle_params["m_kilog"] * (self.vehicle_params["a_m"] / a_b_sum)
        return fzf, fzr
    
    def state_derivative(self, vehicle_state, control_inputs):
        #TODO: Remove this function as its been made obsolete by JIT function
        delta_f, fxf, fxr, delta_r = control_inputs
        ux, uy, r = vehicle_state
        a = self.vehicle_params["a_m"]
        b = self.vehicle_params["b_m"]

        alpha_f, alpha_r = self.tire_slip_angles(ux, uy, r, delta_f, delta_r, a, b)
        fzf, fzr = self.axle_normal_forces()

        #print(f"alpha_f: {alpha_f} alpha_r: {alpha_r}")
        #NOTE: Using Fiala Model for Both Fyf and Fyr (Can add slip-specific models later)


        fyf = self.fy_fiala(fzf, alpha_f)
        fyr = self.fy_fiala(fzr, alpha_r)

        #fyf = self.fy_front(fzf, alpha_f)
        #fyr = self.fy_front(fzr, alpha_r)

        inv_mass = 1 / self.vehicle_params["m_kilog"]
        

        cos_df = jnp.cos(delta_f)
        sin_df = jnp.sin(delta_f)
        cos_dr = jnp.cos(delta_r)
        sin_dr = jnp.sin(delta_r)

        d_ux = inv_mass * (-1.0 * fxf * cos_df - fyf * sin_df - fxr * cos_dr - fyr * sin_dr) + uy * r
        d_uy = inv_mass * (fyf * cos_df - fxf * sin_df + fyr * cos_dr - fxr * sin_dr) - ux * r
        d_r = (1 / self.vehicle_params["iz_m2kilog"]) * (a * (fyf * cos_df - fxf * sin_df) - b * (fyr * cos_dr - fxr * sin_dr))

        return jnp.array([d_ux, d_uy, d_r])
    
    def get_state_deriv_func(self):
        #Returns JIT'd function that computes state derivatives
        @jax.jit
        def state_deriv(vehicle_state, control_inputs):
            delta_f, fxf, fxr, delta_r = control_inputs
            ux, uy, r = vehicle_state
            a = self.vehicle_params["a_m"]
            b = self.vehicle_params["b_m"]

            alpha_f, alpha_r = self.tire_slip_angles(ux, uy, r, delta_f, delta_r, a, b)
            fzf, fzr = self.axle_normal_forces()

            #print(f"alpha_f: {alpha_f} alpha_r: {alpha_r}")
            #NOTE: Using Fiala Model for Both Fyf and Fyr (Can add slip-specific models later)


            fyf = self.fy_fiala(fzf, alpha_f)
            fyr = self.fy_fiala(fzr, alpha_r)

            #fyf = self.fy_front(fzf, alpha_f)
            #fyr = self.fy_front(fzr, alpha_r)

            inv_mass = 1 / self.vehicle_params["m_kilog"]
        

            cos_df = jnp.cos(delta_f)
            sin_df = jnp.sin(delta_f)
            cos_dr = jnp.cos(delta_r)
            sin_dr = jnp.sin(delta_r)

            d_ux = inv_mass * (-1.0 * fxf * cos_df - fyf * sin_df - fxr * cos_dr - fyr * sin_dr) + uy * r
            d_uy = inv_mass * (fyf * cos_df - fxf * sin_df + fyr * cos_dr - fxr * sin_dr) - ux * r
            d_r = (1 / self.vehicle_params["iz_m2kilog"]) * (a * (fyf * cos_df - fxf * sin_df) - b * (fyr * cos_dr - fxr * sin_dr))

            return jnp.array([d_ux, d_uy, d_r])
        
        return state_deriv

    def state_deriv_delta_fxr_control(self):
        full_state_deriv_func = self.get_state_deriv_func()

        @jax.jit
        def state_deriv(vehicle_state, truncated_control):
            # state: [ux, uy, r]
            # control: [delta_f, fx_r]
            delta_f, fx_r = truncated_control

            full_control = jnp.array([delta_f, 0., fx_r, 0.])
            
            state_deriv = full_state_deriv_func(vehicle_state, full_control)
            return state_deriv

        return state_deriv
    #Calculate Front Tire Forces Using Fiala Model
    def fy_fiala(self, fz_front, alpha_f):
        c_stiff = self.vehicle_params["c_stiffness_n_p_rad"]
        mu = self.terrain_params["mu"]
        thresh_alpha = jnp.arctan2(3 * fz_front * mu, c_stiff)
        
        # To make function compatible with JAX operations (such as jit) 
        # we replace traditional Python branching with jax.lax.cond
        
        f_out = -1.0 * fz_front * mu * c_stiff * alpha_f

        #NOTE: Full Fiala model does not work well with the system
        
        # f_out = jax.lax.cond(jnp.abs(alpha_f) >= thresh_alpha,
        #              slip_cond,
        #              linear_cond,
        #              alpha_f, c_stiff, mu, fz_front, #operands to functions
        #              )
        return f_out

