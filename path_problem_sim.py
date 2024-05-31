import numpy as np
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

from Models import bicycle_model


#For simplicity (I don't wanna implement the Ricatti Eq) let's experiment with Sequential Quadratic Programming
#STEPS
# Define Dynamics 
# Affinize Dynamics at each timestep
# Define Problem
    # Define Dynamics Constrains
    # Define Costs

#THE SOLUTION WAS A SIMPLE SCIPY INTERPOLATE ALL ALONG! LOL! WILL IMPLEMENT AND TEST
#Example of interpolation from Rajan's Code: 
#   self.psi_rad_interpFcn = scipy.interpolate.interp1d(self.centerline['s_m'], self.centerline['psi_rad'], axis = 0, kind = 'linear', fill_value = 'extrapolate');

def get_veh_and_path_dyn(veh_dyn_func, ref_path_info):
    kappa_ref, s_deriv_ref = ref_path_info

    @jax.jit
    def veh_and_path_dyn(vehicle_state, control_inputs, path_fixed_state, s_idx):
        v_x, v_y, r = vehicle_state
        e_lat, d_psi = path_fixed_state
        
        inertial_deriv = veh_dyn_func(vehicle_state, control_inputs)
        #---------Calculation of state derivatives in Path-Fixed Coordinates-----------

        s_dot = (v_x * jnp.cos(d_psi) - v_y * jnp.sin(d_psi)) / (1 - kappa_ref[s_idx])
        e_dot = (v_x * jnp.sin(d_psi) + v_y * jnp.cos(d_psi))
        d_psi_dot = r - kappa_ref[s_idx] * s_dot

        return jnp.concatenate(inertial_deriv, jnp.array([s_dot, e_dot, d_psi_dot]))


    return veh_and_path_dyn



def main():
    pass


if __name__ == "__main__":
    main()