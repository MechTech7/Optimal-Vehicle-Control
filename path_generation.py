import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import yaml
from tqdm import tqdm

# This will create the nominal paths for the vehicle to track
# Paths will be created by rolling out the vehicle simulator and 

def rollout_path(dyn_model, control_fn, 
                 initial_veh_state, rollout_horizon_m,
                 ds = 0.01):
    n_steps = int(rollout_horizon_m / ds)
    
