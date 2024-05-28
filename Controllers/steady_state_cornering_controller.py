import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

"""
Basic Controller to track a corner of fixed radius at a fixed velocity
"""
#TODO: Determine the linearization point for steady-state cornering

class SteadyStateCornering:
    def __init__(self, dyn_func, lin_state, lin_controls, Q, R):
        self.dyn_func = dyn_func
        self.lin_state = lin_state
        self.lin_controls = lin_controls

        A, B = jax.jacfwd(dyn_func, argnums=(0, 1))(lin_state, lin_controls)

        self.A = np.array(A)
        self.B = np.array(B)

        self.Q = Q
        self.R = R

    def determine_control_gain(self):
        #TODO: We could try re-linearizing A, B matrices at each iteration (this could cause instabilty)
        n = self.Q.shape[0]
        eps = 1e-4  # Riccati recursion convergence tolerance
        max_iters = 1000  # Riccati recursion maximum number of iterations
        P_prev = np.zeros((n, n))  # initialization
        converged = False
        for i in tqdm(range(max_iters)):
            K = -1.0 * np.linalg.inv(self.R + self.B.T @ P_prev @ self.B) @ self.B.T @ P_prev @ self.A
            P = self.Q + self.A.T @ P_prev @ (self.A + self.B @ K)
    
            p_diff = np.max(np.abs(P - P_prev))
            P_prev = P
            if p_diff <= eps:
                converged = True
                break
        if not converged:
            raise RuntimeError("Ricatti recursion did not converge!")
        
        self.gain_mat = K

    def control(self, in_state):
        return self.gain_mat @ (in_state - self.lin_state) + self.lin_controls
