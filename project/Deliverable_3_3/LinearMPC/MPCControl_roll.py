import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def compute_steady_state(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """
        Compute the reference state and input for roll angle tracking.
        For roll angle tracking, we create a reference with the target angle and zero rate.
        """
        r = np.array(r).reshape((-1,))
        roll_target = r[0]  # Target roll angle
        
        # Reference state: [target_roll_angle, 0_roll_rate]
        xss = np.array([roll_target, 0.0])
        
        # Reference input: equilibrium input (zero torque at hover)
        uss = self.us
        
        return xss, uss

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE


        Q = 1*np.eye(self.nx)  # for tuning
        R = 1*np.eye(self.nu)

        # No terminal constraints for reference tracking (Deliverable 3.3)
        # Terminal weight Qf and terminal controller K (commented out for 3.3)
        # K, Qf, _ = dlqr(self.A, self.B, Q, R)
        # K = -K
        # A_cl = self.A + self.B @ K

        # Constraints
        Hu = np.array([[ 1.],
                       [-1.]])
       
        U = Polyhedron.from_Hrep(Hu, np.array([20.0 - self.us[0], 20.0 + self.us[0]]))

        # Define variables
        xs_col = self.xs.reshape(-1, 1)   # (nx,1)
        us_col = self.us.reshape(-1, 1)   # (nu,1)

        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))

        # Reference parameters for tracking
        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        x_ref.value = self.xs
        u_ref.value = self.us

        # Costs - now using reference parameters
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:, i] - cp.reshape(x_ref, (self.nx,))), Q)
            cost += cp.quad_form((u_var[:, i] - cp.reshape(u_ref, (self.nu,))), R)

        # No terminal cost for reference tracking (Deliverable 3.3)
        # cost += cp.quad_form((x_var[:, -1] - self.xs), Qf)
                
        constraints = []

        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)
        
        # System dynamics - use deviation from equilibrium
        constraints.append((x_var[:, 1:] - xs_col) == self.A @ (x_var[:, :-1] - xs_col) + self.B @ (u_var - us_col))
        
        # Input constraints
        constraints.append(U.A @ (u_var - us_col) <= U.b.reshape(-1, 1))
        
        # No terminal constraints for reference tracking (Deliverable 3.3)
        # constraints.append(O.A @ (x_var[:, -1] - xs_col) <= O.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE  
        if x_target is None:
            x_target = self.xs
        xss, uss = self.compute_steady_state(x_target)
        
        self.x_ref.value = xss
        self.u_ref.value = uss 
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
