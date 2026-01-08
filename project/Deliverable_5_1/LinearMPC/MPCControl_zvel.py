import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def compute_steady_state(self,r:np.ndarray,d_hat:np.ndarray)-> tuple[np.ndarray,np.ndarray]: 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        target = r[-1]
        dxss_var = cp.Variable(self.nx, name='xss')
        duss_var = cp.Variable(self.nu, name='uss')

        u_min = 40
        u_max = 80

        # Objective: minimize input squared
        ss_obj = cp.quad_form(duss_var, np.eye(self.nu))
        
        # Constraints: steady-state and input bounds
        ss_cons = [
            duss_var >= u_min - self.us,
            duss_var <= u_max - self.us,
            dxss_var == self.A @ dxss_var + self.B @ duss_var + self.B @ d_hat,
            self.C @ dxss_var == target - self.C @ self.xs.reshape(-1,1) - self.Cd @ d_hat,
        ]

        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve(solver=cp.PIQP)
        assert prob.status == cp.OPTIMAL
        xss = dxss_var.value + self.xs
        uss = duss_var.value + self.us
        
        return xss,uss

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        # Define variables
        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))
        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        x_ref.value = self.xs
        u_ref.value = self.us
        x_ref_col = x_ref.value.reshape(-1, 1)   # (nx,1)
        u_ref_col = u_ref.value.reshape(-1, 1)   # (nu,1)
        x_hat = cp.Parameter(self.nx, name='x0_hat')     # (estimated) initial state x0
        d_hat = cp.Parameter(1, 'd_hat') 
        
        
        Q = 50*np.eye(self.nx)# for tuning
        R = 0.1*np.eye(self.nu)

        self.L = self.compute_observer_gain()

        #constraints
        Hu = np.array([[ 1.],
                    [-1.]])
       
        U = Polyhedron.from_Hrep(Hu, np.array([80.0,-40.0]))
       
        # Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i]-x_ref), Q)
            cost += cp.quad_form((u_var[:,i]-u_ref), R)
                
        constraints = []
        constraints.append((x_var[:, 0]) == x_hat - x_ref_col)
        # System dynamics
        constraints.append((x_var[:,1:] - x_ref_col) == self.A @ (x_var[:,:-1] - x_ref_col) + self.B @ (u_var - u_ref_col))
        # Input constraints
        constraints.append(U.A @ (u_var - u_ref_col) <= U.b.reshape(-1, 1) - U.A @ u_ref_col)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var     # garde une référence pour get_u
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.x_hat = x0_var
        print("x_hat setup",self.x_hat.value)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x0_var.value = x0
        x_hat,d_hat = self.compute_estimates(x0)
        xss,uss = self.compute_steady_state(x_target,d_hat)
        self.x_hat.value = x_hat
        print("x_hat getu",self.x_hat.value)
        self.x_ref.value = xss
        self.u_ref.value = uss
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    def compute_estimates(self,x0:np.ndarray)-> None:
        # Attention x_hat est un parametre ce ocp
        print("befor")
        tmp = (
            self.A_hat @ np.concatenate((self.x_hat, self.d_hat)) +
            self.L @ ( x0 - self.C @ self.x_hat + self.Cd @ self.d_hat)
        )
        print("after")
        self.x_hat = tmp[:self.nx]
        self.d_hat = tmp[self.nx:]
    
    def compute_observer_gain(self)-> tuple[np.ndarray]:

        self.C = np.array([[1]])
        self.Cd = np.array([[1]])

        print("Cd",self.Cd)
        print("A",self.A)
        print("B",self.B)

        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, self.B)),
            np.hstack((0, 1))
        ))
        print("A_hat",self.A_hat)
        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((1, self.nu))))
        print("B_hat",self.B_hat)
        # C_hat = [C  Cd]
        self.C_hat = np.hstack((self.C, self.Cd))
        print("C_hat",self.C_hat)

        poles = np.array([0.5, 0.6])
        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        L = -res.gain_matrix.T
        print("A_hat",L)

        return L