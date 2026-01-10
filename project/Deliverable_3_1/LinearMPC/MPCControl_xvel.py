import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        Q = np.diag([5.0, 200.0, 50.0])# for tuning
        R = 1*np.eye(self.nu)

        # Terminal weight Qf and terminal controller K
        K,Qf,_ = dlqr(self.A,self.B,Q,R)
        K = -K

        A_cl = self.A + self.B @ K

        #constraints
        Hx = np.array([[0., 1., 0.],
               [0.,-1., 0.]])
        kx = np.array([0.1745, 0.1745])

        Hu = np.array([[ 1.],
                    [-1.]])
        ku = np.array([0.26,0.26])

        X = Polyhedron.from_Hrep(Hx, kx - (Hx @ self.xs))
        U = Polyhedron.from_Hrep(Hu, ku - (Hu @ self.us))  
       

        # maximum inavariant set for recusive feasability
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O = X.intersect(KU)
        
        max_iter = 30
        for iter in range(max_iter):  
            Oprev = O
            F,f = O.A,O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            if O == Oprev:
                break
        
        #plot max invariance set
       
        # Define variables
        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))

        xs_col = self.xs.reshape(-1, 1)   # (nx,1)
        us_col = self.us.reshape(-1, 1)   # (nu,1)

        # Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i]-self.xs), Q)
            cost += cp.quad_form((u_var[:,i]-self.us), R)
        # Terminal cost
        cost += cp.quad_form((x_var[:, -1]-self.xs), Qf)
                
        constraints = []
        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)
        # System dynamics
        constraints.append((x_var[:,1:] - xs_col) == self.A @ (x_var[:,:-1] - xs_col) + self.B @ (u_var - us_col))
        # State constraints
        constraints.append(X.A @ (x_var[:, :-1]-xs_col) <= X.b.reshape(-1, 1))
        # Input constraints
        constraints.append(U.A @ (u_var - us_col) <= U.b.reshape(-1, 1))
        # Terminal Constraints
        constraints.append(O.A @ (x_var[:, -1] - xs_col) <= O.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var     # garde une référence pour get_u
        self.x_var = x_var
        self.u_var = u_var

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
    
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj