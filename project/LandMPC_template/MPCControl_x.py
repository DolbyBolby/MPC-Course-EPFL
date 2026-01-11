import numpy as np
from mpt4py import Polyhedron
import cvxpy as cp
from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        self.x_var = cp.Variable((self.nx, self.N + 1))      
        self.u_var = cp.Variable((self.nu, self.N))        
        self.x0_hat = cp.Parameter((self.nx,), name='x0')      
        self.x_ref  = cp.Parameter((self.nx,), name='xref')   
        self.u_ref  = cp.Parameter((self.nu,), name='uref')    
        # **X-CONTROLLER WEIGHTS** (HIGH on x_ids = [1,4,6,9] = ωy,α,vx,px)
        Q = np.diag([20, 500, 10, 30]) #* 10   # [ωy, α, vx, px]
        R = np.array([[0.08]])                    # δ1 (u_ids=[1])
        
        
        
        # **δ1 constraint** (u_ids=[1])
        M = np.array([[-1.], [1.]])
        m = np.array([np.deg2rad(15) + self.us[0], np.deg2rad(15) - self.us[0]])
        U = Polyhedron.from_Hrep(M, m)

        F = np.array([[0,-1.,0,0], [0,1.,0,0]])
        f = np.array([np.deg2rad(10) + self.xs[1], np.deg2rad(10) - self.xs[1]])
        X = Polyhedron.from_Hrep(F, f)
        # Slack variables
        self.slack_var = cp.Variable((2, self.N), name="slack")
        Slack = 1e6 * np.eye(2)
                
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(self.x_var[:,i] - self.x_ref, Q)
            cost += cp.quad_form(self.u_var[:,i] - self.u_ref, R)
            cost += cp.quad_form(self.slack_var[:,i], Slack)
        
        constraints = []
        constraints.append(self.x_var[:,0] == self.x0_hat)
        constraints.append(self.x_var[:,1:] == self.A @ self.x_var[:,:-1] + self.B @ self.u_var)
        
        constraints.append(U.A @ self.u_var <= U.b.reshape(-1, 1))
        constraints.append(X.A @ self.x_var[:,:-1] <= X.b.reshape(-1,1) + self.slack_var)
        


        # Slack (keep for cost)
        constraints.append(self.slack_var >= 0)
        
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # Delta coordinates (error from trim point)
        delta_x0 = x0 - self.xs
        self.x0_hat.value = delta_x0
        self.x_ref.value = np.zeros(self.nx)        # Track trim point xs
        self.u_ref.value = np.zeros(self.nu)
        
        self.ocp.solve(solver=cp.PIQP)
        if self.ocp.status != cp.OPTIMAL:
            print(f"MPC x status: {self.ocp.status}")
            return self.us, np.zeros((self.nx,self.N+1)), np.zeros((self.nu,self.N))
        
        u0 = self.u_var.value[:, 0] + self.us
        x_traj = self.x_var.value + self.xs.reshape(-1, 1)
        u_traj = self.u_var.value + self.us.reshape(-1, 1)
    
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj