import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def compute_steady_state(self,r:np.ndarray)-> tuple[np.ndarray,np.ndarray]: 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        target = r[-1]
        C = np.array([[0, 1]])

        dxss_var = cp.Variable(self.nx, name='xss')
        duss_var = cp.Variable(self.nu, name='uss')

        u_min = -20
        u_max = 20

        # Objective: minimize input squared
        ss_obj = cp.quad_form(duss_var, np.eye(self.nu))
        
        # Constraints: steady-state and input bounds
        ss_cons = [
            duss_var >= u_min - self.us,
            duss_var <= u_max - self.us,
            dxss_var == self.A @ dxss_var + self.B @ duss_var,
            C @ dxss_var == target - C@self.xs.reshape(-1,1),
        ]

        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve()
        assert prob.status == cp.OPTIMAL
        #print("SS status:", self. ocp.status, "r:", target)
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
        
        Q = 1*np.eye(self.nx)# for tuning
        R = 1*np.eye(self.nu)

        # Terminal weight Qf and terminal controller K
        K,Qf,_ = dlqr(self.A,self.B,Q,R)
        K = -K

        A_cl = self.A + self.B @ K

        #constraints
        Hu = np.array([[ 1.],
                    [-1.]])
       
        U = Polyhedron.from_Hrep(Hu, np.array([20.0,20.0]))
       
        # maximum inavariant set for recusive feasability
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O = KU
        
        max_iter = 30
        for iter in range(max_iter): 
            Oprev = O
            F,f = O.A,O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            if O == Oprev:
                break
        
        # Costs
        cost = 0
        S = 0.1*np.eye(1)  # Penalty on slack variables
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i]-x_ref), Q)
            cost += cp.quad_form((u_var[:,i]-u_ref), R)
        # Terminal cost
        cost += cp.quad_form((x_var[:, -1]-x_ref), Qf)
                
        constraints = []
        constraints.append((x_var[:, 0]) == x0_var)
        # System dynamics
        constraints.append((x_var[:,1:] - x_ref_col) == self.A @ (x_var[:,:-1] - x_ref_col) + self.B @ (u_var-u_ref_col))
        # Input constraints
        constraints.append(U.A @ (u_var-u_ref_col) <= U.b.reshape(-1, 1) - U.A @ u_ref_col)
        # Terminal Constraints
        constraints.append(O.A @ (x_var[:, -1]-x_ref_col) <= O.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var
        self.x0_var = x0_var
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute control input with offset-free observer.
        """
        # Initialize observer on first call
        if not hasattr(self, 'd_hat'):
            self.d_hat = np.zeros(self.nx)
            self.x_prev = x0.copy()
            self.u_prev = np.zeros(self.nu)
        else:
            # Estimate disturbance using prediction error
            x_pred = self.A @ self.x_prev + self.B @ self.u_prev
            prediction_error = x0 - x_pred
            # Update disturbance estimate with integral action
            gain_d = 0.05  # Disturbance estimation gain (réduit pour stabilité)
            self.d_hat = self.d_hat + gain_d * prediction_error
            # Saturer l'estimation pour éviter l'infaisabilité
            d_max = 0.3
            self.d_hat = np.clip(self.d_hat, -d_max, d_max)
        
        # Solve MPC with disturbance model
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.PIQP, warm_start=True)
        
        # Fallback if solver fails
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: MPCControl_roll solver status = {self.ocp.status}")
            # Try again
            self.ocp.solve(solver=cp.PIQP, warm_start=False)
            if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Last resort: use previous solution or zero input
                print(f"Warning: MPCControl_roll both solvers failed - using fallback control")
                if hasattr(self, 'u_prev') and self.u_prev is not None:
                    u0 = self.u_prev.copy()
                else:
                    u0 = self.us.copy()
                x_traj = np.tile(x0.reshape(-1, 1), (1, self.N+1))
                u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
                self.x_prev = x0.copy()
                self.u_prev = u0.copy()
                return u0, x_traj, u_traj

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        
        # Store for next iteration
        self.x_prev = x0.copy()
        self.u_prev = u0.copy()

        return u0, x_traj, u_traj
    
    def estimate_parameters(self, x_k: np.ndarray, x_km1: np.ndarray, u_km1: np.ndarray) -> None:
        """
        Update estimated state and disturbance using Luenberger observer.
        x_k: current measurement
        x_km1: previous measurement
        u_km1: previous input
        """
        C = np.array([[1.0, 0]])
        Cd = np.array([[0.0]])  # disturbance not directly measured
        
        # Augmented state z = [x; d]
        z_hat_k = np.concatenate([self.x_hat, self.d_hat])
        
        # Measurement
        y_k = C @ x_k
        y_pred_k = C @ self.x_hat + Cd @ self.d_hat
        
        # Augmented system matrices
        nd = 1
        A_hat = np.vstack((
            np.hstack((self.A, np.zeros((self.nx, nd)))),
            np.hstack((np.zeros((nd, self.nx)), np.eye(nd)))
        ))
        B_hat = np.vstack((self.B, np.zeros((nd, self.nu))))
        
        # Observer update
        z_hat_next = A_hat @ z_hat_k + B_hat @ u_km1 + self.L @ (y_k - y_pred_k)
        
        self.x_hat = z_hat_next[:self.nx]
        self.d_hat = z_hat_next[self.nx:]
    
    def compute_observer_gain(self):
        C = np.array([[1.0, 0]])
        ny = C.shape[0]
        nd = 1

        Bd = np.array([[0], [0]])
        Cd = np.array([[1]])

        # A_hat = [A  Bd;  0  I]
        A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((ny, self.nx)), np.eye(ny)))
        ))

        # B_hat = [B; 0]
        B_hat = np.vstack((self.B, np.zeros((nd, self.nu))))

        # C_hat = [C  Cd]
        C_hat = np.hstack((C, np.ones((ny,nd))))

        # Use direct gain specification for robustness (avoids pole placement issues)
        # L = [l_roll; l_pitch; l_d] where:
        #   - l_roll, l_pitch: state estimation gains (small)
        #   - l_d: disturbance estimation gain (moderate)
        L = np.array([
            [0.05],   # roll angle correction
            [0.05],   # roll rate correction
            [0.2]     # disturbance correction
        ])
        return L