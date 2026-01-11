import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def compute_steady_state(self,r:np.ndarray)-> tuple[np.ndarray,np.ndarray]: 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        return self.compute_steady_state_with_disturbance(r, np.zeros(self.nx))
    
    def compute_steady_state_with_disturbance(self, r: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute steady-state with disturbance compensation.
        At steady state: xss = A*xss + B*uss + d
        => (I - A)*xss = B*uss + d
        
        We solve for absolute values xss and uss that satisfy:
        - The steady-state equation with disturbance
        - The output constraint C*xss = r (target velocity)
        - Input bounds
        """
        r = np.array(r).reshape((-1,))
        v_ref = r[-1]  # Target velocity (last element of state is velocity)
        C = np.array([[0, 0, 1]])  # Output matrix (selects velocity)

        xss_var = cp.Variable(self.nx, name='xss')
        uss_var = cp.Variable(self.nu, name='uss')

        # Absolute input bounds
        u_min = -0.26
        u_max =  0.26

        # Objective: minimize input effort
        ss_obj = cp.quad_form(uss_var - self.us, np.eye(self.nu))
        
        # Constraints: steady-state WITH disturbance
        # (I - A)*xss = B*uss + d
        I_minus_A = np.eye(self.nx) - self.A
        ss_cons = [
            uss_var >= u_min,
            uss_var <= u_max,
            I_minus_A @ xss_var == self.B @ uss_var + d.reshape(-1),
            C @ xss_var == v_ref,
        ]

        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve()
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Fallback: try without disturbance
            ss_cons_fallback = [
                uss_var >= u_min,
                uss_var <= u_max,
                I_minus_A @ xss_var == self.B @ uss_var,
                C @ xss_var == v_ref,
            ]
            prob_fallback = cp.Problem(cp.Minimize(ss_obj), ss_cons_fallback)
            prob_fallback.solve()
            
            if prob_fallback.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and xss_var.value is not None:
                xss = xss_var.value
                uss = uss_var.value
            else:
                # Last resort: use nominal
                xss = self.xs.copy()
                uss = self.us.copy()
        else:
            if xss_var.value is not None and uss_var.value is not None:
                xss = xss_var.value
                uss = uss_var.value
            else:
                xss = self.xs.copy()
                uss = self.us.copy()
        
        return xss, uss

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

       # Define variables
        Q = np.diag([1.0, 2000.0, 20.0])# for tuning
        R = 1*np.eye(self.nu)
        S = 0.001*np.eye(1)  # Poids slack très faible pour permettre les violations temporaires

        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))
        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        x_ref.value = self.xs
        u_ref.value = self.us
        e_var = cp.Variable((1,self.N+1))

        xs_col = self.xs.reshape(-1, 1)   # (nx,1)
        us_col = self.us.reshape(-1, 1)   # (nu,1)

        #constraints
        Hx = np.array([[0., 1., 0.],
               [0.,-1., 0.]])
        kx = np.array([0.1745, 0.1745])

        Hu = np.array([[ 1.],
                    [-1.]])
        ku = np.array([0.26,0.26])

        X = Polyhedron.from_Hrep(Hx, kx - (Hx @ self.xs))
        U = Polyhedron.from_Hrep(Hu, ku - (Hu @ self.us))  
        

        # Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i] - cp.reshape(x_ref, (self.nx,))), Q)
            cost += cp.quad_form((u_var[:,i] - cp.reshape(u_ref, (self.nu,))), R)
            cost += cp.quad_form((e_var[:,i]),S)

        constraints = []

        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)
        # System dynamics in ABSOLUTE coordinates (nominal model without d)
        # x[k+1] = A*x[k] + B*u[k]
        # The disturbance compensation is implicit through the reference (xss, uss)
        constraints.append(
            x_var[:,1:] == self.A @ x_var[:,:-1] + self.B @ u_var
        )
        # State constraints (in deviation from nominal xs for constraint formulation)
        constraints.append(X.A @ (x_var[:, :-1]-xs_col) <= X.b.reshape(-1, 1) + e_var[:,:-1])
        # Input constraints (applied in deviation coordinates)
        constraints.append(U.A @ (u_var - us_col) <= U.b.reshape(-1, 1))
        # Slack variable constraints
        constraints.append(e_var >= 0)

        # Store problem and variables
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.e_var = e_var

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
            # Update disturbance estimate with integral action (gain réduit)
            gain_d = 0.05  # Disturbance estimation gain (réduit pour stabilité)
            self.d_hat = self.d_hat + gain_d * prediction_error
            # Saturer l'estimation pour éviter l'infaisabilité
            d_max = 0.5  # Limite de perturbation
            self.d_hat = np.clip(self.d_hat, -d_max, d_max)
        
        # Compute steady state ACCOUNTING for disturbance
        xss, uss = self.compute_steady_state_with_disturbance(x_target, self.d_hat)
        
        # Solve MPC with disturbance model
        self.x0_var.value = x0
        self.x_ref.value = xss
        self.u_ref.value = uss
        self.ocp.solve(solver=cp.PIQP, warm_start=True)
        
        # Fallback if solver fails
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Try with nominal steady-state (no disturbance)
            self.x_ref.value = self.xs
            self.u_ref.value = self.us
            self.ocp.solve(solver=cp.PIQP, warm_start=True)
            if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Last resort: use previous solution or zero input
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
        C = np.array([[0, 0, 1]])
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
        C = np.array([[0, 0, 1]])
        ny = C.shape[0]
        nd = 1

        Bd = np.array([[0], [0], [0]])
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

        # Use direct gain specification for robustness
        # L has size (nx+nd, ny) = (4, 1)
        L = np.array([
            [0.05],   # x velocity correction
            [0.05],   # x acceleration correction  
            [0.05],   # x jerk correction
            [0.2]     # disturbance correction
        ])
        return L