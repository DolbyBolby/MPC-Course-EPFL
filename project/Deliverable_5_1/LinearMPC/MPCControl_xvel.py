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
        target = r[-1]
        C = np.array([[0, 0, 1]])

        dxss_var = cp.Variable(self.nx, name='xss')
        duss_var = cp.Variable(self.nu, name='uss')

        u_min = -0.26
        u_max = 0.26

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
        xss = dxss_var.value + self.xs
        uss = duss_var.value + self.us
        
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
        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))
        e_var = cp.Variable((1,self.N+1))

        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        x_ref.value = self.xs
        u_ref.value = self.us
        x_ref_col = x_ref.value.reshape(-1, 1)   # (nx,1)
        u_ref_col = u_ref.value.reshape(-1, 1)   # (nu,1)

        Q = np.diag([1.0, 2000.0, 20.0]) # for tuning
        R = 1*np.eye(self.nu)
        S = 0.1*np.eye(1)

        #constraints
        Hx = np.array([[0., 1., 0.],
               [0.,-1., 0.]])
        kx = np.array([0.1745, 0.1745])

        Hu = np.array([[ 1.],
                    [-1.]])
        ku = np.array([0.26,0.26])

        X = Polyhedron.from_Hrep(Hx, kx)
        U = Polyhedron.from_Hrep(Hu, ku)  

        # Terminal weight Qf and terminal controller K
        K,Qf,_ = dlqr(self.A,self.B,Q,R)
        K = -K
        A_cl = self.A + self.B @ K

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
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i] - x_ref), Q)
            cost += cp.quad_form((u_var[:,i] - u_ref), R)
            cost += cp.quad_form((e_var[:,i]),S)
        # Terminal cost
        cost += cp.quad_form((x_var[:, -1] - x_ref), Qf)

        constraints = []
        # Initial condition
        constraints.append(x_var[:, 0] == x0_var)
        # System dynamics
        constraints.append((x_var[:,1:] - x_ref_col) == self.A @ (x_var[:,:-1] - x_ref_col) + self.B @ (u_var - u_ref_col))
        # State constraints
        constraints.append(X.A @ (x_var[:, :-1] - x_ref_col) <= X.b.reshape(-1, 1)- X.A @ x_ref_col + e_var[:,:-1])
        # Input constraints
        constraints.append(U.A @ (u_var - u_ref_col) <= U.b.reshape(-1, 1) - U.A @ u_ref_col)
        # Terminal Constraints
        constraints.append(O.A @ (x_var[:, -1] - x_ref_col) <= O.b.reshape(-1, 1))
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
        #################################################
        # YOUR CODE HERE
        xss,uss = self.compute_steady_state(x_target)
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
    
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    