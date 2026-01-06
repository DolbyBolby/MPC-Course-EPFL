import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def compute_steady_state(self,r:np.ndarray)-> tuple[np.ndarray,np.ndarray] : 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        d_hat = np.array(d_hat).reshape((-1,))
        r = np.array(r).reshape((-1,))
        v_ref = r[-1]
        C = np.array([[1.0]])

        xss_var = cp.Variable(self.nx, name='xs')
        uss_var = cp.Variable(self.nu, name='us')

        xs_col = self.xs.reshape(-1, 1)   # (nu,1)

        u_min = 40
        u_max = 80

        # Objective: minimize input squared
        ss_obj = cp.quad_form(uss_var, np.eye(self.nu))
        
        # Constraints: steady-state and input bounds
        ss_cons = [
            uss_var >= u_min,
            uss_var <= u_max,
            xss_var == self.A @ xss_var + self.B @ uss_var + Bd @ d_hat,
            C @ xss_var == v_ref
        ]

        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve()
        assert prob.status == cp.OPTIMAL
        # if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        #     print("Infeasible steady-state for r =", r)
        #     return None, None
        

        xss = xss_var.value
        uss = uss_var.value

        return xss,uss

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
    
        Q = 50*np.eye(self.nx)# for tuning
        R = 0.1*np.eye(self.nu)

        #constraints

        Hu = np.array([[ 1.],
                    [-1.]])
       
        U = Polyhedron.from_Hrep(Hu, np.array([80.0 - self.us[0], self.us[0] - 40.0]))

       # Define variables
        xs_col = self.xs.reshape(-1, 1)   # (nx,1)
        us_col = self.us.reshape(-1, 1)   # (nu,1)

        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))
        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        #e_var = cp.Variable((1,self.N+1))
        x0_hat_par = cp.Parameter(self.nx, name='x0_hat')     # (estimated) initial state x0
        d_hat_par = cp.Parameter(1, 'd_hat')            # (estimated) disturbance

        # Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i]-cp.reshape(x_ref, (self.nx,))), Q)
            cost += cp.quad_form((u_var[:,i]-cp.reshape(u_ref, (self.nu,))), R)
            #cost += cp.quad_form((e_var[:,i]),S) 
                
        constraints = []

        constraints.append((x_var[:, 0]) == x0_hat_par)
        # System dynamics
        constraints.append((x_var[:,1:] - xs_col) == self.A @ (x_var[:,:-1] - xs_col) + self.B @ (u_var-us_col))
        # Input constraints
        constraints.append(U.A @ (u_var-us_col) <= U.b.reshape(-1, 1))

        # all contraints

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var     # garde une référence pour get_u
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.estimate_parameters
        xss,uss = self.compute_steady_state(x_target)
        
        self.x0_var.value = x0
        self.x_ref.value = xss
        self.u_ref.value = uss
        self.ocp.solve(solver=cp.PIQP)
        assert self.ocp.status == cp.OPTIMAL
        # print("SS status:", self.ocp.status, "r:", x_target)
        # if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        #     print("Infeasible steady-state for r =", x_target)
        #     return None, None

        u0 = self.u_var.value[:, 0]
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    def estimate_parameters(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        return
    
    def compute_observer_gain(self):

        C = np.array([[1.0]])
        ny = C.shape[0]
        nd = 1

        Bd = np.array([[0], [0]])
        Cd = np.array([[1]])

        # A_hat = [A  Bd;  0  I]
        A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((self.ny, self.nx)), np.eye(ny)))
        ))

        # B_hat = [B; 0]
        B_hat = np.vstack((self.B, np.zeros((nd, self.nu))))

        # C_hat = [C  Cd]
        C_hat = np.hstack((self.C, np.ones((ny,nd))))

        poles = np.array([0.5, 0.6, 0.7])
        from scipy.signal import place_poles
        res = place_poles(A_hat.T, C_hat.T, poles)
        L = -res.gain_matrix.T
        return L 