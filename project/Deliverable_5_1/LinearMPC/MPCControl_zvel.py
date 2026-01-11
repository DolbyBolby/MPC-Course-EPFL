import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from scipy.signal import place_poles

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def compute_steady_state(self,r:np.ndarray,d_hat:np.ndarray)-> tuple[np.ndarray,np.ndarray]: 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        return self.compute_steady_state_with_disturbance(r, np.zeros(1))
    
    def compute_steady_state_with_disturbance(self, r: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        OFFSET-FREE TRACKING: Compute steady-state with disturbance compensation.
        
        System with disturbance: x[k+1] = A*x[k] + B*u[k] + Bd*d
        For pure integrator (A=I, Bd=I): v[k+1] = v[k] + B*u[k] + d
        
        At steady state: x_ss = A*x_ss + B*u_ss + Bd*d
        => (I - A)*x_ss = B*u_ss + Bd*d
        
        For integrator (A=I, Bd=I): 0 = B*u_ss + d
        => u_ss = -d/B (compensates disturbance)
        
        Target: x_ss = x_ref (we want velocity = reference)
        """
        r = np.array(r).reshape((-1,))
        v_ref = r[-1]  # Target velocity
        
        # Steady-state calculation for integrator
        # At equilibrium: 0 = B*u_ss + d  =>  u_ss = -d/B
        B_scalar = self.B[0, 0]  # Extract scalar from 1x1 matrix
        d_scalar = d[0]  # Extract scalar from array
        
        u_ss_desired = -d_scalar / B_scalar
        
        # Apply ABSOLUTE input constraints [40, 80]N
        u_min = 40.0
        u_max = 80.0
        u_ss = np.clip(u_ss_desired, u_min, u_max)
        
        # Steady-state velocity equals reference
        x_ss = np.array([v_ref])
        

        # Objective: minimize input squared
        ss_obj = cp.quad_form(duss_var, np.eye(self.nu))
        print("self.B")
        # Constraints: steady-state and input bounds
        ss_cons = [
            duss_var >= u_min - self.us,
            duss_var <= u_max - self.us,
            dxss_var == self.A @ dxss_var + self.B @ duss_var + self.B @ d_hat,
            self.C @ dxss_var == target - self.C @ self.xs.reshape(-1,1) - self.Cd @ d_hat,
        ]
        print("b")
        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve(solver=cp.PIQP)
        assert prob.status == cp.OPTIMAL
        xss = dxss_var.value + self.xs
        uss = duss_var.value + self.us
        print("end steady state")
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

        #x_hat = cp.Parameter(self.nx, name='x0_hat')     # (estimated) initial state x0
        #d_hat = cp.Parameter(1, 'd_hat') 
        
        
        Q = 50*np.eye(self.nx)# for tuning
        R = 0.1*np.eye(self.nu)

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
        constraints.append((x_var[:, 0]) == x0_var - x_ref_col)
        # System dynamics
        constraints.append((x_var[:,1:] - x_ref_col) == self.A @ (x_var[:,:-1] - x_ref_col) + self.B @ (u_var - u_ref_col))
        # Input constraints
        constraints.append(U.A @ (u_var - u_ref_col) <= U.b.reshape(-1, 1) - U.A @ u_ref_col)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var
        self.x_var = x_var
        self.u_var = u_var
        self.x_ref = x_ref
        self.u_ref = u_ref
        #self.x_hat = x0_var
        #print("x_hat setup",self.x_hat.value)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        if not hasattr(self, 'd_hat'):
            # First call: initialize observer
            self.d_hat = np.zeros(1)
            self.x_prev = x0.copy()
            self.u_prev = np.zeros(self.nu)
        else:
            # Predict state using NOMINAL model (no disturbance)
            # x_pred[k] = A*x[k-1] + B*u[k-1]
            x_pred = self.A @ self.x_prev + self.B @ self.u_prev
            
            # Prediction error (actual - predicted)
            # e[k] = x_measured[k] - x_pred[k]
            # For system with constant disturbance: e ≈ d
            e = x0 - x_pred
            
            # Update disturbance estimate with integral action
            # d_hat[k] = d_hat[k-1] + L_d * e[k]
            gain_d = 0.3  # Observer gain (faster convergence)
            self.d_hat = self.d_hat + gain_d * e
            
            # Saturate estimate (physical limits)
            # Gravity: d_real ≈ -g*Ts = -0.49 m/s/step
            # Allow margin just above 0.49 to prevent saturation at equilibrium
            d_max = 0.55  # Bounds: -0.55 to +0.55 (includes -0.49 with margin)
            self.d_hat = np.clip(self.d_hat, -d_max, d_max)
        
        # ========== STEP 2: TARGET CALCULATION ==========
        # Compute steady-state that rejects disturbance
        x_ss, u_ss = self.compute_steady_state_with_disturbance(x_target, self.d_hat)
        
        # ========== STEP 3: SOLVE MPC IN DEVIATION FORM ==========
        # Update parameters
        self.x0_var.value = x0
        self.x_ref.value = x_ss
        self.u_ref.value = u_ss
        
        # Update input constraint bounds for δu
        # u ∈ [40, 80] with u = u_ss + δu
        # => δu ∈ [40 - u_ss, 80 - u_ss]
        u_min, u_max = 40.0, 80.0
        du_min = u_min - u_ss[0]
        du_max = u_max - u_ss[0]
        
        
        
        self.du_lb.value = np.full((self.nu, self.N), du_min)
        self.du_ub.value = np.full((self.nu, self.N), du_max)
        
        # Initialize debug counter if needed
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        
        # Solve MPC (disable warm_start to avoid stale solutions)
        try:
            self.ocp.solve(solver=cp.PIQP, warm_start=False, verbose=False)
        except Exception as e:
            self.ocp.status = "error"
        
        # ========== STEP 4: EXTRACT SOLUTION ==========
        # DEBUG: Print solver status periodically
        
        if self._debug_counter % 50 == 1:  # Print every 50 iterations
            if self.ocp.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                du0_debug = self.du_var.value[:, 0]
                u0_debug = u_ss + du0_debug
        
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
           
            # Fallback: use steady-state input
            u0 = u_ss.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N+1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
        else:
            # Convert deviation variables back to absolute
            # u = u_ss + δu
            du0 = self.du_var.value[:, 0]
            u0 = u_ss + du0
            
            self._last_du0 = du0[0]
            
            # x_traj = x_ss + δx_traj
            dx_traj = self.dx_var.value
            x_traj = x_ss.reshape(-1, 1) + dx_traj
            
            # u_traj = u_ss + δu_traj  
            du_traj = self.du_var.value
            u_traj = u_ss.reshape(-1, 1) + du_traj
        
        # Store for next iteration (observer needs history)
        self.x_prev = x0.copy()
        self.u_prev = u0.copy()

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
        print("end")
    
    def compute_observer_gain(self)-> tuple[np.ndarray]:

        self.C = np.array([[1]])
        self.Cd = np.array([[1]])

        # print("Cd",self.Cd)
        # print("A",self.A)
        # print("B",self.B)

        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, self.B)),
            np.hstack((0, 1))
        ))
        #print("A_hat",self.A_hat)
        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((1, self.nu))))
        #print("B_hat",self.B_hat)
        # C_hat = [C  Cd]
        self.C_hat = np.hstack((self.C, self.Cd))
        #print("C_hat",self.C_hat)

        poles = np.array([0.5, 0.6])
        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        L = -res.gain_matrix.T
        print("L",L)

    def compute_observer_gain(self) -> np.ndarray:
        """
        Compute observer gain for offset-free disturbance estimation.
        Uses direct gain specification instead of pole placement for robustness.
        
        For constant disturbance estimation:
        - Fast adaptation: observer responds quickly to disturbance changes
        - Robustness: not too aggressive to avoid noise amplification
        """
        nd = 1  # Disturbance dimension
        l_x = 0.1  # State error gain (10% correction per step)
        l_d = 0.3  # Disturbance error gain (30% correction per step)
        
        L = np.array([[l_x], [l_d]])
        
        return L