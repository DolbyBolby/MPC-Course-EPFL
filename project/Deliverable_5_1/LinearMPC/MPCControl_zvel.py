import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from scipy.signal import place_poles

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def compute_steady_state(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        
        # Warning if saturation occurs (disturbance too large)
        if abs(u_ss - u_ss_desired) > 1e-6:
            print(f"⚠️  Z-controller: d={d_scalar:.4f} requires u={u_ss_desired:.2f}N, "
                  f"saturated to [{u_min}, {u_max}] → u_ss={u_ss:.2f}N")

        return x_ss, np.array([u_ss])

    def _setup_controller(self) -> None:
        Q = 1000 * np.eye(self.nx)  # State tracking weight (priorité maximale)
        R = 0.001 * np.eye(self.nu)  # Input variation weight (quasi-nul → suit u_ss)

        # Input constraints: ABSOLUTE bounds 40 ≤ u ≤ 80
        u_min = 40.0
        u_max = 80.0

        # DEVIATION VARIABLES (key for offset-free tracking)
        dx_var = cp.Variable((self.nx, self.N + 1))  # δx = x - x_ss
        du_var = cp.Variable((self.nu, self.N))       # δu = u - u_ss
        
        # Parameters
        x0_var = cp.Parameter((self.nx,))     # Current state (absolute)
        x_ref = cp.Parameter((self.nx,))      # Target state x_ss (absolute)
        u_ref = cp.Parameter((self.nu,))      # Target input u_ss (absolute)

        # Cost: minimize deviations from steady-state
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(dx_var[:, i], Q)
            cost += cp.quad_form(du_var[:, i], R)
        # Terminal cost (augmenté pour forcer convergence)
        cost += 50.0 * cp.quad_form(dx_var[:, self.N], Q)

        # Constraints
        constraints = []
        
        # Initial condition: δx[0] = x0 - x_ss
        constraints.append(dx_var[:, 0] == (x0_var - cp.reshape(x_ref, (self.nx,))))
        
        # Dynamics in DEVIATION form: δx[k+1] = A*δx[k] + B*δu[k]
        # (This is exact for linear systems with constant disturbance)
        constraints.append(
            dx_var[:, 1:] == self.A @ dx_var[:, :-1] + self.B @ du_var
        )
        
        # Input constraints in ABSOLUTE form: u_min ≤ u_ss + δu ≤ u_max
        # Rewrite as: (u_min - u_ss) ≤ δu ≤ (u_max - u_ss)
        # These bounds are updated at each solve based on current u_ss
        self.du_lb = cp.Parameter((self.nu, self.N))  # Lower bound for δu
        self.du_ub = cp.Parameter((self.nu, self.N))  # Upper bound for δu
        constraints.append(du_var >= self.du_lb)
        constraints.append(du_var <= self.du_ub)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x0_var = x0_var
        self.dx_var = dx_var
        self.du_var = du_var
        self.x_ref = x_ref
        self.u_ref = u_ref

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OFFSET-FREE MPC: Compute control with disturbance estimation.
        
        Algorithm:
        1. Estimate constant disturbance d from prediction error
        2. Compute target steady-state (x_ss, u_ss) that compensates d
        3. Solve MPC with nominal dynamics, cost relative to (x_ss, u_ss)
        
        Key: x0 is MEASUREMENT, observer estimates d_hat
        """
        # ========== STEP 1: DISTURBANCE OBSERVER ==========
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
        
        # DEBUG: Verify bounds make sense
        if not hasattr(self, '_bounds_printed'):
            print(f"\n🔍 First MPC call - Constraint check:")
            print(f"   u_ss = {u_ss[0]:.2f}N")
            print(f"   δu bounds: [{du_min:.2f}, {du_max:.2f}]")
            print(f"   Implies u ∈ [{u_ss[0]+du_min:.2f}, {u_ss[0]+du_max:.2f}]N")
            self._bounds_printed = True
        
        self.du_lb.value = np.full((self.nu, self.N), du_min)
        self.du_ub.value = np.full((self.nu, self.N), du_max)
        
        # Initialize debug counter if needed
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        # DEBUG: Verify parameters are set correctly
        if self._debug_counter % 50 == 1:
            print(f"   x0_var = {self.x0_var.value}")
            print(f"   x_ref = {self.x_ref.value}")
            print(f"   du_lb[0,0] = {self.du_lb.value[0,0]:.2f}, du_ub[0,0] = {self.du_ub.value[0,0]:.2f}")
        
        # Solve MPC (disable warm_start to avoid stale solutions)
        try:
            self.ocp.solve(solver=cp.PIQP, warm_start=False, verbose=False)
        except Exception as e:
            print(f"❌ Z-MPC solver EXCEPTION: {e}")
            self.ocp.status = "error"
        
        # ========== STEP 4: EXTRACT SOLUTION ==========
        # DEBUG: Print solver status periodically
        
        if self._debug_counter % 50 == 1:  # Print every 50 iterations
            print(f"\n🔧 Z-MPC Debug (iter {self._debug_counter}):")
            print(f"   INPUT: x0={x0[0]:+.4f}, target={x_target[0] if x_target is not None else 0.0:+.4f}")
            print(f"   Solver status: {self.ocp.status}")
            print(f"   d_hat={self.d_hat[0]:+.4f}, u_ss={u_ss[0]:+.2f}N")
            print(f"   du_bounds=[{du_min:+.2f}, {du_max:+.2f}]")
            if self.ocp.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                du0_debug = self.du_var.value[:, 0]
                u0_debug = u_ss + du0_debug
                print(f"   SOLUTION: δu[0]={du0_debug[0]:+.4f}, u[0]={u0_debug[0]:+.2f}N")
        
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"❌ Z-MPC solver failed: {self.ocp.status}")
            print(f"   d_hat={self.d_hat[0]:.4f}, x_ss={x_ss[0]:.4f}, u_ss={u_ss[0]:.2f}")
            # Fallback: use steady-state input
            u0 = u_ss.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N+1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
        else:
            # Convert deviation variables back to absolute
            # u = u_ss + δu
            du0 = self.du_var.value[:, 0]
            u0 = u_ss + du0
            
            # DEBUG: Check if du_var is actually changing
            if hasattr(self, '_last_du0'):
                if np.abs(du0[0] - self._last_du0) < 1e-6 and self._debug_counter > 10:
                    print(f"⚠️  WARNING: δu[0] hasn't changed for multiple iterations!")
                    print(f"   δu[0] = {du0[0]:.4f}, u_ss = {u_ss[0]:.2f}, u[0] = {u0[0]:.2f}")
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

    def estimate_parameters(self, x_k: np.ndarray, x_km1: np.ndarray, u_km1: np.ndarray) -> None:
        """
        Update estimated state and disturbance using Luenberger observer.
        x_k: current measurement
        x_km1: previous measurement
        u_km1: previous input
        """
        C = np.array([[1.0]])
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

    def compute_observer_gain(self) -> np.ndarray:
        """
        Compute observer gain for offset-free disturbance estimation.
        Uses direct gain specification instead of pole placement for robustness.
        
        For constant disturbance estimation:
        - Fast adaptation: observer responds quickly to disturbance changes
        - Robustness: not too aggressive to avoid noise amplification
        """
        nd = 1
        
        # For a simple 1D system with constant disturbance:
        # The observer gain directly determines how fast the disturbance is estimated
        # L = [l_x; l_d] where:
        #   - l_x: gain for state estimation (typically small)
        #   - l_d: gain for disturbance estimation (typically moderate)
        
        # Classical approach: use Kalman-like gains
        # For discrete-time offset-free MPC with constant disturbance
        l_x = 0.1  # State error gain (10% correction per step)
        l_d = 0.3  # Disturbance error gain (30% correction per step)
        
        L = np.array([[l_x], [l_d]])
        
        return L