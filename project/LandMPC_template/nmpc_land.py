import numpy as np
import casadi as ca
from typing import Tuple

class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """



    def __init__(self, rocket, H,xs, us, Ts):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]
        self.Ts = Ts
        self.xs = xs
        self.us = us
        self.N = int (H/Ts)
        self.H = H
        self.nx, self.nu= xs.size, us.size

        
        # linear model around landing 

        #A, B = rocket.linearize(xs, us)   # continuous

        #define controler 
        # Q_lqr = np.diag([
        #     1, 1, 1,      # angular rates ωx,ωy,ωz
        #     5, 5, 5,      # angles α,β,γ
        #     2, 2, 5,      # velocities vx,vy,vz (vertical more important)
        #     20, 20, 30    # positions x,y,z (z strongest)
        # ])

        # R_lqr = np.diag([
        #     5, 5,   # δ1, δ2
        #     0.5,    # Pavg
        #     0.5     # Pdiff
        # ])
        # Q_lqr = np.diag([
        #     50*100,  50*100,  20*10,      # ωx,ωy,ωz  ← Attitude rates FIRST (highest)
        #     100, 100, 50,      # α,β,γ     ← Attitude angles (next)
        #     10*100,  10*100,  20*10,      # vx,vy,vz  ← Velocities  
        #     20*10,  20*10,  50       # x,y,z     ← Positions LAST (z higher)
        # ])

        # R_lqr = np.diag([
        #     0.5, 0.5,          # δ1,δ2     ← LOW (aggressive attitude control)
        #     0.1,  0.2          # Pavg,Pdif ← LOW Pdif for roll
        # ])
        # self.P = 50 * self.Q   # terminal cost;

        Q_lqr = np.diag([
            240*0.02, 240*0.5, 120*0.7,     # ωx,ωy,ωz  ← 4x higher → FAST attitude
            360, 360, 180*1.2,     # α,β,γ 
            60*1.2, 60*2, 120,     # vx,vy,vz  
            120, 120*3, 240      # x,y,z     ← positions still last
        ])

        R_lqr = np.diag([
            0.08, 0.08,          # δ1,δ2 ← **10x lower** = aggressive tilting!
            0.04, 0.08          # Pavg,Pdif
        ])

        
                                    
        self.Q = Q_lqr
        self.R = R_lqr
        self.P = 8 * self.Q  # **REDUCE terminal weight** = less conservative

        self._setup_controller()
            

    def _setup_controller(self) -> None:

        
        opti = ca.Opti()
        
        # decision variables
        self.X = opti.variable(self.nx, self.N + 1)
        self.U = opti.variable(self.nu, self.N)

        opti.set_initial(self.X, np.tile(self.xs.reshape(-1, 1), (1, self.N + 1)))
        opti.set_initial(self.U, np.tile(self.us.reshape(-1, 1), (1, self.N)))

        # parameters
        print ('nx', self.nx)
        print ('nu', self.nu)
        x0_par = opti.parameter(self.nx)
        xs_par = opti.parameter(self.nx)
        us_par = opti.parameter(self.nu)
        Q_par  = opti.parameter(self.nx, self.nx)
        R_par  = opti.parameter(self.nu, self.nu)
        P_par  = opti.parameter(self.nx, self.nx)

        cost = 0
        
        # Initial condition
        opti.subject_to(self.X[:, 0] == x0_par)

        for k in range(self.N):

            xk = self.X[:, k]
            uk = self.U[:, k]

            # Dynamics constraint
            x_next = self.rk4(xk, uk)
            opti.subject_to(self.X[:, k+1] == x_next)

            # Cost: (x_k - xs)^T Q (x_k - xs) + (u_k - us)^T R (u_k - us) 
            dx = xk - xs_par
            du = uk - us_par
            cost += ca.mtimes([dx.T, Q_par, dx]) + ca.mtimes([du.T, R_par, du])

            # Input constraints
            delta1 = uk[0]
            delta2 = uk[1]
            Pavg   = uk[2]
            Pdiff  = uk[3]
            opti.subject_to(ca.fabs(delta1) <= np.deg2rad(15))
            opti.subject_to(ca.fabs(delta2) <= np.deg2rad(15))
            opti.subject_to(Pavg >= 10)
            opti.subject_to(Pavg <= 90)
            opti.subject_to(ca.fabs(Pdiff) <= 20)

        # Terminal cost (x_N - xs)^T P (x_N - xs)
        dxN = self.X[:, self.N] - xs_par
        cost += ca.mtimes([dxN.T, P_par, dxN])

        # State constraints (applied along horizon)
        beta_index = 4    # depending on state ordering: [ω, φ= (α,β,γ), v, p]
        z_index    = 11
        beta = self.X[beta_index, :]
        z    = self.X[z_index, :]
        opti.subject_to(ca.fabs(beta) <= np.deg2rad(80))
        opti.subject_to(z >= 0)

        opti.minimize(cost)

        # Solver options
        opts = {'expand': True,
                'ipopt': {'print_level': 0, 'tol': 1e-3}}
        opti.solver('ipopt', opts)
        

        # store everything needed later
        self.ocp = opti
        self.x0_par = x0_par
        self.xs_par = xs_par
        self.us_par = us_par
        self.Q_par = Q_par
        self.R_par = R_par
        self.P_par = P_par


    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        opti = self.ocp
        # set parameters
        opti.set_value(self.x0_par, x0)
        opti.set_value(self.xs_par, self.xs)
        opti.set_value(self.us_par, self.us)
        opti.set_value(self.Q_par, self.Q)
        opti.set_value(self.R_par, self.R)
        opti.set_value(self.P_par, self.P)
        print ('xo',x0)
        print("Target xs:", self.xs[9:12])  # should be ~[1,0,3]
        print("Target us:", self.us)        # Pavg should be lower than hover (~60%)
        sol = opti.solve()

        
        X_opt = sol.value(self.X)
        U_opt = sol.value(self.U)

        u0 = U_opt[:, 0]
        x_ol = X_opt
        u_ol = U_opt
        t_ol = t0 + np.arange(self.N + 1) * self.Ts

        return u0, x_ol, u_ol, t_ol
    

    def rk4 (self, x,u):
        k1 = self.H * self.f(x, u)
        k2 = self.H * self.f(x + k1 / 2, u)
        k3 = self.H * self.f(x + k2 / 2, u)
        k4 = self.H * self.f(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6