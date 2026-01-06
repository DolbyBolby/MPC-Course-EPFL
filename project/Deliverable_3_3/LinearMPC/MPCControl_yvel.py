import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr
import matplotlib.pyplot as plt

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def compute_steady_state(self,r:np.ndarray)-> tuple[np.ndarray,np.ndarray]: 
        """
        Compute the steady-state state xs and input us that minimize us^2,
        subject to the system steady-state equations and input constraints.
        """
        r = np.array(r).reshape((-1,))
        v_ref = r[-1]
        C = np.array([[0, 0, 1]])

        dxss_var = cp.Variable(self.nx, name='xs')
        duss_var = cp.Variable(self.nu, name='us')

        xs_col = self.xs.reshape(-1, 1)   # (nu,1)

        u_min = -0.26 - self.us
        u_max =  0.26 - self.us

        # Objective: minimize input squared
        ss_obj = cp.quad_form(duss_var, np.eye(self.nu))
        
        # Constraints: steady-state and input bounds
        ss_cons = [
            duss_var >= u_min,
            duss_var <= u_max,
            dxss_var == self.A @ dxss_var + self.B @ duss_var,
            C @ dxss_var == v_ref - C@xs_col,
        ]

        prob = cp.Problem(cp.Minimize(ss_obj), ss_cons)
        prob.solve()
        assert prob.status == cp.OPTIMAL
        # print("SS status:", prob.status, "duss:", duss_var.value )
        # if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        #     print("Infeasible steady-state for duss =",duss_var.value)

        xss = dxss_var.value + self.xs
        uss = duss_var.value + self.us
        
        return xss,uss



    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE


        Q = np.diag([1.0, 2000.0, 20.0])# for tuning
        R = 1*np.eye(self.nu)


        # Terminal weight Qf and terminal controller K
        K,Qf,_ = dlqr(self.A,self.B,Q,R)
        K = -K

        A_cl = self.A + self.B @ K

        # x_ref = cp.Parameter((self.nx,))
        # u_ref = cp.Parameter((self.nu,))
        # x_ref.value = self.xs
        # u_ref.value = self.us
        x_ref = cp.Parameter((self.nx,))
        u_ref = cp.Parameter((self.nu,))
        x_ref.value = self.xs
        u_ref.value = self.us


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

        # Create a figure
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #O.plot(ax=ax)
        #plt.show()

       # Define variables
        
        xs_col = self.xs.reshape(-1, 1)   # (nx,1)
        us_col = self.us.reshape(-1, 1)   # (nu,1)

        x_var = cp.Variable((self.nx, self.N + 1))
        u_var = cp.Variable((self.nu, self.N))
        x0_var = cp.Parameter((self.nx,))
        

        # Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form((x_var[:,i] - cp.reshape(x_ref, (self.nx,))), Q)
            cost += cp.quad_form((u_var[:,i] - cp.reshape(u_ref, (self.nu,))), R)

        # Terminal cost
        cost += cp.quad_form((x_var[:, -1] - cp.reshape(x_ref, (self.nx,))), Qf)

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

        # Store problem and variables
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
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
        #################################################
        # YOUR CODE HERE

        xss,uss = self.compute_steady_state(x_target)
        
        self.x0_var.value = x0
        self.x_ref.value = xss
        self.u_ref.value = uss
        self.ocp.solve(solver=cp.PIQP)
        # assert self.ocp.status == cp.OPTIMAL
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
