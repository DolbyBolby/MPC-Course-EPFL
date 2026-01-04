import numpy as np
from control import dlqr
from mpt4py import Polyhedron
import matplotlib.pyplot as plt
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float


    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        # Vérifier que les dimensions du sous-système z sont correctes
        print("self.A shape:", self.A.shape)
        print("self.B shape:", self.B.shape)
        print("xs_z shape:", self.x_ids.shape)
        print("us_z shape:", self.u_ids.shape)
        print('xs',self.xs)
        print('us',self.us)

        #Computing K with  LQR 
        # Q= np.diag([50.0, 10.0])   # extrem don't work, random work but go below 3, no noise is slow //Closed-loop eigvals: [0.92005507 0.97705465]
        # R = np.array([[0.5]])     
        # Q = np.diag([200.0, 50.0]) #random extrem and no noise work but tooo slow //Closed-loop eigvals: [0.84284718 0.97504436]
        # R = np.array([[0.5]])
        Q = np.diag([300.0, 80.0]) #the best now
        R = np.array([[0.05]]) 
        # Q = np.diag([300.0, 1200.0])   # random and extrem dont work, no noise go quickly at z=3 // Closed-loop eigvals: [0.72884042 0.89991437]
        # R = np.array([[0.2]])
        # Q = np.diag([10, 50]) #extrem dont work, random work no noise go quickly at z=3 // Closed-loop eigvals: [0.95002993+0.04020782j 0.95002993-0.04020782j]
        # R = np.array([[0.5]])
        # Q = np.diag([100, 70]) #extrem dont work, random work no noise go at z=3 but bit slowly //Closed-loop eigvals: [0.93833553+0.01604655j 0.93833553-0.01604655j]
        # R = np.array([[0.5]])
        # Q = np.diag([0.1, 20.0]) # random and extrem don't work, no noise work well //Closed-loop eigvals: [0.93408085+0.06127015j 0.93408085-0.06127015j]
        # R = np.array([[0.05]])
        # Q = np.diag([800.0, 800.0])   # extrem don't work, random work well, no noise work a bit slow//Closed-loop eigvals: [0.78766993 0.95014544]
        # R = np.array([[1.0]])
        # Q = np.diag([1000.0, 1200.0])# extrem random don't work, no noise work a bit slow//Closed-loop eigvals: [0.68350763 0.94614848]
        # R = np.array([[0.5]])
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        self.K = K # don't know if thats supposed to be -K
        self.Qf= Qf
        print("Qf =", Qf)
        A_cl = self.A - self.B @ self.K   
        eigvals = np.linalg.eigvals(A_cl) # vérifier la stabilité
        print("Closed-loop eigvals:", eigvals)

        # minimal invariant set epsilon
        w_min, w_max = -15, 5
        W = Polyhedron.from_Hrep(A=np.array([[1], [-1]]),b=np.array([w_max, -w_min]))
        Bd = self.B
        print ('Bd',Bd)
        BW = W.affine_map(Bd) 
        Z = Polyhedron.from_Hrep( A = np.array([[0.0, -1.0]]),b = np.array([self.xs[1]]))
        #Z = Polyhedron.from_Hrep(A = np.array([[0.0, -1.0]]),b = np.array([0.0]))
        Epsilon = self.min_invariant_set(A_cl, BW)

        # Visualization of epsilon
        VIZ_BOX = Polyhedron.from_Hrep( A=np.vstack((np.eye(2), -np.eye(2))),b=np.array([3.0, 3.0, 3.0, 3.0]))
        Epsilon_plot = Epsilon.intersect(Z).intersect(VIZ_BOX)
        Z_plot = Z.intersect(VIZ_BOX)
        fig1, ax1 = plt.subplots(1, 1)
        Z_plot.plot(ax1, color='g', opacity=0.5, label='State constraints Z')
        Epsilon_plot.plot(ax1, color='r', opacity=0.5, label='Invariant set Epsilon')
        plt.legend()
        plt.show()

        #tightened set constraints
        Z_tilde = Z - Epsilon
        #tightened input constraints
        U = Polyhedron.from_Hrep(np.array([[-1],[1]]),np.array([-(40 - self.us[0] ),80 - self.us[0]]))
        KE = Epsilon.affine_map(self.K) #changer k en self 
        U_tilde = U - KE # change - in +
                # U_tilde is your Polyhedron object: A u <= b
        # For 1D, A is shape (2,1) and b is (2,)
        # Solve for bounds: u_min <= u <= u_max

        A = U_tilde.A
        b = U_tilde.b

        # Assuming U_tilde is 1D (nu=1)
        u_bounds = []
        for i in range(len(b)):
            if A[i,0] != 0:
                u_bounds.append(b[i] / A[i,0])

        # Because inequalities may flip sign, take proper min/max
        u_min = min(u_bounds)
        u_max = max(u_bounds)
        vertices = np.array([u_min, u_max])
        print("Vertices of the tightened input constraint U~:", vertices)

        # Plotting
        plt.figure(figsize=(6,2))
        plt.plot(vertices, [0,0], 'ro', label='Vertices of U~')
        plt.hlines(0, u_min, u_max, colors='b', lw=4, alpha=0.3, label='U~ range')
        plt.xlabel('Input (thrust)')
        plt.yticks([])
        plt.title('Tightened Input Constraint $\~U$')
        plt.legend()
        plt.grid(True)
        plt.show()
        #terminal set 
        KU_tilde= Polyhedron.from_Hrep(U_tilde.A@self.K, U_tilde.b) #changer k en self
        Z_tilde_KU_tilde = Z_tilde.intersect(KU_tilde)
        Xf = self.max_invariant_set(A_cl, Z_tilde_KU_tilde)  
        assert Xf.contains(np.zeros(self.nx)) #check that the terminal set contain the origin

        #verification of the set 
        print("Z contains origin:", Z.contains(np.zeros(self.nx)))
        print("Z_tilde contains origin:", Z_tilde.contains(np.zeros(self.nx)))
        print("U_tilde empty:", U_tilde.is_empty)
        print("Xf contains origin:", Xf.contains(np.zeros(self.nx)))
        print("U_tilde contains zero:", U_tilde.contains(np.zeros(self.nu)))
        #visualisation of max invariant set
        fig3, ax3 = plt.subplots(1, 1)
        Xf.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{X}_f$')
        Epsilon_plot.plot(ax3, color='b', opacity=0.5, label='Invariant set Epsilon')
        plt.legend()
        plt.show()

        # FORMULATION OF TUBE MPC
        
            # Define variables
        #nx, nu = self.B.shape
        self.z_var = cp.Variable((self.N+1, self.nx), name='z')
        self.u_var = cp.Variable((self.N, self.nu), name='u')
        self.z0_var = cp.Parameter((self.nx,), name='z0')

        # self.z_var = cp.Variable((nx, self.N+1), name='z')
        # self.u_var = cp.Variable((nu,self.N), name='u')
        # self.z0_var = cp.Parameter((nx,), name='z0')

        # self.x_var = cp.Variable((self.nx, self.N+1), name='x')
        # self.u_var = cp.Variable((self.nu,self.N), name = 'u')
        # self.x0_var = cp.Parameter((self.nx,), name='x0')

        ## Costs
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(self.z_var[i], Q)
            cost += cp.quad_form(self.u_var[i], R)
        cost += cp.quad_form(self.z_var[-1], self.Qf)
        print("selfQf =", self.Qf)


        ## Constraints
        constraints = []

        # initial condition: x0 in z0 + E
        constraints.append(Epsilon.A @ (self.z0_var - self.z_var[0]) <= Epsilon.b) #change Epsilon.A @ (self.z_var[0] - self.z0_var) <= Epsilon.b
                            
        # dynamics
        #constraints.append(self.z_var[1:].T == self.A @ self.z_var[:-1].T + self.B @ self.u_var.T)
        for k in range(self.N):
            constraints.append(self.z_var[k+1] == self.A @ self.z_var[k] + self.B @ self.u_var[k])
        # state constraints
        constraints.append(Z_tilde.A @ self.z_var[:-1].T <= Z_tilde.b.reshape(-1, 1))

        # input constraints
        constraints.append( U_tilde.A @ self.u_var.T <= U_tilde.b.reshape(-1, 1)) #change U_tilde.A @ self.u_var[:-1].T <= U_tilde.b.reshape(-1, 1)
       

        # terminal set
        constraints.append(Xf.A @ self.z_var[-1].T <= Xf.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost),constraints)


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        delta_z0 = x0-self.xs
        self.z0_var.value = np.atleast_1d(delta_z0)
        self.ocp.solve(solver=cp.PIQP)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"MPC Warning: Status {self.ocp.status}")
            return np.array([self.us[0]]), np.zeros((self.nx, self.N+1)), np.zeros((self.nu, self.N))

        u0_nom = self.u_var.value[0, :]     
        z0_nom = self.z_var.value[0, :] 
        delta_u0 = u0_nom - self.K @ (delta_z0 - z0_nom)
        u0 = delta_u0 + self.us
        # z_traj = self.z_var.value + self.xs.reshape(-1,1)
        # u_traj = self.u_var.value + self.us.reshape(-1,1)
        z_traj = self.z_var.value.T + self.xs.reshape(-1, 1)
        u_traj = self.u_var.value.T + self.us.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u0, z_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
    
    def min_invariant_set (self, A_cl: np.ndarray, BW, max_iter: int = 30) -> Polyhedron:
        Omega = BW
        BW_Ai = BW #for i=0
        max_iter = 30
        
        for i in range(1, max_iter):
            # propagate through closed-loop dynamics
            BW_Ai = BW_Ai.affine_map(A_cl)

            # Minkowski sum
            Omega_next = Omega.minkowski_sum(BW_Ai)
            Omega_next.minHrep()
            
            # Check convergence using bounding box norm
            bbox_prev = Polyhedron.bounding_box(Omega)
            bbox_next = Polyhedron.bounding_box(Omega_next)
            if np.linalg.norm(bbox_next.b - bbox_prev.b) < 1e-1:
                print('Minimal robust invariant set computation converged after', i,  'iterations.')
                Epsilon = Omega_next 
                return Epsilon  
            else:
                Omega = Omega_next
    
    def max_invariant_set(self, A_cl, X: Polyhedron, max_iter = 30) -> Polyhedron:

        O = X
        for _ in range(max_iter):
            Oprev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(
                np.vstack((F, F @ A_cl)),
                np.hstack((f, f))
            )
            O.minHrep(True)
            _ = O.Vrep  # improves numerical robustness
            if O == Oprev:
                return O
        return O


