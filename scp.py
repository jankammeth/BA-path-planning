import numpy as np
import scipy.sparse as sp
import osqp
import time

def _jerk_matrix_sparse(N, K, h):
    """(2N(K-1)) x (2NK) first-difference operator per vehicle/axis, scaled by 1/h."""
    rows, cols, vals = [], [], []
    r = 0
    for i in range(N):
        base = 2*i*K
        for k in range(K-1):
            # x
            rows += [r, r]; cols += [base+2*k, base+2*(k+1)]; vals += [-1.0/h, 1.0/h]; r += 1
            # y
            rows += [r, r]; cols += [base+2*k+1, base+2*(k+1)+1]; vals += [-1.0/h, 1.0/h]; r += 1
    m, n = 2*N*(K-1), 2*N*K
    return sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsc()

def _accel_initial_selector_sparse(N, K):
    """(2N) x (2NK) select a_x[0], a_y[0] per vehicle as equality."""
    rows, cols, vals = [], [], []
    for i in range(N):
        base = 2*i*K
        rows += [2*i,   2*i+1]
        cols += [base,  base+1]
        vals += [1.0,   1.0]
    return sp.coo_matrix((vals, (rows, cols)), shape=(2*N, 2*N*K)).tocsc()

def _accel_final_selector_sparse(N, K):
    """(2N) x (2NK) select a_x[K-1], a_y[K-1] per vehicle as equality."""
    rows, cols, vals = [], [], []
    for i in range(N):
        base = 2*i*K + 2*(K-1)
        rows += [2*i,   2*i+1]
        cols += [base,  base+1]
        vals += [1.0,   1.0]
    return sp.coo_matrix((vals, (rows, cols)), shape=(2*N, 2*N*K)).tocsc()

#TODO: review
def _vel_transform_axis_sparse(K, h):
    """K x K lower-triangular (strict) with entries h for j<k."""
    r, c = np.tril_indices(K, k=-1)
    data = np.full_like(r, fill_value=h, dtype=float)
    return sp.coo_matrix((data, (r, c)), shape=(K, K)).tocsc()

#TODO: review
def _pos_transform_axis_sparse(K, h):
    """K x K lower-triangular (strict) with entries h^2*(k-j-0.5) for j<k."""
    r, c = np.tril_indices(K, k=-1)
    data = (h*h) * (r - c - 0.5)
    return sp.coo_matrix((data, (r, c)), shape=(K, K)).tocsc()

#TODO: review
def _interleave_xy_block(T_axis):
    """From KxK axis transform -> 2Kx2K interleaved (ax0,ay0,ax1,ay1,...) via Kron with I2."""
    return sp.kron(T_axis, sp.eye(2, format="csc"), format="csc")   

#TODO: review
def _blockdiag_over_vehicles(B_per_vehicle, N):
    """Block-diagonal repeat over vehicles (CSC)."""
    return sp.block_diag([B_per_vehicle]*N, format="csc")


class SCP():
    def __init__(self,
                 n_vehicles = 5,
                 time_horizon = 3.0,
                 time_step = 0.1,
                 min_distance = 0.1, 
                 space_dims=[-10, -10, 10, 10],
                 ):
        #Initialize SCP

        self.N = n_vehicles
        self.T = time_horizon
        self.h = time_step
        self.K = int(self.T / self.h) + 1
        self.R_margin = min_distance
        self.space_dims = space_dims

        # Initialization: Equality constraints and solution
        self.initial_positions = None
        self.final_states = None
        self.trajectories = None

        # Initialization: Inequality Contraints
        self.pos_min = np.array([space_dims[0], space_dims[1]])
        self.pos_max = np.array([space_dims[2], space_dims[3]])

        #NOTE: vel, accel margins?
        self.vel_min = -0.75
        self.vel_max = 0.75

        self.acc_min = -15.0
        self.acc_max = 15.0

        self.jerk_min = -15
        self.jerk_max = 15

        # Initialization: Equality & Inequality constraint matrices
        # l <= Ax <= u,     x: stacked accelerations
        self.A_jerk = None      #jerk inequality
        self.l_jerk = []
        self.u_jerk = []

        self.A_accel_initial = np.empty((0, 2 * self.N * self.K))   #acceleration equality (initial)
        self.l_accel_initial = []
        self.u_accel_initial = []

        self.A_accel_final = np.empty((0, 2 * self.N * self.K))     #acceleration equality (final)
        self.l_accel_final = []
        self.u_accel_final = []

        self.A_accel_constraints = None    #acceleration inequality  
        self.l_accel_constraints = None
        self.u_accel_constraints = None

        self.A_vel_constraints = None       #velocity inequality
        self.l_vel_constraints = []
        self.u_vel_constraints = []

        self.A_pos_constraints = None       #position inequality
        self.l_pos_constraints = []
        self.u_pos_constraints = []

        print("---=== SCP Problem initialized ===---")
        print(f"Number of timesteps: {self.K}")
        print(f"Timestep: {self.h}")
        print(f"Minimum distance between vehicles: {self.R}")
        print(f"Space dimensions: {self.space_dims}")

    #NOTE: ev. flatten array
    def set_initial_states(self, positions, velocities=None, accelerations=None):
        """Set initial states for all vehicles."""
        if positions.shape != (self.N, 2):
            raise ValueError(f"Positions must be shape ({self.N}, 2)")
            
        if velocities is None:
            velocities = np.zeros((self.N, 2))
        if accelerations is None:
            accelerations = np.zeros((self.N, 2))
            
        self.initial_states = {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations
        }

    #NOTE: ev. flatten array
    def set_final_states(self, positions, velocities=None, accelerations=None):
        """Set final states for all vehicles."""
        if positions.shape != (self.N, 2):
            raise ValueError(f"Positions must be shape ({self.N}, 2)")
            
        if velocities is None:
            velocities = np.zeros((self.N, 2))
        if accelerations is None:
            accelerations = np.zeros((self.N, 2))
            
        self.final_states = {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations
        }

    def precompute_optimization_matrices(self):
        N, K, h = self.N, self.K, self.h

        # -------- jerk (sparse) --------
        self.A_jerk = self._jerk_matrix_sparse(N, K, h)
        self.l_jerk = np.full(2*N*(K-1), self.jerk_min, dtype=float)
        self.u_jerk = np.full(2*N*(K-1), self.jerk_max, dtype=float)

        # -------- acceleration equalities (sparse selectors) --------
        self.A_accel_initial = _accel_initial_selector_sparse(N, K)
        self.l_accel_initial = np.asarray(self.initial_accelerations, dtype=float)
        self.u_accel_initial = np.asarray(self.initial_accelerations, dtype=float)

        self.A_accel_final = _accel_final_selector_sparse(N, K)
        self.l_accel_final = np.asarray(self.final_accelerations, dtype=float)
        self.u_accel_final = np.asarray(self.final_accelerations, dtype=float)

        # -------- acceleration box (identity, sparse) --------
        m = 2*N*K
        self.A_accel_constraints = sp.eye(m, format="csc")
        self.l_accel_constraints = np.full(m, self.acc_min + self.acc_margin, dtype=float)
        self.u_accel_constraints = np.full(m, self.acc_max - self.acc_margin, dtype=float)

        #TODO: review
        # -------- velocity transform (sparse) --------
        Tv = _vel_transform_axis_sparse(K, h)   # KxK
        Bv = _interleave_xy_block(Tv)           # 2K x 2K (x/y interleaved)
        self.A_vel_constraints = _blockdiag_over_vehicles(Bv, N)   # (2NK) x (2NK)

        # bounds: interior timesteps are box; final (per vehicle, 2 axes) are equalities
        self.l_vel_constraints, self.u_vel_constraints = [], []
        for i in range(N):
            # 2K-2 interiors
            self.l_vel_constraints.extend([self.vel_min + self.vel_margin] * (2*K - 2))
            self.u_vel_constraints.extend([self.vel_max - self.vel_margin] * (2*K - 2))
            # final velocity equality (vx_T - vx_0, vy_T - vy_0)
            dvx = self.final_velocities[2*i]   - self.initial_velocities[2*i]
            dvy = self.final_velocities[2*i+1] - self.initial_velocities[2*i+1]
            self.l_vel_constraints.extend([dvx, dvy])
            self.u_vel_constraints.extend([dvx, dvy])
        self.l_vel_constraints = np.asarray(self.l_vel_constraints, dtype=float)
        self.u_vel_constraints = np.asarray(self.u_vel_constraints, dtype=float)

        #TODO: review
        # -------- position transform (sparse) --------
        Tp = _pos_transform_axis_sparse(K, h)   # KxK
        Bp = _interleave_xy_block(Tp)           # 2K x 2K
        self.A_pos_constraints = _blockdiag_over_vehicles(Bp, N)   # (2NK) x (2NK)

        # bounds: interior workspace box; final position equality (uses offsets you add in l/u)
        self.l_pos_constraints, self.u_pos_constraints = [], []
        for i in range(N):
            for k in range(K):
                p_init_x = self.initial_positions[2*i]
                p_init_y = self.initial_positions[2*i+1]
                v_init_x = self.initial_velocities[2*i]
                v_init_y = self.initial_velocities[2*i+1]
                p_init_contrib_x = p_init_x + k*h*v_init_x
                p_init_contrib_y = p_init_y + k*h*v_init_y

                if k == K - 1:
                    # final position equality
                    self.l_pos_constraints.extend([
                        self.final_positions[2*i]   - p_init_contrib_x,
                        self.final_positions[2*i+1] - p_init_contrib_y
                    ])
                    self.u_pos_constraints.extend([
                        self.final_positions[2*i]   - p_init_contrib_x,
                        self.final_positions[2*i+1] - p_init_contrib_y
                    ])
                else:
                    # workspace box
                    self.l_pos_constraints.extend([
                        self.pos_min[0] - p_init_contrib_x,
                        self.pos_min[1] - p_init_contrib_y
                    ])
                    self.u_pos_constraints.extend([
                        self.pos_max[0] - p_init_contrib_x,
                        self.pos_max[1] - p_init_contrib_y
                    ])
        self.l_pos_constraints = np.asarray(self.l_pos_constraints, dtype=float)
        self.u_pos_constraints = np.asarray(self.u_pos_constraints, dtype=float)
    
    def generate_trajectories(self, max_iterations = 20):
        """Main method to generate collision-free trajectories using SCP."""
        if self.initial_states is None or self.final_states is None:
            raise ValueError("Initial and final states must be set before generating trajectories")
        
        start_time = time.time()
        #solve linear interpolation for initial guess of optimization variable
        self._solve_initial_trajectory()

        self.plot_positions()
        # Initialize previous solution
        prev_solution = self.accelerations

        # SCP iterations
        iteration = 0
        converged = False
        feasibility_satisfied = False
        while iteration < max_iterations and not converged:
            print(f"SCP iteration {iteration+1}/{max_iterations}")
            
            #Stack sparse constraint matrices
            A = sp.vstack([
                self.A_jerk, 
                self.A_accel_initial, 
                self.A_accel_final, 
                self.A_accel_constraints, 
                self.A_vel_constraints, 
                self.A_pos_constraints
            ])
            
            l = np.hstack([
                self.l_jerk, 
                self.l_accel_initial, 
                self.l_accel_final, 
                self.l_accel_constraints, 
                self.l_vel_constraints, 
                self.l_pos_constraints
            ])
            
            u = np.hstack([
                self.u_jerk, 
                self.u_accel_initial, 
                self.u_accel_final, 
                self.u_accel_constraints, 
                self.u_vel_constraints, 
                self.u_pos_constraints
            ])
            
            # Add collision avoidance constraints
            A_collision, l_collision, u_collision = self.add_collision_constraints(prev_solution)
            
            # Combine all constraints
            A = np.vstack([A, A_collision])
            l = np.hstack([l, l_collision])
            u = np.hstack([u, u_collision])
            
            # Minimize acceleration squared
            P = np.eye(2 * self.N * self.K)
            q = np.array([0] * 2 * self.N * self.K)
            
            # Convert to sparse matrices for OSQP
            P = sp.csc_matrix(P)
            A = sp.csc_matrix(A)
            
            # Setup and solve
            problem = osqp.OSQP()
            problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=True, warm_start=True, max_iter=10000)

            # Warm start with previous solution
            problem.warm_start(x=prev_solution)
            
            # Solve and save result
            result = problem.solve()
            
            # Check for solver errors
            if result.info.status != "solved":
                print(f"Solver failed at iteration {iteration+1}: {result.info.status}")
                return False, None, None
            
            # Update solution
            new_solution = result.x
            
            # Check convergence
            converged = False
            solution_diff = np.linalg.norm(new_solution - prev_solution)
            #print(f"Solution change: {solution_diff}")
            if solution_diff < 0.01:  # Convergence threshold
                converged = True

            positions, velocities = self._accelerations_to_positions_velocities(new_solution)
            avoidance_satisfied = self._fast_check_avoidance_constraints(positions)
            if not avoidance_satisfied:
                print("Warning: Avoidance constraints not satisfied at discrete timesteps.")
                converged = False

            # Perform continuous collision checking - only if avoidance is satisfied at discrete timesteps
            if avoidance_satisfied:
                continuous_avoidance_satisfied = self._optimized_check_continuous_avoidance(
                    positions, velocities, new_solution
                )
                if not continuous_avoidance_satisfied:
                    print("Warning: Continuous avoidance constraints not satisfied.")

                    # If we've converged but continuous collision checking fails, continue iterations
                    if converged and iteration < max_iterations - 1:
                        print("Continuing iterations to resolve continuous collisions.")
                        converged = False

                feasibility_satisfied = self._simple_feasibility_check(positions, velocities, new_solution)
                if not feasibility_satisfied:
                    print("Warning: Feasibility constraints not satisfied.")

            # Update previous solution for next iteration
            prev_solution = new_solution
            iteration += 1


        # Save the final solution
        self.accelerations = prev_solution
        if feasibility_satisfied:
            print(f"SCP trajectory solved successfully after {iteration} iterations.")
            return True, positions, velocities
        else:
            print(f"SCP trajectory failed to satisfy feasibility constraints after {iteration} iterations.")
            return False, positions, velocities
        
    def _solve_initial_trajectory(self):
        """
        Solve the optimization problem without the collision constraints but already taking into account:
        Initial and Final positions, velocities, accelerations
        Position boundary, min/max velocity, acceleration, jerk
        """

        problem = osqp.OSQP()

        # Minimize acceleration squared
        P = np.eye(2 * self.N * self.K)

        A = np.vstack([self.A_jerk, self.A_accel_initial, self.A_accel_final, self.A_accel_constraints, self.A_vel_constraints, self.A_pos_constraints])
        l = np.hstack([self.l_jerk, self.l_accel_initial, self.l_accel_final, self.l_accel_constraints, self.l_vel_constraints, self.l_pos_constraints])
        u = np.hstack([self.u_jerk, self.u_accel_initial, self.u_accel_final, self.u_accel_constraints, self.u_vel_constraints, self.u_pos_constraints])
        q = np.array([0] * 2 * self.N * self.K)

        # convert into sparse matrixes for OSQP
        P = sp.csc_matrix(P)
        A = sp.csc_matrix(A)

        problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=True)

        # save the result in the optimization variable
        results = problem.solve()
        self.accelerations = results.x
