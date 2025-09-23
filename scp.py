import numpy as np
import scipy.sparse as sp
import osqp
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
        self.R = min_distance
        self.space_dims = space_dims

        #Hyperparameters
        self.convergence_tolerance = 1e-2
        self.safety_margin = 0.0  # Match first implementation

        # Initialization: Equality constraints and solution
        self.trajectories = None

        self.initial_positions = None
        self.initial_velocities = None
        self.initial_accelerations = None
        self.final_positions = None
        self.final_velocities = None
        self.final_accelerations = None

        # Initialization: Inequality Contraints
        self.pos_min = np.array([space_dims[0], space_dims[1]])
        self.pos_max = np.array([space_dims[2], space_dims[3]])

        # Limits to match first implementation
        self.vel_min = -10
        self.vel_max = 10

        self.acc_min = -5.0
        self.acc_max = 5.0

        self.jerk_min = -20
        self.jerk_max = 20

        # Initialization: constraint matrices
        self.A_jerk = None
        self.l_jerk = []
        self.u_jerk = []

        self.A_accel_initial = np.empty((0, 2 * self.N * self.K))
        self.l_accel_initial = []
        self.u_accel_initial = []

        self.A_accel_final = np.empty((0, 2 * self.N * self.K))
        self.l_accel_final = []
        self.u_accel_final = []

        self.A_accel_constraints = None
        self.l_accel_constraints = None
        self.u_accel_constraints = None

        print("---=== SCP Problem initialized ===---")
        print(f"Number of timesteps: {self.K}")
        print(f"Timestep: {self.h}")
        print(f"Minimum distance between vehicles: {self.R}")
        print(f"Space dimensions: {self.space_dims}")

    def set_initial_states(self, positions, velocities=None, accelerations=None):
        """Set initial states for all vehicles in (N, 2) format."""
        if positions.shape != (self.N, 2):
            raise ValueError(f"Positions must be shape ({self.N}, 2)")
            
        if velocities is None:
            velocities = np.zeros((self.N, 2))
        if accelerations is None:
            accelerations = np.zeros((self.N, 2))
            
        # Also convert into flattened arrays
        self.initial_positions = positions.flatten()
        self.initial_velocities = velocities.flatten()
        self.initial_accelerations = accelerations.flatten()

        assert len(self.initial_positions) == len(self.initial_velocities) == len(self.initial_accelerations)
        
    def set_final_states(self, positions, velocities=None, accelerations=None):
        """Set final states for all vehicles in (N, 2) format."""
        if positions.shape != (self.N, 2):
            raise ValueError(f"Positions must be shape ({self.N}, 2)")
            
        if velocities is None:
            velocities = np.zeros((self.N, 2))
        if accelerations is None:
            accelerations = np.zeros((self.N, 2))
            
        # Also convert into flattened arrays
        self.final_positions = positions.flatten()
        self.final_velocities = velocities.flatten()
        self.final_accelerations = accelerations.flatten()

        assert len(self.final_positions) == len(self.final_velocities) == len(self.final_accelerations)

    def generate_trajectories(self, max_iterations=15):
        """Main method to generate collision-free trajectories using SCP."""
        start_time = time.time()

        self.precompute_optimization_matrices()
        accelerations_flat = self._solve_initial_trajectory()

        #SCP iterations
        iteration = 0
        converged = False

        # Fast check: if initial solution has no collisions, return early
        try:
            init_positions, init_velocities = self._accelerations_to_positions_velocities(accelerations_flat)
            if self.N <= 1:
                accelerations = accelerations_flat.reshape(self.N, self.K, 2)
                self.trajectories = {
                    'positions': init_positions,
                    'velocities': init_velocities,
                    'accelerations': accelerations
                }
                print("No collision possible (N<=1); skipping SCP iterations.")
                end_time = time.time()
                print(f"Trajectory generation completed in {end_time - start_time:.3f} seconds")
                return self.trajectories

            # Compute pairwise distances over time and check minimum
            # positions shape: (N, K, 2)
            diff = init_positions[:, None, :, :] - init_positions[None, :, :, :]  # (N, N, K, 2)
            dist2 = np.sum(diff*diff, axis=-1)  # (N, N, K)
            iu = np.triu_indices(self.N, 1)
            pair_dist2 = dist2[iu]  # (num_pairs, K)
            min_pair_dist2 = np.min(pair_dist2) if pair_dist2.size > 0 else np.inf
            thresh2 = (self.R + self.safety_margin) ** 2
            if min_pair_dist2 >= thresh2:
                accelerations = accelerations_flat.reshape(self.N, self.K, 2)
                self.trajectories = {
                    'positions': init_positions,
                    'velocities': init_velocities,
                    'accelerations': accelerations
                }
                print("Initial trajectory collision-free; skipping SCP iterations.")
                end_time = time.time()
                print(f"Trajectory generation completed in {end_time - start_time:.3f} seconds")
                return self.trajectories
        except Exception as e:
            # If the quick check fails for any reason, proceed with SCP iterations
            print(f"Fast no-collision check failed, continuing. Reason: {e}")

        while iteration < max_iterations and not converged:
            print(f"SCP Iteration {iteration+1}")

            new_accelerations_flat = self._solve_with_avoidance_constraints(accelerations_flat)
            
            rel_step_norm = np.linalg.norm(new_accelerations_flat - accelerations_flat)/np.linalg.norm(accelerations_flat)
            print(rel_step_norm)
            if rel_step_norm <= self.convergence_tolerance:
                converged = True
                print(f"Converged after {iteration+1} iterations.")

            accelerations_flat = new_accelerations_flat
            iteration += 1

        accelerations = accelerations_flat.reshape(self.N, self.K, 2)
        positions, velocities = self._compute_positions_velocities(accelerations)

        self.trajectories = {
            'positions': positions,  # Shape (N, K, 2)
            'velocities': velocities,  # Shape (N, K, 2)
            'accelerations': accelerations  # Shape (N, K, 2)
        }

        end_time = time.time()
        print(f"Trajectory generation completed in {end_time - start_time:.3f} seconds")

        return self.trajectories

    def precompute_optimization_matrices(self):
        N, K, h = self.N, self.K, self.h

        # -------- jerk (sparse) --------
        self.A_jerk = _jerk_matrix_sparse(N, K, h)
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
        self.l_accel_constraints = np.full(m, self.acc_min, dtype=float)
        self.u_accel_constraints = np.full(m, self.acc_max, dtype=float)
    
    def _solve_initial_trajectory(self):
        """Solve initial trajectory without avoidance constraints."""

        # Create boundary constraints for position and velocity
        A_boundary, l_boundary, u_boundary = self._create_boundary_constraints()

        problem = osqp.OSQP()

        P = sp.csc_matrix(np.eye(2*self.N*self.K))
        q = np.array([0] * 2 * self.N * self.K)

        A = sp.vstack([
            self.A_jerk, 
            self.A_accel_initial, 
            self.A_accel_final, 
            self.A_accel_constraints,
            A_boundary
        ], format="csc")
        
        l = np.hstack([
            self.l_jerk, 
            self.l_accel_initial, 
            self.l_accel_final, 
            self.l_accel_constraints,
            l_boundary
        ])
        
        u = np.hstack([
            self.u_jerk, 
            self.u_accel_initial, 
            self.u_accel_final, 
            self.u_accel_constraints,
            u_boundary
        ])

        problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

        results = problem.solve()
        if results.info.status_val not in (1, 2):  # Solved / Solved Inaccurate
            raise RuntimeError(f"OSQP failed: {results.info.status}")

        accelerations_flat = results.x
        
        return accelerations_flat

    def _create_boundary_constraints(self):
        """Create position and velocity boundary constraints matching the first implementation."""
        N, K, h = self.N, self.K, self.h
        
        # Number of constraints: 
        # - Position bounds: 2*N*(K-1) (interior timesteps only)
        # - Final position equality: 2*N
        # - Final velocity equality: 2*N
        n_pos_interior = 2*N*(K-1)  # Interior position bounds
        n_pos_final = 2*N           # Final position equality
        n_vel_final = 2*N           # Final velocity equality
        n_constraints = n_pos_interior + n_pos_final + n_vel_final
        
        A_boundary = np.zeros((n_constraints, 2*N*K))
        l_boundary = []
        u_boundary = []
        
        row_idx = 0
        
        # Position constraints for each vehicle
        for i in range(N):
            # Get initial states for this vehicle
            p_init_x = self.initial_positions[2*i]
            p_init_y = self.initial_positions[2*i+1]
            v_init_x = self.initial_velocities[2*i]
            v_init_y = self.initial_velocities[2*i+1]
            
            for k in range(K):
                # Position coefficients: p[k] = p_init + v_init*k*h + sum(a[j]*h^2*(k-j-0.5))
                
                if k < K-1:  # Interior timesteps - position bounds
                    # x-position constraint
                    for j in range(k):
                        A_boundary[row_idx, 2*i*K + 2*j] = h*h * (k - j - 0.5)
                    
                    # Bounds
                    offset_x = p_init_x + k*h*v_init_x
                    l_boundary.append(self.pos_min[0] - offset_x)
                    u_boundary.append(self.pos_max[0] - offset_x)
                    row_idx += 1
                    
                    # y-position constraint
                    for j in range(k):
                        A_boundary[row_idx, 2*i*K + 2*j + 1] = h*h * (k - j - 0.5)
                    
                    # Bounds
                    offset_y = p_init_y + k*h*v_init_y
                    l_boundary.append(self.pos_min[1] - offset_y)
                    u_boundary.append(self.pos_max[1] - offset_y)
                    row_idx += 1
                    
                else:  # Final timestep - position equality
                    # x-position equality
                    for j in range(k):
                        A_boundary[row_idx, 2*i*K + 2*j] = h*h * (k - j - 0.5)
                    
                    # Equality bound
                    offset_x = p_init_x + k*h*v_init_x
                    target_x = self.final_positions[2*i] - offset_x
                    l_boundary.append(target_x)
                    u_boundary.append(target_x)
                    row_idx += 1
                    
                    # y-position equality
                    for j in range(k):
                        A_boundary[row_idx, 2*i*K + 2*j + 1] = h*h * (k - j - 0.5)
                    
                    # Equality bound
                    offset_y = p_init_y + k*h*v_init_y
                    target_y = self.final_positions[2*i+1] - offset_y
                    l_boundary.append(target_y)
                    u_boundary.append(target_y)
                    row_idx += 1
            
            # Final velocity equality constraints
            # v[K-1] = v_init + sum(a[j]*h) = v_final
            
            # x-velocity
            for j in range(K-1):
                A_boundary[row_idx, 2*i*K + 2*j] = h
            target_vx = self.final_velocities[2*i] - v_init_x
            l_boundary.append(target_vx)
            u_boundary.append(target_vx)
            row_idx += 1
            
            # y-velocity
            for j in range(K-1):
                A_boundary[row_idx, 2*i*K + 2*j + 1] = h
            target_vy = self.final_velocities[2*i+1] - v_init_y
            l_boundary.append(target_vy)
            u_boundary.append(target_vy)
            row_idx += 1
        
        A_boundary = sp.csc_matrix(A_boundary)
        l_boundary = np.array(l_boundary)
        u_boundary = np.array(u_boundary)
        
        return A_boundary, l_boundary, u_boundary

    def _compute_positions_velocities(self, accelerations):
        """
        Compute positions and velocities matching the first implementation exactly.
        """
        positions = np.zeros((self.N, self.K, 2))
        velocities = np.zeros((self.N, self.K, 2))
        
        # reshape flattened arrays back into (N,2)
        init_pos = self.initial_positions.reshape(self.N, 2)
        init_vel = self.initial_velocities.reshape(self.N, 2)
        
        for i in range(self.N):
            positions[i, 0] = init_pos[i]
            velocities[i, 0] = init_vel[i]
            
            for k in range(1, self.K):
                # velocity update: v[k] = v[0] + h * sum(a[j] for j in 0..k-1)
                velocities[i, k] = init_vel[i].copy()
                for j in range(k):
                    velocities[i, k] += self.h * accelerations[i, j]
                
                # position update: p[k] = p[0] + h*k*v[0] + sum(h^2*(k-j-0.5)*a[j] for j in 0..k-1)
                positions[i, k] = init_pos[i].copy() + self.h * k * init_vel[i]
                for j in range(k):
                    positions[i, k] += self.h**2 * (k - j - 0.5) * accelerations[i, j]
        
        return positions, velocities
    
    def _solve_with_avoidance_constraints(self, accelerations_flat):
        """Solve with collision avoidance constraints."""
        
        A_collision, l_collision, u_collision = self._add_collision_constraints(accelerations_flat)
        A_boundary, l_boundary, u_boundary = self._create_boundary_constraints()

        if not sp.issparse(A_collision):
            A_collision = sp.csc_matrix(A_collision)

        P = sp.csc_matrix(np.eye(2 * self.N * self.K))
        q = np.array([0] * 2 * self.N * self.K)

        A = sp.vstack([
            self.A_jerk, 
            self.A_accel_initial, 
            self.A_accel_final, 
            self.A_accel_constraints, 
            A_boundary,
            A_collision
        ], format="csc")
        
        l = np.hstack([
            self.l_jerk, 
            self.l_accel_initial, 
            self.l_accel_final, 
            self.l_accel_constraints, 
            l_boundary,
            l_collision
        ])
        
        u = np.hstack([
            self.u_jerk, 
            self.u_accel_initial, 
            self.u_accel_final, 
            self.u_accel_constraints, 
            u_boundary,
            u_collision
        ])

        problem = osqp.OSQP()
        problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True, max_iter=10000)
        problem.warm_start(x=accelerations_flat)

        result = problem.solve()
        if result.info.status_val not in (1, 2):
            print(f"Warning: OSQP status {result.info.status}")
        
        new_accelerations_flat = result.x
        
        return new_accelerations_flat

    def _add_collision_constraints(self, previous_solution):
        """
        Add linearized collision avoidance constraints exactly matching the first implementation.
        """
        # Convert flat solution to positions
        prev_positions, _ = self._accelerations_to_positions_velocities(previous_solution)

        # Number of collision constraints
        n_pairs = (self.N * (self.N - 1)) // 2
        n_constraints = n_pairs * self.K

        # Initialize constraint matrix and bounds
        A_collision = np.zeros((n_constraints, 2 * self.N * self.K))
        l_collision = np.ones(n_constraints) * (self.R + self.safety_margin)
        u_collision = np.ones(n_constraints) * np.inf

        # Current constraint row index
        row_idx = 0

        # For each time step
        for k in range(self.K):
            # For each pair of vehicles (i,j) where i < j
            for i in range(self.N):
                for j in range(i+1, self.N):
                    # Get previous positions
                    p_i_prev = prev_positions[i, k]
                    p_j_prev = prev_positions[j, k]

                    # Compute difference and distance
                    diff_vector = p_i_prev - p_j_prev
                    dist = np.linalg.norm(diff_vector)

                    # Handle numerical issues
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2*np.pi)
                        diff_vector = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0

                    # Compute normalized direction vector η
                    eta = diff_vector / dist

                    # Build constraint: η^T * (p_i[k] - p_j[k]) >= R + linearization_term
                    # where p_i[k] = p_init_i + k*h*v_init_i + sum(h^2*(k-m-0.5)*a_i[m])

                    # Set coefficients for acceleration variables
                    for m in range(k):
                        # Vehicle i contributions (positive)
                        A_collision[row_idx, 2*i*self.K + 2*m] = eta[0] * self.h**2 * (k - m - 0.5)
                        A_collision[row_idx, 2*i*self.K + 2*m + 1] = eta[1] * self.h**2 * (k - m - 0.5)

                        # Vehicle j contributions (negative)
                        A_collision[row_idx, 2*j*self.K + 2*m] = -eta[0] * self.h**2 * (k - m - 0.5)
                        A_collision[row_idx, 2*j*self.K + 2*m + 1] = -eta[1] * self.h**2 * (k - m - 0.5)

                    # Compute right-hand side with linearization
                    p_init_i_x = self.initial_positions[2*i]
                    p_init_i_y = self.initial_positions[2*i+1]
                    v_init_i_x = self.initial_velocities[2*i]
                    v_init_i_y = self.initial_velocities[2*i+1]

                    p_init_j_x = self.initial_positions[2*j]
                    p_init_j_y = self.initial_positions[2*j+1]
                    v_init_j_x = self.initial_velocities[2*j]
                    v_init_j_y = self.initial_velocities[2*j+1]

                    # Initial position + velocity contribution
                    init_pos_contrib = eta[0] * (p_init_i_x - p_init_j_x) + eta[1] * (p_init_i_y - p_init_j_y)
                    init_vel_contrib = eta[0] * (v_init_i_x - v_init_j_x) + eta[1] * (v_init_i_y - v_init_j_y)
                    init_vel_contrib *= k * self.h

                    # Linearization term
                    linearization_term = eta[0] * (p_i_prev[0] - p_j_prev[0]) + eta[1] * (p_i_prev[1] - p_j_prev[1]) - dist

                    # Set the RHS
                    rhs = (self.R + self.safety_margin) + linearization_term - (init_pos_contrib + init_vel_contrib)
                    l_collision[row_idx] = rhs

                    row_idx += 1
        
        return A_collision, l_collision, u_collision

    def _accelerations_to_positions_velocities(self, accelerations_flat):
        """Convert flat acceleration array to positions and velocities."""
        N, K, h = self.N, self.K, self.h
        
        # Initialize arrays
        positions = np.zeros((N, K, 2))
        velocities = np.zeros((N, K, 2))
        
        # Set initial conditions
        for i in range(N):
            positions[i, 0, 0] = self.initial_positions[2*i]
            positions[i, 0, 1] = self.initial_positions[2*i + 1]
            velocities[i, 0, 0] = self.initial_velocities[2*i]
            velocities[i, 0, 1] = self.initial_velocities[2*i + 1]
        
        # Compute trajectories using the exact same method as first implementation
        for i in range(N):
            for k in range(1, K):
                # Velocity: v[k] = v[0] + h * sum(a[j] for j in 0..k-1)
                velocities[i, k, 0] = velocities[i, 0, 0]
                velocities[i, k, 1] = velocities[i, 0, 1]
                for j in range(k):
                    ax = accelerations_flat[2*i*K + 2*j]
                    ay = accelerations_flat[2*i*K + 2*j + 1]
                    velocities[i, k, 0] += h * ax
                    velocities[i, k, 1] += h * ay
                
                # Position: p[k] = p[0] + k*h*v[0] + sum(h^2*(k-j-0.5)*a[j] for j in 0..k-1)
                positions[i, k, 0] = positions[i, 0, 0] + k*h*velocities[i, 0, 0]
                positions[i, k, 1] = positions[i, 0, 1] + k*h*velocities[i, 0, 1]
                for j in range(k):
                    ax = accelerations_flat[2*i*K + 2*j]
                    ay = accelerations_flat[2*i*K + 2*j + 1]
                    positions[i, k, 0] += h*h * (k - j - 0.5) * ax
                    positions[i, k, 1] += h*h * (k - j - 0.5) * ay
        
        return positions, velocities

    def visualize_trajectories(self, show_animation=True, save_path=None):
        """Visualize the generated 2D trajectories."""
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")
            
        positions = self.trajectories['positions']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Plot bounds of the space
        xmin, ymin, xmax, ymax = self.space_dims
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        
        # Plot initial points as triangles
        for i in range(self.N):
            ax.scatter(
                positions[i, 0, 0], positions[i, 0, 1],
                marker='^', s=100, color=f'C{i}', label=f'Vehicle {i+1} start'
            )
            
        # Plot final points as squares
        for i in range(self.N):
            ax.scatter(
                positions[i, -1, 0], positions[i, -1, 1],
                marker='s', s=100, color=f'C{i}', label=f'Vehicle {i+1} end'
            )
            
        if show_animation:
            # Animate trajectories
            circles = []  # Keep track of vehicle circles
            
            # Initialize circles
            for i in range(self.N):
                vehicle_circle = Circle(positions[i, 0], self.R/2, color=f'C{i}', alpha=0.5)
                circles.append(vehicle_circle)
                ax.add_patch(vehicle_circle)
            
            # Safety circles showing minimum distance
            safety_circles = []
            for i in range(self.N):
                safety_circle = Circle(positions[i, 0], self.R, color=f'C{i}', alpha=0.1, fill=True)
                safety_circles.append(safety_circle)
                ax.add_patch(safety_circle)
            
            # Animate
            for k in range(self.K):
                # Update circle positions
                for i in range(self.N):
                    # Update vehicle position
                    circles[i].center = positions[i, k]
                    safety_circles[i].center = positions[i, k]
                    
                    # Draw trajectory path segments
                    if k > 0:
                        ax.plot(
                            positions[i, k-1:k+1, 0], 
                            positions[i, k-1:k+1, 1],
                            color=f'C{i}'
                        )
                
                plt.pause(0.05)
                plt.draw()
        else:
            # Plot full trajectories
            for i in range(self.N):
                # Plot the full trajectory
                ax.plot(
                    positions[i, :, 0], positions[i, :, 1],
                    color=f'C{i}', label=f'Vehicle {i+1} path'
                )
                
                # Plot safety circles at final positions
                safety_circle = Circle(positions[i, -1], self.R, color=f'C{i}', alpha=0.1, fill=True)
                ax.add_patch(safety_circle)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        ax.legend()
        plt.title('2D Collision-Free Trajectories')
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
        return fig, ax
    
    def visualize_time_snapshots(self, num_snapshots=5, save_path=None):
        """
        Visualize trajectories as a series of time snapshots similar to Figure 2 in the paper.
        
        Args:
            num_snapshots: Number of snapshots to display
            save_path: Optional path to save the figure
        """
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")
        
        positions = self.trajectories['positions']
        K = self.K
        
        # Select frames to visualize evenly spaced in time
        frame_indices = np.linspace(0, K-1, num_snapshots, dtype=int)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_snapshots, figsize=(15, 3))
        
        # Plot each frame
        for f, frame_idx in enumerate(frame_indices):
            ax = axes[f]
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            
            # Remove ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set bounds with some padding
            xmin, ymin, xmax, ymax = self.space_dims
            ax.set_xlim([xmin - 0.5, xmax + 0.5])
            ax.set_ylim([ymin - 0.5, ymax + 0.5])
            
            # Show time in the title
            current_time = frame_idx * self.h
            ax.set_title(f"t = {current_time:.1f}s")
            
            # Plot circles for each vehicle at this time
            for i in range(self.N):
                pos = positions[i, frame_idx]  # Get x, y position
                
                # Create circles showing vehicle and required minimum distance
                vehicle_circle = Circle(pos, self.R/2, color=f'C{i}', alpha=0.7)
                safety_circle = Circle(pos, self.R, color=f'C{i}', alpha=0.1, fill=True)
                
                ax.add_patch(vehicle_circle)
                ax.add_patch(safety_circle)
                
                # Draw trajectory lines up to current frame
                if frame_idx > 0:
                    # Get positions up to current frame
                    traj_x = positions[i, :frame_idx+1, 0]
                    traj_y = positions[i, :frame_idx+1, 1]
                    
                    # Draw trajectory
                    ax.plot(traj_x, traj_y, '-', color=f'C{i}', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, axes


# Example usage to test the implementation
if __name__ == "__main__":
    # Create the SCP planner
    planner = SCP(
        n_vehicles=3,
        time_horizon=3.0,
        time_step=0.2,
        min_distance=0.5,
        space_dims=[-5, -5, 5, 5]
    )
    
    # Set initial positions (spread out)
    initial_positions = np.array([
        [-2, -2],
        [0, -2], 
        [2, -2]
    ])
    
    # Set final positions (crossed over)
    final_positions = np.array([
        [2, 2],
        [0, 2],
        [-2, 2]
    ])
    
    # Set the states
    planner.set_initial_states(initial_positions)
    planner.set_final_states(final_positions)
    
    # Generate trajectories
    trajectories = planner.generate_trajectories(max_iterations=10)
    
    # Visualize
    planner.visualize_trajectories(show_animation=False)
    planner.visualize_time_snapshots(num_snapshots=4)
