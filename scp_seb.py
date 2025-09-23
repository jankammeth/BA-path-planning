import numpy as np
import osqp
import scipy.sparse as sp
from scipy.sparse import block_diag
import matplotlib.pyplot as plt

class SCPSeb:  # Changed class name to match main function
    def __init__(self,
                 n_vehicles=5,
                 time_horizon=10,
                 time_step=0.1,
                 discrete_num_steps=None,  # Will be computed from time_horizon and time_step
                 min_distance=1.0,
                 space_dims=[-10, -10, 10, 10]):

        self.N = n_vehicles
        self.T = time_horizon
        self.h = time_step
        # Compute number of discrete steps from time_horizon and time_step
        if discrete_num_steps is None:
            self.K = int(time_horizon / time_step) + 1
        else:
            self.K = discrete_num_steps
        
        self.R = min_distance
        self.R_margin = 0.5
        self.space_dims = space_dims

        print(f"Number of timesteps: {self.K}")
        print(f"Timestep: {self.h}")
        print(f"Minimum distance between vehicles: {self.R}")
        print(f"Space dimensions: {self.space_dims}")

        # Position limits
        self.pos_min = np.array([space_dims[0], space_dims[1]])
        self.pos_max = np.array([space_dims[2], space_dims[3]])

        # Velocity limits
        self.vel_min = -0.75
        self.vel_max = 0.75
        self.vel_margin = 0.05

        # Acceleration limits
        self.acc_min = -15.0
        self.acc_max = 15.0
        self.acc_margin = 1.0

        # Jerk limits
        self.jerk_min = np.array([-15, -15])
        self.jerk_max = np.array([15, 15])

        # State variables
        self.initial_positions = None
        self.initial_velocities = None
        self.initial_accelerations = None
        self.final_positions = None
        self.final_velocities = None
        self.final_accelerations = None

        # Optimization variable
        self.accelerations = np.array([None] * 2 * self.N * self.K)
        
        # Obstacle trajectories
        self.obstacle_trajectories = None

        # Constraint matrices
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

        self.A_vel_constraints = None
        self.l_vel_constraints = []
        self.u_vel_constraints = []

        self.A_pos_constraints = None
        self.l_pos_constraints = []
        self.u_pos_constraints = []

        # Store final trajectories
        self.trajectories = None

    def set_initial_states(self, initial_positions):
        """Set initial states - now accepts just positions and sets velocities/accelerations to zero"""
        self.initial_positions = initial_positions.flatten()
        self.initial_velocities = np.zeros(2 * self.N)
        self.initial_accelerations = np.zeros(2 * self.N)

    def set_final_states(self, final_positions):
        """Set final states - now accepts just positions and sets velocities/accelerations to zero"""
        self.final_positions = final_positions.flatten()
        self.final_velocities = np.zeros(2 * self.N)
        self.final_accelerations = np.zeros(2 * self.N)

    def set_obstacle_trajectories(self, obstacle_trajectories):
        """Set the obstacle trajectories for collision avoidance."""
        self.obstacle_trajectories = obstacle_trajectories

    def precompute_optimization_matrices(self):
        # Constraints on jerk (rate of change of acceleration)
        jerk_matrix = np.zeros((2 * self.N * (self.K - 1), 2 * self.N * self.K))

        for i in range(self.N):
            for k in range(self.K - 1):
                idx_current = 2 * i * self.K + 2 * k
                idx_next = 2 * i * self.K + 2 * (k + 1)

                jerk_matrix[2 * i * (self.K - 1) + 2 * k, idx_current] = -1
                jerk_matrix[2 * i * (self.K - 1) + 2 * k, idx_next] = 1

                jerk_matrix[2 * i * (self.K - 1) + 2 * k + 1, idx_current + 1] = -1
                jerk_matrix[2 * i * (self.K - 1) + 2 * k + 1, idx_next + 1] = 1

        self.A_jerk = jerk_matrix / self.h

        for i in range(self.N):
            for _ in range(self.K - 1):
                self.l_jerk.extend([self.jerk_min[0], self.jerk_min[1]])
                self.u_jerk.extend([self.jerk_max[0], self.jerk_max[1]])

        # Initial acceleration constraints
        for craft in range(self.N):
            front = np.zeros((2, 2 * self.K * craft))
            middle = np.eye(2)
            back = np.zeros((2, 2 * self.K * (self.N - craft) - 2))
            full = np.hstack([front, middle, back])
            self.A_accel_initial = np.vstack((self.A_accel_initial, full))

        for acc in self.initial_accelerations:
            self.l_accel_initial.append(acc)
            self.u_accel_initial.append(acc)

        # Final acceleration constraints
        for craft in range(self.N):
            front = np.zeros((2, 2 * self.K * (craft + 1) - 2))
            middle = np.eye(2)
            back = np.zeros((2, 2 * self.K * (self.N - craft - 1)))
            full = np.hstack([front, middle, back])
            self.A_accel_final = np.vstack((self.A_accel_final, full))

        for acc in self.final_accelerations:
            self.l_accel_final.append(acc)
            self.u_accel_final.append(acc)

        # Acceleration bounds
        self.A_accel_constraints = np.eye(2 * self.N * self.K)
        self.l_accel_constraints = np.array([self.acc_min + self.acc_margin] * 2 * self.N * self.K)
        self.u_accel_constraints = np.array([self.acc_max - self.acc_margin] * 2 * self.N * self.K)

        # Velocity constraints
        vel_transform = np.zeros((2 * (self.K), 2 * (self.K)))

        for k in range(self.K):
            for j in range(k + 1):
                vel_transform[2 * k, 2 * j] = self.h
                vel_transform[2 * k + 1, 2 * j + 1] = self.h

        self.A_vel_constraints = block_diag([vel_transform for _ in range(self.N)]).toarray()

        for i in range(self.N):
            for j in range(2 * self.K - 2):
                self.l_vel_constraints.append(self.vel_min + self.vel_margin)
                self.u_vel_constraints.append(self.vel_max - self.vel_margin)
            self.l_vel_constraints.append(self.final_velocities[2 * i] - self.initial_velocities[2 * i])
            self.l_vel_constraints.append(self.final_velocities[2 * i + 1] - self.initial_velocities[2 * i + 1])
            self.u_vel_constraints.append(self.final_velocities[2 * i] - self.initial_velocities[2 * i])
            self.u_vel_constraints.append(self.final_velocities[2 * i + 1] - self.initial_velocities[2 * i + 1])

        # Position constraints
        pos_transform = np.zeros((2 * self.K, 2 * self.K))

        for k in range(self.K):
            for j in range(k):
                pos_transform[2 * k, 2 * j] = self.h ** 2 * (k - j - 0.5)
                pos_transform[2 * k + 1, 2 * j + 1] = self.h ** 2 * (k - j - 0.5)

        self.A_pos_constraints = block_diag([pos_transform for _ in range(self.N)]).toarray()

        for i in range(self.N):
            for k in range(self.K):
                p_init_x = self.initial_positions[2 * i]
                v_init_x = self.initial_velocities[2 * i]
                p_init_contrib_x = p_init_x + k * self.h * v_init_x

                p_init_y = self.initial_positions[2 * i + 1]
                v_init_y = self.initial_velocities[2 * i + 1]
                p_init_contrib_y = p_init_y + k * self.h * v_init_y

                if k == self.K - 1:
                    self.l_pos_constraints.append(self.final_positions[2 * i] - p_init_contrib_x)
                    self.l_pos_constraints.append(self.final_positions[2 * i + 1] - p_init_contrib_y)
                else:
                    self.l_pos_constraints.append(self.pos_min[0] - p_init_contrib_x)
                    self.l_pos_constraints.append(self.pos_min[1] - p_init_contrib_y)

                if k == self.K - 1:
                    self.u_pos_constraints.append(self.final_positions[2 * i] - p_init_contrib_x)
                    self.u_pos_constraints.append(self.final_positions[2 * i + 1] - p_init_contrib_y)
                else:
                    self.u_pos_constraints.append(self.pos_max[0] - p_init_contrib_x)
                    self.u_pos_constraints.append(self.pos_max[1] - p_init_contrib_y)

    def generate_trajectories(self, max_iterations=20):
        """Generate trajectories - now precomputes matrices and returns trajectories"""
        # Precompute matrices
        self.precompute_optimization_matrices()
        
        # Solve initial guess
        self.calculate_initial_guess()
        
        # Initialize previous solution
        prev_solution = self.accelerations

        # SCP iterations
        iteration = 0
        converged = False
        feasibility_satisfied = False
        
        while iteration < max_iterations and not converged:
            # Get basic constraints
            A = np.vstack([
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
            problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True, max_iter=10000)

            # Warm start with previous solution
            problem.warm_start(x=prev_solution)
            
            # Solve and save result
            result = problem.solve()
            
            # Check for solver errors
            if result.info.status != "solved":
                print(f"Solver failed at iteration {iteration+1}: {result.info.status}")
                return None
            
            # Update solution
            new_solution = result.x
            
            # Check convergence
            solution_diff = np.linalg.norm(new_solution - prev_solution)
            if solution_diff < 0.01:
                converged = True

            positions, velocities = self._accelerations_to_positions_velocities(new_solution)
            avoidance_satisfied = self._fast_check_avoidance_constraints(positions)
            if not avoidance_satisfied:
                converged = False

            if avoidance_satisfied:
                continuous_avoidance_satisfied = self._optimized_check_continuous_avoidance(
                    positions, velocities, new_solution
                )
                if not continuous_avoidance_satisfied and converged and iteration < max_iterations - 1:
                    converged = False

                feasibility_satisfied = self._simple_feasibility_check(positions, velocities, new_solution)

            prev_solution = new_solution
            iteration += 1

        # Save the final solution and trajectories
        self.accelerations = prev_solution
        positions, velocities = self._accelerations_to_positions_velocities(self.accelerations)
        self.trajectories = positions  # Store for visualization
        
        if feasibility_satisfied:
            print(f"SCP trajectory solved successfully after {iteration} iterations.")
            return self.trajectories
        else:
            print(f"SCP trajectory failed to satisfy feasibility constraints after {iteration} iterations.")
            return self.trajectories

    def visualize_trajectories(self, show_animation=False, save_path=None):
        """Visualize the computed trajectories"""
        if self.trajectories is None:
            print("No trajectories available. Call generate_trajectories() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot space boundaries
        ax.plot([self.space_dims[0], self.space_dims[2], self.space_dims[2], self.space_dims[0], self.space_dims[0]],
                [self.space_dims[1], self.space_dims[1], self.space_dims[3], self.space_dims[3], self.space_dims[1]], 
                'k--', alpha=0.5, linewidth=2, label='Boundary')

        # Define colors for different vehicles
        colors = plt.cm.Set1(np.linspace(0, 1, self.N))

        # Plot trajectories
        for i in range(self.N):
            ax.plot(self.trajectories[i, :, 0], self.trajectories[i, :, 1], '-',
                    color=colors[i], linewidth=2, label=f'Vehicle {i + 1}')

            # Plot initial position as large circle
            ax.plot(self.trajectories[i, 0, 0], self.trajectories[i, 0, 1], 'o',
                    color=colors[i], markersize=12, markeredgecolor='black', markeredgewidth=2)

            # Plot final position as X
            ax.plot(self.trajectories[i, -1, 0], self.trajectories[i, -1, 1], 'x',
                    color=colors[i], markersize=15, markeredgewidth=3)

        # Set labels and title
        ax.set_xlabel('X Position [m]', fontsize=12)
        ax.set_ylabel('Y Position [m]', fontsize=12)
        ax.set_title('Multi-Vehicle Collision-Free Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axis('equal')

        # Add text annotations
        ax.text(0.02, 0.98, f'Vehicles: {self.N}\nTime: {self.T}s\nMin Distance: {self.R}m', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved as {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_time_snapshots(self, num_snapshots=5, save_path=None):
        """Visualize vehicle positions at different time snapshots"""
        if self.trajectories is None:
            print("No trajectories available. Call generate_trajectories() first.")
            return

        fig, axes = plt.subplots(1, num_snapshots, figsize=(4*num_snapshots, 4))
        if num_snapshots == 1:
            axes = [axes]

        # Select time indices for snapshots
        time_indices = np.linspace(0, self.K-1, num_snapshots, dtype=int)
        colors = plt.cm.Set1(np.linspace(0, 1, self.N))

        for snapshot_idx, time_idx in enumerate(time_indices):
            ax = axes[snapshot_idx]
            
            # Plot space boundaries
            ax.plot([self.space_dims[0], self.space_dims[2], self.space_dims[2], self.space_dims[0], self.space_dims[0]],
                    [self.space_dims[1], self.space_dims[1], self.space_dims[3], self.space_dims[3], self.space_dims[1]], 
                    'k--', alpha=0.5)

            # Plot vehicle positions at this time
            for i in range(self.N):
                ax.plot(self.trajectories[i, time_idx, 0], self.trajectories[i, time_idx, 1], 
                       'o', color=colors[i], markersize=10, label=f'Vehicle {i+1}')
                
                # Draw minimum distance circles
                circle = plt.Circle((self.trajectories[i, time_idx, 0], self.trajectories[i, time_idx, 1]), 
                                  self.R, fill=False, color=colors[i], alpha=0.3, linestyle='--')
                ax.add_patch(circle)

            ax.set_xlim(self.space_dims[0]-1, self.space_dims[2]+1)
            ax.set_ylim(self.space_dims[1]-1, self.space_dims[3]+1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f't = {time_idx * self.h:.1f}s')
            
            if snapshot_idx == 0:
                ax.set_ylabel('Y Position [m]')
            ax.set_xlabel('X Position [m]')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time snapshots saved as {save_path}")
        plt.show()

    # Include all the helper methods from the original code
    def _fast_check_avoidance_constraints(self, positions):
        for k in range(self.K):
            pos_k = positions[:, k, :]
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist = np.linalg.norm(pos_k[i] - pos_k[j])
                    if dist < self.R - 0.01:
                        return False
        return True

    def _simple_feasibility_check(self, positions, velocities, accelerations):
        for k in range(self.K):
            for i in range(self.N):
                if np.linalg.norm(velocities[i, k]) > self.vel_max:
                    return False
                if np.linalg.norm(accelerations[2*i*self.K + 2*k]) > self.acc_max:
                    return False
        return True

    def _optimized_check_continuous_avoidance(self, positions, velocities, accelerations, num_samples=5):
        for k in range(self.K - 1):
            potential_collisions = []
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist_k = np.linalg.norm(positions[i, k] - positions[j, k])
                    dist_k1 = np.linalg.norm(positions[i, k+1] - positions[j, k+1])
                    threshold = self.R + self.h * 5.0
                    if dist_k < threshold or dist_k1 < threshold or dist_k1 < dist_k:
                        potential_collisions.append((i, j))

            if not potential_collisions:
                continue

            for i, j in potential_collisions:
                for s in range(1, num_samples):
                    fraction = s / num_samples
                    dt = self.h * fraction

                    p0_i = positions[i, k]
                    v0_i = velocities[i, k]
                    a0_i = accelerations[2*i*self.K + 2*k]
                    pos_i = p0_i + v0_i * dt + 0.5 * a0_i * dt**2

                    p0_j = positions[j, k]
                    v0_j = velocities[j, k]
                    a0_j = accelerations[2*j*self.K + 2*k]
                    pos_j = p0_j + v0_j * dt + 0.5 * a0_j * dt**2

                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < self.R - 0.01:
                        return False
        return True

    def calculate_initial_guess(self):
        problem = osqp.OSQP()
        P = np.eye(2 * self.N * self.K)
        A = np.vstack([self.A_jerk, self.A_accel_initial, self.A_accel_final, 
                       self.A_accel_constraints, self.A_vel_constraints, self.A_pos_constraints])
        l = np.hstack([self.l_jerk, self.l_accel_initial, self.l_accel_final, 
                       self.l_accel_constraints, self.l_vel_constraints, self.l_pos_constraints])
        u = np.hstack([self.u_jerk, self.u_accel_initial, self.u_accel_final, 
                       self.u_accel_constraints, self.u_vel_constraints, self.u_pos_constraints])
        q = np.array([0] * 2 * self.N * self.K)

        P = sp.csc_matrix(P)
        A = sp.csc_matrix(A)
        problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        results = problem.solve()
        self.accelerations = results.x

    def add_collision_constraints(self, previous_solution=None):
        prev_positions, _ = self._accelerations_to_positions_velocities(previous_solution)
        n_pairs = (self.N * (self.N - 1)) // 2
        n_vehicle_constraints = n_pairs * self.K
        
        n_obstacle_constraints = 0
        if self.obstacle_trajectories is not None:
            n_obstacles = len(self.obstacle_trajectories)
            n_obstacle_constraints = self.N * n_obstacles * self.K
        
        n_constraints = n_vehicle_constraints + n_obstacle_constraints
        A_collision = np.zeros((n_constraints, 2 * self.N * self.K))
        l_collision = np.ones(n_constraints) * (self.R + self.R_margin)
        u_collision = np.ones(n_constraints) * np.inf

        row_idx = 0
        for k in range(self.K):
            for i in range(self.N):
                for j in range(i+1, self.N):
                    p_i_prev = prev_positions[i, k]
                    p_j_prev = prev_positions[j, k]
                    diff_vector = p_i_prev - p_j_prev
                    dist = np.linalg.norm(diff_vector)

                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2*np.pi)
                        diff_vector = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0

                    eta = diff_vector / dist

                    for m in range(k):
                        A_collision[row_idx, 2*i*self.K + 2*m] = eta[0] * self.h**2 * (k - m - 0.5)
                        A_collision[row_idx, 2*i*self.K + 2*m + 1] = eta[1] * self.h**2 * (k - m - 0.5)
                        A_collision[row_idx, 2*j*self.K + 2*m] = -eta[0] * self.h**2 * (k - m - 0.5)
                        A_collision[row_idx, 2*j*self.K + 2*m + 1] = -eta[1] * self.h**2 * (k - m - 0.5)

                    p_init_i_x = self.initial_positions[2*i]
                    p_init_i_y = self.initial_positions[2*i+1]
                    v_init_i_x = self.initial_velocities[2*i]
                    v_init_i_y = self.initial_velocities[2*i+1]

                    p_init_j_x = self.initial_positions[2*j]
                    p_init_j_y = self.initial_positions[2*j+1]
                    v_init_j_x = self.initial_velocities[2*j]
                    v_init_j_y = self.initial_velocities[2*j+1]

                    init_pos_contrib = eta[0] * (p_init_i_x - p_init_j_x) + eta[1] * (p_init_i_y - p_init_j_y)
                    init_vel_contrib = eta[0] * (v_init_i_x - v_init_j_x) + eta[1] * (v_init_i_y - v_init_j_y)
                    init_vel_contrib *= k * self.h

                    linearization_term = eta[0] * (p_i_prev[0] - p_j_prev[0]) + eta[1] * (p_i_prev[1] - p_j_prev[1]) - dist
                    rhs = self.R + linearization_term - (init_pos_contrib + init_vel_contrib)
                    l_collision[row_idx] = rhs

                    row_idx += 1

        if self.obstacle_trajectories is not None:
            for k in range(self.K):
                for i in range(self.N):
                    for obs_idx in range(len(self.obstacle_trajectories)):
                        p_i_prev = prev_positions[i, k]
                        p_obs = self.obstacle_trajectories[obs_idx, k]
                        diff_vector = p_i_prev - p_obs
                        dist = np.linalg.norm(diff_vector)
                        
                        if dist < 1e-6:
                            angle = np.random.uniform(0, 2*np.pi)
                            diff_vector = np.array([np.cos(angle), np.sin(angle)])
                            dist = 1.0
                        
                        eta = diff_vector / dist
                        
                        for m in range(k):
                            A_collision[row_idx, 2*i*self.K + 2*m] = eta[0] * self.h**2 * (k - m - 0.5)
                            A_collision[row_idx, 2*i*self.K + 2*m + 1] = eta[1] * self.h**2 * (k - m - 0.5)
                        
                        p_init_i_x = self.initial_positions[2*i]
                        p_init_i_y = self.initial_positions[2*i+1]
                        v_init_i_x = self.initial_velocities[2*i]
                        v_init_i_y = self.initial_velocities[2*i+1]
                        
                        init_pos_contrib = eta[0] * p_init_i_x + eta[1] * p_init_i_y
                        init_vel_contrib = (eta[0] * v_init_i_x + eta[1] * v_init_i_y) * k * self.h
                        obs_contrib = eta[0] * p_obs[0] + eta[1] * p_obs[1]
                        linearization_term = np.dot(eta, p_i_prev) - np.dot(eta, p_obs) - dist
                        
                        rhs = self.R + linearization_term - (init_pos_contrib + init_vel_contrib) + obs_contrib
                        l_collision[row_idx] = rhs
                        
                        row_idx += 1
        
        return A_collision, l_collision, u_collision

    def _position_index(self, vehicle_idx, time_step, coord):
        """Helper method to get position index for a vehicle at time step"""
        return 2 * vehicle_idx * self.K + 2 * time_step + coord

    def _accelerations_to_positions_velocities(self, accelerations):
        """Convert acceleration solution to positions for all vehicles and time steps"""
        N = self.N
        K = self.K
        h = self.h

        positions = np.zeros((N, K, 2))
        velocities = np.zeros((N, K, 2))

        # Set initial positions and velocities
        for i in range(N):
            positions[i, 0, 0] = self.initial_positions[2 * i]
            positions[i, 0, 1] = self.initial_positions[2 * i + 1]
            velocities[i, 0, 0] = self.initial_velocities[2 * i]
            velocities[i, 0, 1] = self.initial_velocities[2 * i + 1]

        # Reconstruct trajectories for each vehicle
        for i in range(N):
            for k in range(1, K):
                ax = accelerations[2 * i * K + 2 * (k - 1)]
                ay = accelerations[2 * i * K + 2 * (k - 1) + 1]

                velocities[i, k, 0] = velocities[i, k - 1, 0] + h * ax
                velocities[i, k, 1] = velocities[i, k - 1, 1] + h * ay

                positions[i, k, 0] = positions[i, k - 1, 0] + h * velocities[i, k - 1, 0] + 0.5 * h ** 2 * ax
                positions[i, k, 1] = positions[i, k - 1, 1] + h * velocities[i, k - 1, 1] + 0.5 * h ** 2 * ay

        return positions, velocities

    def plot_positions(self, save_path=None):
        """Plot the positions of all vehicles - kept for backward compatibility"""
        self.visualize_trajectories(save_path=save_path)


# Utility functions from the original code
def adjust_paths_with_crossings(start_coords, end_coords, reg_term=1e-1):
    """
    Check if any robot's path crosses another robot's starting point.
    If it does, add a small regularization term to the end position.
    """
    n = len(start_coords)

    for i in range(n):
        start = start_coords[i][:2]
        end = end_coords[i][:2]

        direction = end - start
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            unit_direction = direction / direction_norm
            normal_direction = np.array([-unit_direction[1], unit_direction[0]])
        else:
            continue

        path_adjusted = False

        for j in range(n):
            if i == j:
                continue

            point = start_coords[j][:2]
            start_to_point = point - start
            projection_length = np.dot(start_to_point, unit_direction)
            closest_point = start + projection_length * unit_direction
            point_to_line_distance = np.linalg.norm(point - closest_point)
            on_segment = (0 <= projection_length <= direction_norm) and (point_to_line_distance < 1e-10)

            if on_segment:
                path_adjusted = True

        if path_adjusted:
            adjustment = reg_term * normal_direction
            end_coords[i][:2] = end + adjustment

    return end_coords


# Example usage that matches the original main function structure
if __name__ == "__main__":
    initial_states = np.array([[0.0, 0.0, 0, 0, 0, 0], [5.0, 5.0, 0, 0, 0, 0], [4.0, 1.0, 0, 0, 0, 0]])
    final_states = np.array([[8.0, 8.0, 0, 0, 0, 0], [0.0, 0.0, 0, 0, 0, 0], [1.0, 3.0, 0, 0, 0, 0]])

    n_vehicles = len(initial_states)
    solver = SCP(n_vehicles=n_vehicles, time_horizon=40)

    # Extract just the positions for the new interface
    initial_positions = initial_states[:, :2]
    final_positions = final_states[:, :2]

    solver.set_initial_states(initial_positions)
    solver.set_final_states(final_positions)

    trajectories = solver.generate_trajectories()
    if trajectories is not None:
        print("Trajectory generation successful")
        solver.visualize_trajectories()
    else:
        print("Trajectory generation failed")