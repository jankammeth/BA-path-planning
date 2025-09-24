#scp_solver_sebastian.py

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.sparse import block_diag

import matplotlib.pyplot as plt

class SCPSolver():
    def __init__(self,
             n_vehicles = 5,  # Number of vehicles
             time_horizon = 10,  # Total time T for trajectory
             time_step = 0.1,  # Discretization time step h
             discrete_num_steps = 10,
             min_distance = 1.0,  # Minimum distance between vehicles R
             space_dims=[-10, -10, 10, 10],  # Dimensions of the space [x_min, y_min, x_max, y_max]
             ):

        self.N = n_vehicles
        self.T = time_horizon
        self.h = time_step
        self.K = discrete_num_steps  # Number of discrete time steps + 1 for initial timestep
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

        # Acceleration limit(the same in x and y)
        self.acc_min = -15.0
        self.acc_max = 15.0
        self.acc_margin = 1.0

        # Jerk limits in x, y
        self.jerk_min = np.array([-15, -15])
        self.jerk_max = np.array([15, 15])

        # Initialization of initial and end variables
        self.initial_positions = None
        self.initial_velocities = None
        self.initial_accelerations = None
        self.final_positions = None
        self.final_velocities = None
        self.final_accelerations = None

        # Initialization of optimization variable
        self.accelerations = np.array([None] * 2 * self.N * self.K)
        
        # Obstacle trajectories
        self.obstacle_trajectories = None  # shape (n_obstacles, K, 2)

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

    def set_initial_states(self, initial_states):
        self.initial_positions = initial_states[:, 0:2].flatten()
        self.initial_velocities = initial_states[:, 2:4].flatten()
        self.initial_accelerations = initial_states[:, 4:6].flatten()

        assert len(self.initial_positions) == len(self.initial_velocities) == len(self.initial_accelerations)

    def set_finial_states(self, final_states):
        self.final_positions = final_states[:, 0:2].flatten()
        self.final_velocities = final_states[:, 2:4].flatten()
        self.final_accelerations = final_states[:, 4:6].flatten()
        assert len(self.final_positions) == len(self.final_velocities) == len(self.final_accelerations)
    
    def set_obstacle_trajectories(self, obstacle_trajectories):
        """Set the obstacle trajectories for collision avoidance."""
        self.obstacle_trajectories = obstacle_trajectories

    def precompute_optimization_matrices(self):
        # Constraints on jerk (rate of change of acceleration)
        jerk_matrix = np.zeros((2 * self.N * (self.K - 1), 2 * self.N * self.K))

        for i in range(self.N):  # For each vehicle
            for k in range(self.K - 1):  # For each time step except the last
                # Current acceleration index
                idx_current = 2 * i * self.K + 2 * k
                # Next acceleration index
                idx_next = 2 * i * self.K + 2 * (k + 1)

                # Set up x-component jerk constraint (ax[k+1] - ax[k])
                jerk_matrix[2 * i * (self.K - 1) + 2 * k, idx_current] = -1
                jerk_matrix[2 * i * (self.K - 1) + 2 * k, idx_next] = 1

                # Set up y-component jerk constraint (ay[k+1] - ay[k])
                jerk_matrix[2 * i * (self.K - 1) + 2 * k + 1, idx_current + 1] = -1
                jerk_matrix[2 * i * (self.K - 1) + 2 * k + 1, idx_next + 1] = 1

        # Scale by time step to get proper units
        self.A_jerk = jerk_matrix / self.h

        for i in range(self.N):
            for _ in range(self.K - 1):
                self.l_jerk.extend([self.jerk_min[0], self.jerk_min[1]])
                self.u_jerk.extend([self.jerk_max[0], self.jerk_max[1]])

        # Constraints on initial acceleration
        for craft in range(self.N):
            front = np.zeros((2, 2 * self.K * craft))
            middle = np.eye(2)
            back = np.zeros((2, 2 * self.K * (self.N - craft) - 2))
            full = np.hstack([front, middle, back])
            self.A_accel_initial = np.vstack((self.A_accel_initial, full))

        for acc in self.initial_accelerations:
            self.l_accel_initial.append(acc)
            self.u_accel_initial.append(acc)

        # Constraints on final acceleration
        for craft in range(self.N):
            front = np.zeros((2, 2 * self.K * (craft + 1) - 2))
            middle = np.eye(2)
            back = np.zeros((2, 2 * self.K * (self.N - craft - 1)))
            full = np.hstack([front, middle, back])
            self.A_accel_final = np.vstack((self.A_accel_final, full))

        for acc in self.final_accelerations:
            self.l_accel_final.append(acc)
            self.u_accel_final.append(acc)

        # Constraints on acceleration
        self.A_accel_constraints = np.eye(2 * self.N * self.K)
        self.l_accel_constraints = np.array([self.acc_min + self.acc_margin] * 2 * self.N * self.K)
        self.u_accel_constraints = np.array([self.acc_max - self.acc_margin] * 2 * self.N * self.K)

        # Constraints on velocities
        vel_transform = np.zeros((2 * (self.K), 2 * (self.K)))

        for k in range(self.K):
            for j in range(k + 1):
                # x component
                vel_transform[2 * k, 2 * j] = self.h
                # y component
                vel_transform[2 * k + 1, 2 * j + 1] = self.h

        self.A_vel_constraints = block_diag([
            vel_transform for _ in range(self.N)
        ]).toarray()

        # check for bound on intermediate velocities and final velocity equality on last two vx and vy per craft
        for i in range(self.N):
            for j in range(2 * self.K - 2):
                self.l_vel_constraints.append(self.vel_min + self.vel_margin)
                self.u_vel_constraints.append(self.vel_max - self.vel_margin)
            self.l_vel_constraints.append(self.final_velocities[2 * i] - self.initial_velocities[2 * i])
            self.l_vel_constraints.append(self.final_velocities[2 * i + 1] - self.initial_velocities[2 * i + 1])
            self.u_vel_constraints.append(self.final_velocities[2 * i] - self.initial_velocities[2 * i])
            self.u_vel_constraints.append(self.final_velocities[2 * i + 1] - self.initial_velocities[2 * i + 1])

        # constraints on position
        pos_transform = np.zeros((2 * self.K, 2 * self.K))

        for k in range(self.K):
            for j in range(k):
                # x component transformation
                pos_transform[2 * k, 2 * j] = self.h ** 2 * (k - j - 0.5)
                # y component transformation
                pos_transform[2 * k + 1, 2 * j + 1] = self.h ** 2 * (k - j - 0.5)

        # Position constraints matrix (same block diagonal structure as before)
        self.A_pos_constraints = block_diag([
            pos_transform for _ in range(self.N)
        ]).toarray()

        # Position lower and upper bounds
        for i in range(self.N):
            for k in range(self.K):
                # Initial position + velocity contribution for x
                p_init_x = self.initial_positions[2 * i]
                v_init_x = self.initial_velocities[2 * i]
                p_init_contrib_x = p_init_x + k * self.h * v_init_x

                # Initial position + velocity contribution for y
                p_init_y = self.initial_positions[2 * i + 1]
                v_init_y = self.initial_velocities[2 * i + 1]
                p_init_contrib_y = p_init_y + k * self.h * v_init_y

                # Lower bounds: either space boundary or final position constraint
                if k == self.K - 1:  # Final position constraint
                    self.l_pos_constraints.append(self.final_positions[2 * i] - p_init_contrib_x)
                    self.l_pos_constraints.append(self.final_positions[2 * i + 1] - p_init_contrib_y)
                else:  # Space boundary
                    self.l_pos_constraints.append(self.pos_min[0] - p_init_contrib_x)
                    self.l_pos_constraints.append(self.pos_min[1] - p_init_contrib_y)

                # Upper bounds: either space boundary or final position constraint
                if k == self.K - 1:  # Final position constraint
                    self.u_pos_constraints.append(self.final_positions[2 * i] - p_init_contrib_x)
                    self.u_pos_constraints.append(self.final_positions[2 * i + 1] - p_init_contrib_y)
                else:  # Space boundary
                    self.u_pos_constraints.append(self.pos_max[0] - p_init_contrib_x)
                    self.u_pos_constraints.append(self.pos_max[1] - p_init_contrib_y)

    def generate_trajectories(self, max_iterations = 20):

        # solve linear interpolation for initial guess of optimization variable
        self.calculate_initial_guess()
        self.plot_positions()
        # Initialize previous solution
        prev_solution = self.accelerations

        # SCP iterations
        iteration = 0
        converged = False
        feasibility_satisfied = False
        while iteration < max_iterations and not converged:
            #print(f"SCP iteration {iteration+1}/{max_iterations}")
            
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

    def _fast_check_avoidance_constraints(self, positions):
        """
        Optimized version that checks if avoidance constraints are satisfied
        at discrete timesteps. Uses vectorized operations for speed.
        """
        for k in range(self.K):
            # Get positions for all vehicles at time k
            pos_k = positions[:, k, :]  # Shape (N, 2)

            # Compute pairwise differences
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist = np.linalg.norm(pos_k[i] - pos_k[j])
                    if dist < self.R - 0.01:
                        print(f"Avoidance constraint violation at timestep {k} between vehicles {i} and {j}: distance = {dist:.3f}")
                        return False
        return True

    def _simple_feasibility_check(self, positions, velocities, accelerations):
        for k in range(self.K):
            for i in range(self.N):
                if np.linalg.norm(velocities[i, k]) > self.vel_max:
                    print(f"Velocity constraint violation at timestep {k} for vehicle {i}: velocity norm = {np.linalg.norm(velocities[i, k]):.3f}")
                    return False
                if np.linalg.norm(accelerations[2*i*self.K + 2*k]) > self.acc_max:
                    print(f"Acceleration constraint violation at timestep {k} for vehicle {i}: acceleration norm = {np.linalg.norm(accelerations[2*i*self.K + 2*k]):.3f}")
                    return False

        return True

    def _optimized_check_continuous_avoidance(self, positions, velocities, accelerations, num_samples=5):
        """
        Optimized version that checks if avoidance constraints are satisfied between timesteps
        by sampling intermediate points. Uses vectorization where possible.

        Only checks timesteps and vehicle pairs that might have issues based on proximity.
        """
        # Only check timestep pairs where vehicles are close enough to possibly collide
        for k in range(self.K - 1):
            # First, check if any vehicles are close at timesteps k and k+1
            potential_collisions = []

            for i in range(self.N):
                for j in range(i+1, self.N):
                    # Distance at timestep k
                    dist_k = np.linalg.norm(positions[i, k] - positions[j, k])
                    # Distance at timestep k+1
                    dist_k1 = np.linalg.norm(positions[i, k+1] - positions[j, k+1])

                    # If either distance is less than R + safety buffer, or if they're
                    # getting closer, then check intermediate timesteps
                    threshold = self.R + self.h * 5.0  # Safety buffer based on max velocity
                    if dist_k < threshold or dist_k1 < threshold or dist_k1 < dist_k:
                        potential_collisions.append((i, j))

            # Skip this timestep if no potential collisions
            if not potential_collisions:
                continue

            # For each potential collision, check intermediate points
            for i, j in potential_collisions:
                for s in range(1, num_samples):
                    fraction = s / num_samples

                    # Compute interpolated positions using quadratic approximation
                    dt = self.h * fraction

                    # Position of vehicle i at intermediate time
                    p0_i = positions[i, k]
                    v0_i = velocities[i, k]
                    a0_i = accelerations[2*i*self.K + 2*k]
                    pos_i = p0_i + v0_i * dt + 0.5 * a0_i * dt**2

                    # Position of vehicle j at intermediate time
                    p0_j = positions[j, k]
                    v0_j = velocities[j, k]
                    a0_j = accelerations[2*i*self.K + 2*k]
                    pos_j = p0_j + v0_j * dt + 0.5 * a0_j * dt**2

                    # Check for collision
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < self.R - 0.01:
                        print(f"Continuous avoidance constraint violation between timesteps {k} and {k+1} at fraction {fraction:.2f}")
                        print(f"Between vehicles {i} and {j}: distance = {dist:.3f}")
                        return False

        return True

    def calculate_initial_guess(self):
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

    def add_collision_constraints(self, previous_solution=None):
        """
        Add linearized collision avoidance constraints based on previous solution.
        If no previous solution is provided, uses a straight-line trajectory.
        
        This includes:
        1. Constraints between optimized crafts
        2. Constraints between optimized crafts and obstacles

        The constraint ensures:
            ||p_i[k] - p_j[k]||_2 >= R

        Linearized as:
            ||p_i^q[k] - p_j^q[k]||_2 + η^T[(p_i[k] - p_j[k]) - (p_i^q[k] - p_j^q[k])] >= R

        where η = (p_i^q[k] - p_j^q[k]) / ||p_i^q[k] - p_j^q[k]||_2
        """
        #print("Calculation of the collision constraints...")
        prev_positions, _ = self._accelerations_to_positions_velocities(previous_solution)

        # Number of collision constraints:
        # 1. Between vehicles: one for each pair of vehicles at each time step
        n_pairs = (self.N * (self.N - 1)) // 2  # Number of vehicle pairs
        n_vehicle_constraints = n_pairs * self.K
        
        # 2. Between vehicles and obstacles
        n_obstacle_constraints = 0
        if self.obstacle_trajectories is not None:
            n_obstacles = len(self.obstacle_trajectories)
            n_obstacle_constraints = self.N * n_obstacles * self.K
        
        n_constraints = n_vehicle_constraints + n_obstacle_constraints

        # Initialize constraint matrix and bounds
        A_collision = np.zeros((n_constraints, 2 * self.N * self.K))
        l_collision = np.ones(n_constraints) * (self.R + self.R_margin)
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

                    # If vehicles are very close or colliding in previous solution,
                    # adjust the vector to avoid numerical issues
                    if dist < 1e-6:
                        # Use a random unit vector if previous positions are identical
                        angle = np.random.uniform(0, 2*np.pi)
                        diff_vector = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0

                    # Compute normalized direction vector η
                    eta = diff_vector / dist

                    # Construct the constraint row:
                    # η^T * (p_i[k] - p_j[k]) >= R + η^T * (p_i^q[k] - p_j^q[k])

                    # For vehicle i position at time step k
                    pos_i_idx_x = self._position_index(i, k, 0)
                    pos_i_idx_y = self._position_index(i, k, 1)

                    # For vehicle j position at time step k
                    pos_j_idx_x = self._position_index(j, k, 0)
                    pos_j_idx_y = self._position_index(j, k, 1)

                    # Set coefficients in constraint matrix
                    # We need to convert from position indices to acceleration indices
                    # This is where we use the position transformation matrix
                    for m in range(k):
                        # x-component coefficients for vehicle i
                        A_collision[row_idx, 2*i*self.K + 2*m] = eta[0] * self.h**2 * (k - m - 0.5)
                        # y-component coefficients for vehicle i
                        A_collision[row_idx, 2*i*self.K + 2*m + 1] = eta[1] * self.h**2 * (k - m - 0.5)

                        # x-component coefficients for vehicle j (negative because of p_i - p_j)
                        A_collision[row_idx, 2*j*self.K + 2*m] = -eta[0] * self.h**2 * (k - m - 0.5)
                        # y-component coefficients for vehicle j
                        A_collision[row_idx, 2*j*self.K + 2*m + 1] = -eta[1] * self.h**2 * (k - m - 0.5)

                    # Adjust the right-hand side:
                    # R + η^T * (p_i^q[k] - p_j^q[k]) - η^T * (p_init_i + v_init_i*k*h - p_init_j - v_init_j*k*h)
                    p_init_i_x = self.initial_positions[2*i]
                    p_init_i_y = self.initial_positions[2*i+1]
                    v_init_i_x = self.initial_velocities[2*i]
                    v_init_i_y = self.initial_velocities[2*i+1]

                    p_init_j_x = self.initial_positions[2*j]
                    p_init_j_y = self.initial_positions[2*j+1]
                    v_init_j_x = self.initial_velocities[2*j]
                    v_init_j_y = self.initial_velocities[2*j+1]

                    # Initial position contribution: η^T * (p_init_i - p_init_j)
                    init_pos_contrib = eta[0] * (p_init_i_x - p_init_j_x) + eta[1] * (p_init_i_y - p_init_j_y)

                    # Initial velocity contribution: η^T * (v_init_i - v_init_j) * k * h
                    init_vel_contrib = eta[0] * (v_init_i_x - v_init_j_x) + eta[1] * (v_init_i_y - v_init_j_y)
                    init_vel_contrib *= k * self.h

                    # Constant term from linearization
                    linearization_term = eta[0] * (p_i_prev[0] - p_j_prev[0]) + eta[1] * (
                                p_i_prev[1] - p_j_prev[1]) - dist
                    # The right-hand side becomes:
                    # R + η^T * (p_i^q[k] - p_j^q[k]) - η^T * (p_init_i + v_init_i*k*h - p_init_j - v_init_j*k*h)
                    rhs = self.R + linearization_term - (init_pos_contrib + init_vel_contrib)
                    l_collision[row_idx] = rhs

                    row_idx += 1

        #print("Finished calculation of colision Constraints")
        # Add constraints between vehicles and obstacles
        if self.obstacle_trajectories is not None:
            for k in range(self.K):
                for i in range(self.N):
                    for obs_idx in range(len(self.obstacle_trajectories)):
                        # Get vehicle's previous position
                        p_i_prev = prev_positions[i, k]
                        
                        # Get obstacle position at time k
                        p_obs = self.obstacle_trajectories[obs_idx, k]
                        
                        # Compute difference and distance
                        diff_vector = p_i_prev - p_obs
                        dist = np.linalg.norm(diff_vector)
                        
                        # Handle numerical issues
                        if dist < 1e-6:
                            angle = np.random.uniform(0, 2*np.pi)
                            diff_vector = np.array([np.cos(angle), np.sin(angle)])
                            dist = 1.0
                        
                        # Compute normalized direction vector η
                        eta = diff_vector / dist
                        
                        # Set coefficients in constraint matrix for vehicle i
                        for m in range(k):
                            # x-component coefficients for vehicle i
                            A_collision[row_idx, 2*i*self.K + 2*m] = eta[0] * self.h**2 * (k - m - 0.5)
                            # y-component coefficients for vehicle i  
                            A_collision[row_idx, 2*i*self.K + 2*m + 1] = eta[1] * self.h**2 * (k - m - 0.5)
                        
                        # Compute right-hand side
                        p_init_i_x = self.initial_positions[2*i]
                        p_init_i_y = self.initial_positions[2*i+1]
                        v_init_i_x = self.initial_velocities[2*i]
                        v_init_i_y = self.initial_velocities[2*i+1]
                        
                        # Initial position/velocity contribution
                        init_pos_contrib = eta[0] * p_init_i_x + eta[1] * p_init_i_y
                        init_vel_contrib = (eta[0] * v_init_i_x + eta[1] * v_init_i_y) * k * self.h
                        
                        # Obstacle position contribution
                        obs_contrib = eta[0] * p_obs[0] + eta[1] * p_obs[1]
                        
                        # Linearization term
                        linearization_term = np.dot(eta, p_i_prev) - np.dot(eta, p_obs) - dist
                        
                        # Right-hand side
                        rhs = self.R + linearization_term - (init_pos_contrib + init_vel_contrib) + obs_contrib
                        l_collision[row_idx] = rhs
                        
                        row_idx += 1
        
        # Return the constraint matrices and bounds
        return A_collision, l_collision, u_collision

    def _position_index(self, vehicle_idx, time_step, coord):
        """Helper method to get position index for a vehicle at time step"""
        return 2 * vehicle_idx * self.K + 2 * time_step + coord

    def _accelerations_to_positions_velocities(self, accelerations):
        """
        Convert acceleration solution to positions for all vehicles and time steps
        """

        N = self.N  # Number of vehicles
        K = self.K  # Number of time steps
        h = self.h  # Time step

        # Initialize arrays
        positions = np.zeros((N, K, 2))  # N vehicles, K time steps, 2 dimensions (x,y)
        velocities = np.zeros((N, K, 2))

        # Set initial positions and velocities
        for i in range(N):
            positions[i, 0, 0] = self.initial_positions[2 * i]  # x position
            positions[i, 0, 1] = self.initial_positions[2 * i + 1]  # y position
            velocities[i, 0, 0] = self.initial_velocities[2 * i]  # x velocity
            velocities[i, 0, 1] = self.initial_velocities[2 * i + 1]  # y velocity

        # Reconstruct trajectories for each vehicle
        for i in range(N):
            for k in range(1, K):
                # Extract accelerations for this vehicle at previous time step
                ax = accelerations[2 * i * K + 2 * (k - 1)]
                ay = accelerations[2 * i * K + 2 * (k - 1) + 1]

                # Update velocity using acceleration
                velocities[i, k, 0] = velocities[i, k - 1, 0] + h * ax
                velocities[i, k, 1] = velocities[i, k - 1, 1] + h * ay

                # Update position using velocity and acceleration
                positions[i, k, 0] = positions[i, k - 1, 0] + h * velocities[
                    i, k - 1, 0] + 0.5 * h ** 2 * ax
                positions[i, k, 1] = positions[i, k - 1, 1] + h * velocities[
                    i, k - 1, 1] + 0.5 * h ** 2 * ay

        return positions, velocities

    def check_if_lines_cross_starting_points(self, start_coords, end_coords):
        """
        Check if any robot's path crosses another robot's starting point.

        Parameters:
        start_coords: numpy array of shape (n, 1) containing start coordinates of n robots
        end_coords: numpy array of shape (n, 1) containing end coordinates of n robots

        Returns:
        List of tuples (i, j) where robot i's path crosses robot j's starting point
        """
        n = len(start_coords)
        crossings = []

        for i in range(n):  # For each robot's path
            start = start_coords[i]
            end = end_coords[i]

            # Calculate direction vector of the path
            direction = end - start

            for j in range(n):  # Check against all other robots' starting points
                if i == j:  # Skip comparing with itself
                    continue

                # Get the starting point of the other robot
                point = start_coords[j]

                # Calculate vector from start of path to the point
                start_to_point = point - start

                # Check if the point lies on the line segment
                # Using parametric representation: start + t * direction = point
                # where -1 <= t <= 1 if point is on the line segment

                # Handle case where direction components are zero (avoid division by zero)
                if direction[-1] == 0 and direction[1] == 0:
                    # Start and end are the same, so we check if they match the point
                    if np.array_equal(start, point):
                        crossings.append((i, j))
                    continue

                # Calculate parameter t for both x and y components
                if direction[-1] != 0:
                    t_x = start_to_point[-1] / direction[0]
                else:
                    # If direction.x is -1, check if point.x equals start.x
                    t_x = -1 if start_to_point[0] == 0 else float('inf')

                if direction[0] != 0:
                    t_y = start_to_point[0] / direction[1]
                else:
                    # If direction.y is -1, check if point.y equals start.y
                    t_y = -1 if start_to_point[1] == 0 else float('inf')

                # If point is on the line, t_x and t_y must be equal
                # Also, t must be between -1 and 1 for the point to be on the line segment
                if abs(t_x - t_y) < 0e-10 and 0 <= t_x <= 1:
                    crossings.append((i, j))

        return crossings

    def plot_positions(self, save_path=None):
        """
        Plot the positions of all vehicles calculated from accelerations

        Args:
            save_path: Optional path to save the figure
        """
        if self.accelerations is None:
            print("No accelerations available. Call generate_trajectories() first.")
            return

        # Convert accelerations to positions
        positions, _ = self._accelerations_to_positions_velocities(self.accelerations)

        # Create figure and axis objects explicitly
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot space boundaries
        ax.plot([self.space_dims[0], self.space_dims[2],
                 self.space_dims[2], self.space_dims[0],
                 self.space_dims[0]],
                [self.space_dims[1], self.space_dims[1],
                 self.space_dims[3], self.space_dims[3],
                 self.space_dims[1]], 'k--', alpha=0.3)

        # Define colors for different vehicles
        colors = plt.cm.jet(np.linspace(0, 1, self.N))

        # Plot trajectories
        for i in range(self.N):
            ax.plot(positions[i, :, 0], positions[i, :, 1], '-',
                    color=colors[i], label=f'Vehicle {i + 1}')

            # Plot initial position as circle
            ax.plot(positions[i, 0, 0], positions[i, 0, 1], 'o',
                    color=colors[i], markersize=8)

            # Plot final position as X
            ax.plot(positions[i, -1, 0], positions[i, -1, 1], 'x',
                    color=colors[i], markersize=8)

            # Add time annotations at select points
            if self.K > 10:
                # Add time markers at a few points along the trajectory
                step = max(1, self.K // 5)  # Show about 5 time markers
                for k in range(0, self.K, step):
                    ax.text(positions[i, k, 0], positions[i, k, 1],
                            f'{k * self.h:.1f}s', fontsize=8)

        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vehicle Trajectories')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        # Add colorbar to show time progression - fix by specifying the axis
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(0, self.T))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)  # Note the change here - pass ax to fig.colorbar
        cbar.set_label('Time (s)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


def adjust_paths_with_crossings(start_coords, end_coords, reg_term=1e-1):
    """
    Check if any robot's path crosses another robot's starting point.
    If it does, add a small regularization term to the end position.

    Parameters:
    start_coords: numpy array of shape (n, 2) containing start coordinates of n robots
    end_coords: numpy array of shape (n, 2) containing end coordinates of n robots
    reg_term: regularization term to add (default 1e-1)

    Returns:
    Adjusted end_coords numpy array
    """
    n = len(start_coords)
    # Create a copy of end_coords to modify

    for i in range(n):  # For each robot's path
        start = start_coords[i][:2]
        end = end_coords[i][:2]

        # Calculate direction vector of the path
        direction = end - start
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            # Normalize direction vector
            unit_direction = direction / direction_norm
            normal_direction = np.array([-unit_direction[1], unit_direction[0]])
        else:
            # If start and end are the same, skip this path
            continue

        path_adjusted = False

        for j in range(n):  # Check against all other robots' starting points
            if i == j:  # Skip comparing with itself
                continue

            # Get the starting point of the other robot
            point = start_coords[j][:2]

            # Calculate vector from start of path to the point
            start_to_point = point - start

            # Use dot product to find projection length
            projection_length = np.dot(start_to_point, unit_direction)

            # Calculate closest point on the line to the target point
            closest_point = start + projection_length * unit_direction

            # Check if the point is close to the line segment
            point_to_line_distance = np.linalg.norm(point - closest_point)

            # Check if the point is on the line segment
            on_segment = (0 <= projection_length <= direction_norm) and (point_to_line_distance < 1e-10)

            if on_segment:
                # If this path crosses another robot's starting point, adjust it
                path_adjusted = True

        if path_adjusted:
            # Add the regularization term to the end position
            adjustment = reg_term * normal_direction
            # Ensure the adjustment is actually applied
            end_coords[i][:2] = end + adjustment
            # For debugging
            print(f"Robot {i}: Original end: {end}, Adjustment: {adjustment}, New end: {end_coords[i]}")

    return end_coords


if __name__ == "__main__":
    initial_states = np.array([[0.0, 0.0, 0, 0, 0, 0], [5.0, 5.0, 0, 0, 0, 0], [4.0, 1.0, 0, 0, 0, 0]])
    final_states = np.array([[8.0, 8.0, 0, 0, 0, 0], [0.0, 0.0, 0, 0, 0, 0], [1.0, 3.0, 0, 0, 0, 0]])

    n_vehicles = len(initial_states)
    print(f"Number of vehicles: {n_vehicles}")
    solver = SCPSolver(n_vehicles=n_vehicles,
                       time_horizon=40)

    adjust_paths_with_crossings(initial_states, final_states, 0.5)

    solver.set_initial_states(initial_states)
    solver.set_finial_states(final_states)
    solver.precompute_optimization_matrices()

    if solver.generate_trajectories():
        print("Trajectory generation successful")
        solver.plot_positions
    else:
        print("Trajectory generation failed")







