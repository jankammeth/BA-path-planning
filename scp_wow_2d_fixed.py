import numpy as np
import cvxpy as cp
import osqp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class SCPCraft2DTrajectory:
    """
    Optimized implementation of the Sequential Convex Programming approach
    for collision-free 2D trajectory generation for multiple WOW crafts
    based on the paper by Augugliaro et al. but simplified to 2D (x,y plane only).
    """
    
    def __init__(self, 
                 n_vehicles=5,  # Number of vehicles 
                 time_horizon=3.0,  # Total time T for trajectory
                 time_step=0.1,  # Discretization time step h
                 min_distance=0.1,  # Minimum distance between vehicles R
                 safety_margin=0.2,  # Additional safety margin for avoidance constraints
                 space_dims=[10, 10],  # Dimensions of the space [x, y]
                 ):
        
        """Initialize the 2D trajectory planner."""
        
        self.N = n_vehicles
        self.T = time_horizon
        self.h = time_step
        self.K = int(self.T / self.h) + 1  # Number of discrete time steps
        self.R = min_distance
        self.safety_margin = safety_margin  # Added safety margin
        self.space_dims = space_dims
        
        # Limits for position, acceleration and jerk
        self.pmin = np.array([0, 0])
        self.pmax = np.array(space_dims)
        
        # Acceleration limits in x, y
        self.amin = np.array([-5, -5])
        self.amax = np.array([5, 5])
        
        # Jerk limits in x, y
        self.jmin = np.array([-20, -20])
        self.jmax = np.array([20, 20])
        
        # Initialize variables
        self.initial_states = None
        self.final_states = None
        self.trajectories = None
        
        # Precompute transform matrices for position and velocity
        self._precompute_transformation_matrices()
        
        # For debug
        self.iteration_data = []
    
    def _precompute_transformation_matrices(self):
        """
        Precompute position and velocity transformation matrices
        to convert from acceleration to position and velocity.
        This is a significant optimization that avoids recalculating
        these matrices repeatedly.
        """
        # Position transformation matrix
        self.pos_transform = np.zeros((self.K, self.K-1))
        for k in range(1, self.K):
            for j in range(k):
                self.pos_transform[k, j] = self.h**2 * (k - j - 0.5)
                
        # Velocity transformation matrix
        self.vel_transform = np.zeros((self.K, self.K-1))
        for k in range(1, self.K):
            for j in range(k):
                self.vel_transform[k, j] = self.h

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
    
    def generate_trajectories(self, max_iterations=15, convergence_threshold=1e-3):
        """Main method to generate collision-free trajectories using SCP."""
        if self.initial_states is None or self.final_states is None:
            raise ValueError("Initial and final states must be set before generating trajectories")
        
        start_time = time.time()
        
        # Clear any previous debug data
        self.iteration_data = []
        
        # Initialize trajectories without avoidance constraints
        accelerations = self._solve_initial_trajectory()
        
        # Store the initial solution for debugging
        positions, velocities = self._compute_positions_velocities(accelerations)
        self.iteration_data.append({
            'positions': positions.copy(),
            'accelerations': accelerations.copy(),
            'objective': self._compute_objective(accelerations)
        })
        
        # SCP iterations
        iteration = 0
        converged = False
        prev_objective = float('inf')
        
        while iteration < max_iterations and not converged:
            print(f"SCP Iteration {iteration+1}")
            
            # Solve the SCP problem with linearized avoidance constraints
            new_accelerations = self._solve_with_avoidance_constraints(accelerations)
            
            # Compute objective value
            objective = self._compute_objective(new_accelerations)
            
            # Store this iteration's solution for debugging
            positions, velocities = self._compute_positions_velocities(new_accelerations)
            self.iteration_data.append({
                'positions': positions.copy(),
                'accelerations': new_accelerations.copy(),
                'objective': objective
            })
            
            # Check convergence
            rel_improvement = abs(prev_objective - objective) / max(1e-10, prev_objective)
            print(f"Objective: {objective:.6f}, Relative improvement: {rel_improvement:.6f}")
            
            if rel_improvement < convergence_threshold:
                converged = True
                print(f"Converged after {iteration+1} iterations.")
            
            # Check if avoidance constraints are satisfied
            avoidance_satisfied = self._fast_check_avoidance_constraints(positions)
            if avoidance_satisfied:
                print("Avoidance constraints satisfied at discrete timesteps.")
            else:
                print("Warning: Avoidance constraints not satisfied at discrete timesteps.")
                
            # Perform continuous collision checking - only if avoidance is satisfied at discrete timesteps
            continuous_avoidance_satisfied = True
            if avoidance_satisfied:
                continuous_avoidance_satisfied = self._optimized_check_continuous_avoidance(
                    positions, velocities, new_accelerations
                )
                if continuous_avoidance_satisfied:
                    print("Continuous avoidance constraints satisfied.")
                else:
                    print("Warning: Continuous avoidance constraints not satisfied.")
                    
                    # If we've converged but continuous collision checking fails, continue iterations
                    if converged and iteration < max_iterations - 1:
                        print("Continuing iterations to resolve continuous collisions.")
                        converged = False
            
            # Update for next iteration
            accelerations = new_accelerations
            prev_objective = objective
            iteration += 1
        
        # Store the final trajectories
        positions, velocities = self._compute_positions_velocities(accelerations)
        
        self.trajectories = {
            'positions': positions,  # Shape (N, K, 2)
            'velocities': velocities,  # Shape (N, K, 2)
            'accelerations': accelerations  # Shape (N, K, 2)
        }
        
        end_time = time.time()
        print(f"Trajectory generation completed in {end_time - start_time:.3f} seconds")
        
        # Perform final feasibility check (but don't do expensive collision checks again)
        is_feasible = self._check_feasibility(skip_continuous_check=True)
        if not is_feasible:
            print("Warning: Generated trajectories may not be feasible.")
            
            # If after all iterations we still have continuous collisions,
            # try increasing the safety margin as a last resort
            if not self._fast_check_avoidance_constraints(positions):
                print("Attempting to fix collisions by increasing the safety margin...")
                old_safety_margin = self.safety_margin
                self.safety_margin += 0.3  # Increase the safety margin significantly
                
                # One more iteration with increased safety margin
                accelerations = self._solve_with_avoidance_constraints(accelerations)
                positions, velocities = self._compute_positions_velocities(accelerations)
                
                self.trajectories = {
                    'positions': positions,
                    'velocities': velocities,
                    'accelerations': accelerations
                }
                
                # Check if this helped
                if self._fast_check_avoidance_constraints(positions):
                    print(f"Successfully resolved collisions by increasing safety margin to {self.safety_margin}.")
                else:
                    print("Warning: Unable to resolve all collisions.")
                    # Restore the original safety margin
                    self.safety_margin = old_safety_margin
        
        return self.trajectories
    
    def _solve_initial_trajectory(self):
        """
        Solve the initial trajectory optimization problem without avoidance constraints.
        This serves as the starting point for the SCP iterations.
        
        Optimized to solve for all vehicles at once to avoid repeated problem setup.
        """
        # Initialize objective and constraints
        objective_terms = []
        constraints = []
        
        # Create optimization variables for all vehicles and timesteps at once
        # Shape: (N, K, 2)
        a_vars = cp.Variable((self.N, self.K, 2))
        
        # Add constraints and objective for all vehicles
        for i in range(self.N):
            # Extract initial and final states for vehicle i
            p_init = self.initial_states['positions'][i]
            v_init = self.initial_states['velocities'][i]
            a_init = self.initial_states['accelerations'][i]
            
            p_final = self.final_states['positions'][i]
            v_final = self.final_states['velocities'][i]
            a_final = self.final_states['accelerations'][i]
            
            # Initial acceleration constraint
            constraints.append(a_vars[i, 0] == a_init)
            
            # Final acceleration constraint
            constraints.append(a_vars[i, -1] == a_final)
            
            # Add acceleration limits for all time steps
            constraints.append(a_vars[i] >= self.amin)
            constraints.append(a_vars[i] <= self.amax)
            
            # Add jerk limits for all time steps except the first
            for k in range(1, self.K):
                j_k = (a_vars[i, k] - a_vars[i, k-1]) / self.h
                constraints.append(j_k >= self.jmin)
                constraints.append(j_k <= self.jmax)
            
            # Compute the final position and velocity using matrix operations
            p_k = p_init + self.h * self.K * v_init
            for k in range(self.K-1):
                p_k = p_k + self.h**2 * (self.K-k-1) * a_vars[i, k]
                
            v_k = v_init
            for k in range(self.K-1):
                v_k = v_k + self.h * a_vars[i, k]
            
            # Add final position and velocity constraints
            constraints.append(p_k == p_final)
            constraints.append(v_k + self.h * a_vars[i, -1] == v_final)
            
            # Add to objective: minimize sum of squared acceleration magnitude for vehicle i
            objective_terms.append(cp.sum_squares(a_vars[i]))
        
        # Solve the problem
        objective = cp.Minimize(cp.sum(objective_terms))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=True)
        
        # Extract solution to numpy array
        accelerations = a_vars.value
        
        return accelerations
    
    def _solve_with_avoidance_constraints(self, prev_accelerations):
        """
        Solve the optimization problem with linearized avoidance constraints
        around the previous solution.
        
        Optimized for speed and better convergence.
        """
        # Compute positions from previous accelerations for linearization
        prev_positions, _ = self._compute_positions_velocities(prev_accelerations)
        
        # Setup the optimization problem
        a_vars = cp.Variable((self.N, self.K, 2))
        objective_terms = []
        constraints = []
        
        # Add dynamics constraints and objective for each vehicle
        for i in range(self.N):
            # Extract initial and final states for vehicle i
            p_init = self.initial_states['positions'][i]
            v_init = self.initial_states['velocities'][i]
            a_init = self.initial_states['accelerations'][i]
            
            p_final = self.final_states['positions'][i]
            v_final = self.final_states['velocities'][i]
            a_final = self.final_states['accelerations'][i]
            
            # Initial acceleration constraint
            constraints.append(a_vars[i, 0] == a_init)
            
            # Final acceleration constraint
            constraints.append(a_vars[i, -1] == a_final)
            
            # Add acceleration limits for all time steps
            constraints.append(a_vars[i] >= self.amin)
            constraints.append(a_vars[i] <= self.amax)
            
            # Add jerk limits for all time steps except the first
            for k in range(1, self.K):
                j_k = (a_vars[i, k] - a_vars[i, k-1]) / self.h
                constraints.append(j_k >= self.jmin)
                constraints.append(j_k <= self.jmax)
            
            # Compute the final position and velocity using matrix operations
            p_k = p_init + self.h * self.K * v_init
            for k in range(self.K-1):
                p_k = p_k + self.h**2 * (self.K-k-1) * a_vars[i, k]
                
            v_k = v_init
            for k in range(self.K-1):
                v_k = v_k + self.h * a_vars[i, k]
            
            # Add final position and velocity constraints
            constraints.append(p_k == p_final)
            constraints.append(v_k + self.h * a_vars[i, -1] == v_final)
            
            # Add to objective: minimize sum of squared acceleration magnitude for vehicle i
            objective_terms.append(cp.sum_squares(a_vars[i]))
        
        # Add collision avoidance constraints with safety margin
        effective_min_distance = self.R + self.safety_margin
        
        # Precompute position matrices for all vehicles and time steps
        p_vars = {}
        for i in range(self.N):
            p_init = self.initial_states['positions'][i]
            v_init = self.initial_states['velocities'][i]
            
            for k in range(self.K):
                # Compute position at time k as a function of acceleration variables
                p_k = p_init + k * self.h * v_init
                for j in range(min(k, self.K-1)):
                    p_k = p_k + self.h**2 * (k-j-0.5) * a_vars[i, j]
                
                p_vars[(i, k)] = p_k
        
        # Add avoidance constraints between all pairs of vehicles
        for i in range(self.N):
            for j in range(i+1, self.N):
                for k in range(self.K):
                    p_i_prev = prev_positions[i, k]
                    p_j_prev = prev_positions[j, k]
                    
                    # Compute the difference vector
                    diff_vector = p_i_prev - p_j_prev
                    norm_diff = np.linalg.norm(diff_vector)
                    
                    # If vehicles are very close or overlapping, apply a stronger constraint
                    if norm_diff < 1e-6:
                        # Use a default direction
                        eta = np.array([1.0, 0.0])
                        initial_distance = 0.1  # Small but non-zero
                    else:
                        eta = diff_vector / norm_diff
                        initial_distance = norm_diff
                    
                    # Get the position variables
                    p_i_k = p_vars[(i, k)]
                    p_j_k = p_vars[(j, k)]
                    
                    # Linearized constraint
                    constraints.append(
                        initial_distance + cp.sum(cp.multiply(eta, p_i_k - p_j_k - p_i_prev + p_j_prev)) >= effective_min_distance
                    )
        
        # Solve the problem
        objective = cp.Minimize(cp.sum(objective_terms))
        problem = cp.Problem(objective, constraints)
        
        try:
            # Use more efficient solver settings
            problem.solve(solver=cp.ECOS, 
                         verbose=False,
                         max_iters=100,
                         abstol=1e-4,
                         reltol=1e-4,
                         feastol=1e-4)
            
            # Check if the solver found a solution
            if problem.status in ["optimal", "optimal_inaccurate"]:
                return a_vars.value
            else:
                print(f"Warning: Solver returned status {problem.status}")
                print("Using relaxed constraints...")
                return self._solve_with_relaxed_constraints(prev_accelerations)
            
        except cp.SolverError as e:
            print(f"Solver error: {e}")
            print("Using relaxed constraints...")
            return self._solve_with_relaxed_constraints(prev_accelerations)
    
    def _solve_with_relaxed_constraints(self, prev_accelerations):
        """
        Solve with relaxed constraints when the original problem is infeasible.
        This is a simplified version that adds slack variables.
        """
        # Compute positions from previous accelerations for linearization
        prev_positions, _ = self._compute_positions_velocities(prev_accelerations)
        
        # Setup the optimization problem
        a_vars = cp.Variable((self.N, self.K, 2))
        objective_terms = []
        constraints = []
        slack_vars = []
        slack_penalty = 1000.0  # High penalty for violating constraints
        
        # Add dynamics constraints and objective for each vehicle (same as before)
        for i in range(self.N):
            # Extract initial and final states for vehicle i
            p_init = self.initial_states['positions'][i]
            v_init = self.initial_states['velocities'][i]
            a_init = self.initial_states['accelerations'][i]
            
            p_final = self.final_states['positions'][i]
            v_final = self.final_states['velocities'][i]
            a_final = self.final_states['accelerations'][i]
            
            # Initial acceleration constraint
            constraints.append(a_vars[i, 0] == a_init)
            
            # Final acceleration constraint
            constraints.append(a_vars[i, -1] == a_final)
            
            # Add acceleration limits for all time steps
            constraints.append(a_vars[i] >= self.amin)
            constraints.append(a_vars[i] <= self.amax)
            
            # Add jerk limits for all time steps except the first
            for k in range(1, self.K):
                j_k = (a_vars[i, k] - a_vars[i, k-1]) / self.h
                constraints.append(j_k >= self.jmin)
                constraints.append(j_k <= self.jmax)
            
            # Compute the final position and velocity using matrix operations
            p_k = p_init + self.h * self.K * v_init
            for k in range(self.K-1):
                p_k = p_k + self.h**2 * (self.K-k-1) * a_vars[i, k]
                
            v_k = v_init
            for k in range(self.K-1):
                v_k = v_k + self.h * a_vars[i, k]
            
            # Add final position and velocity constraints
            constraints.append(p_k == p_final)
            constraints.append(v_k + self.h * a_vars[i, -1] == v_final)
            
            # Add to objective: minimize sum of squared acceleration magnitude for vehicle i
            objective_terms.append(cp.sum_squares(a_vars[i]))
        
        # Precompute position matrices for all vehicles and time steps
        p_vars = {}
        for i in range(self.N):
            p_init = self.initial_states['positions'][i]
            v_init = self.initial_states['velocities'][i]
            
            for k in range(self.K):
                # Compute position at time k as a function of acceleration variables
                p_k = p_init + k * self.h * v_init
                for j in range(min(k, self.K-1)):
                    p_k = p_k + self.h**2 * (k-j-0.5) * a_vars[i, j]
                
                p_vars[(i, k)] = p_k
        
        # Add relaxed avoidance constraints with slack variables
        effective_min_distance = self.R + self.safety_margin
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                for k in range(self.K):
                    p_i_prev = prev_positions[i, k]
                    p_j_prev = prev_positions[j, k]
                    
                    # Compute the difference vector
                    diff_vector = p_i_prev - p_j_prev
                    norm_diff = np.linalg.norm(diff_vector)
                    
                    # If vehicles are very close or overlapping, apply a stronger constraint
                    if norm_diff < 1e-6:
                        # Use a default direction
                        eta = np.array([1.0, 0.0])
                        initial_distance = 0.1  # Small but non-zero
                    else:
                        eta = diff_vector / norm_diff
                        initial_distance = norm_diff
                    
                    # Get the position variables
                    p_i_k = p_vars[(i, k)]
                    p_j_k = p_vars[(j, k)]
                    
                    # Create a slack variable for this constraint
                    slack = cp.Variable(1, nonneg=True)
                    slack_vars.append(slack)
                    
                    # Add relaxed constraint with slack variable
                    constraints.append(
                        initial_distance + cp.sum(cp.multiply(eta, p_i_k - p_j_k - p_i_prev + p_j_prev)) + slack >= effective_min_distance
                    )
                    
                    # Add penalty to objective
                    objective_terms.append(slack_penalty * slack)
        
        # Solve the problem with relaxed constraints
        objective = cp.Minimize(cp.sum(objective_terms))
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, 
                         verbose=False,
                         max_iters=100,
                         abstol=1e-4,
                         reltol=1e-4,
                         feastol=1e-4)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                return a_vars.value
            else:
                print(f"Relaxed problem failed with status {problem.status}")
                return prev_accelerations
        except cp.SolverError as e:
            print(f"Solver error on relaxed problem: {e}")
            return prev_accelerations
    
    def _compute_positions_velocities(self, accelerations):
        """
        Compute positions and velocities for all vehicles at all time steps
        given the accelerations. Vectorized for better performance.
        """
        positions = np.zeros((self.N, self.K, 2))
        velocities = np.zeros((self.N, self.K, 2))
        
        for i in range(self.N):
            # Initial states
            positions[i, 0] = self.initial_states['positions'][i]
            velocities[i, 0] = self.initial_states['velocities'][i]
            
            # Vectorized computation for positions and velocities
            for k in range(1, self.K):
                # Update velocity using dynamics equation
                velocities[i, k] = velocities[i, 0].copy()
                for j in range(k):
                    velocities[i, k] += self.h * accelerations[i, j]
                
                # Update position using dynamics equation
                positions[i, k] = positions[i, 0].copy() + self.h * k * velocities[i, 0]
                for j in range(k):
                    positions[i, k] += self.h**2 * (k-j-0.5) * accelerations[i, j]
        
        return positions, velocities
    
    def _compute_objective(self, accelerations):
        """Calculate the objective function value of the current solution."""
        return np.sum(accelerations**2)  # Faster than looping
    
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
                    if dist < self.R:
                        print(f"Avoidance constraint violation at timestep {k} between vehicles {i} and {j}: distance = {dist:.3f}")
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
                    a0_i = accelerations[i, k]
                    pos_i = p0_i + v0_i * dt + 0.5 * a0_i * dt**2
                    
                    # Position of vehicle j at intermediate time
                    p0_j = positions[j, k]
                    v0_j = velocities[j, k]
                    a0_j = accelerations[j, k]
                    pos_j = p0_j + v0_j * dt + 0.5 * a0_j * dt**2
                    
                    # Check for collision
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < self.R:
                        print(f"Continuous avoidance constraint violation between timesteps {k} and {k+1} at fraction {fraction:.2f}")
                        print(f"Between vehicles {i} and {j}: distance = {dist:.3f}")
                        return False
        
        return True
    
    def _check_feasibility(self, skip_continuous_check=False):
        """
        Check if the generated trajectories satisfy all constraints.
        Can skip expensive continuous collision checking if already performed.
        """
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")
            
        positions = self.trajectories['positions']
        velocities = self.trajectories['velocities']
        accelerations = self.trajectories['accelerations']
        
        # Check position bounds
        for i in range(self.N):
            for k in range(self.K):
                p = positions[i, k]
                if np.any(p < self.pmin) or np.any(p > self.pmax):
                    print(f"Position constraint violated for vehicle {i} at time {k}")
                    return False
        
        # Check acceleration bounds
        for i in range(self.N):
            for k in range(self.K):
                a = accelerations[i, k]
                if np.any(a < self.amin) or np.any(a > self.amax):
                    print(f"Acceleration constraint violated for vehicle {i} at time {k}")
                    return False
        
        # Check jerk bounds
        for i in range(self.N):
            for k in range(1, self.K):
                j = (accelerations[i, k] - accelerations[i, k-1]) / self.h
                if np.any(j < self.jmin) or np.any(j > self.jmax):
                    print(f"Jerk constraint violated for vehicle {i} at time {k}")
                    return False
        
        # Check discrete avoidance constraints
        if not self._fast_check_avoidance_constraints(positions):
            return False
        
        # Check continuous avoidance constraints (skip if requested)
        if not skip_continuous_check and not self._optimized_check_continuous_avoidance(positions, velocities, accelerations):
            return False
        
        return True
    
    def visualize_trajectories(self, show_animation=True, save_path=None):
        """Visualize the generated 2D trajectories."""
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")
            
        positions = self.trajectories['positions']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Plot bounds of the space
        ax.set_xlim([0, self.space_dims[0]])
        ax.set_ylim([0, self.space_dims[1]])
        
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
            min_x, max_x = 0, self.space_dims[0]
            min_y, max_y = 0, self.space_dims[1]
            ax.set_xlim([min_x - 0.5, max_x + 0.5])
            ax.set_ylim([min_y - 0.5, max_y + 0.5])
            
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