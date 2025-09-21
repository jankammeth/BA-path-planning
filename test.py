import numpy as np
import osqp
import scipy.sparse as sp
from scipy.sparse import block_diag
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class SCPOSQPTrajectory:
    """
    Optimized implementation of the Sequential Convex Programming approach
    for collision-free 2D trajectory generation for multiple WOW crafts
    using OSQP directly with sparse matrices.
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
        self.K = int(self.T / self.h) # Number of discrete time steps (0 to K-1)
        # Note: In the original, K was int(T/h) + 1. Here, K is the number of intervals,
        # so accelerations are defined for K steps (a_0 to a_{K-1}).
        # Positions and velocities will have K+1 steps (p_0 to p_K).
        self.R = min_distance
        self.safety_margin = safety_margin  # Added safety margin
        self.space_dims = space_dims
        
        # OSQP problem dimensions
        self.num_accel_vars = 2 * self.N * self.K # 2 (x,y) * N vehicles * K time steps
        
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
        
        # OSQP problem components
        self.P = None
        self.q = np.zeros(self.num_accel_vars) # Minimize accelerations => q is zero
        self.A_base = None
        self.l_base = None
        self.u_base = None
        
        # For debug
        self.iteration_data = []
        
    def _accel_idx(self, vehicle_idx, k_step, coord_idx):
        """Helper to get the flattened index for acceleration a[vehicle_idx, k_step, coord_idx]"""
        return vehicle_idx * 2 * self.K + coord_idx * self.K + k_step
        # Simplified indexing if accelerations are [ax0, ay0, ax1, ay1, ... for V1, then V2, ...]
        # return vehicle_idx * 2 * self.K + k_step * 2 + coord_idx
    
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
    
    def _build_base_constraints(self):
        """
        Builds the OSQP constraint matrix A_base and bounds l_base, u_base
        for all static constraints (jerk, acceleration limits, initial/final states,
        and implicit position/velocity dynamics via transformation matrices).
        """
        if self.initial_states is None or self.final_states is None:
            raise ValueError("Initial and final states must be set before building constraints")

        constraints = []
        l_bounds = []
        u_bounds = []

        # Objective: Minimize sum of squared accelerations
        self.P = sp.eye(self.num_accel_vars, format='csc')

        # === 1. Jerk Constraints (a_k - a_{k-1})/h in [j_min, j_max] ===
        # (K-1) intervals, 2 coords, N vehicles
        A_jerk = sp.lil_matrix((2 * self.N * (self.K - 1), self.num_accel_vars))
        for i in range(self.N):
            for k in range(self.K - 1): # Jerk is from a_k to a_{k+1}
                # x-component
                row_idx_x = i * 2 * (self.K - 1) + 2 * k
                A_jerk[row_idx_x, self._accel_idx(i, k+1, 0)] = 1/self.h
                A_jerk[row_idx_x, self._accel_idx(i, k, 0)] = -1/self.h
                l_bounds.append(self.jmin[0])
                u_bounds.append(self.jmax[0])

                # y-component
                row_idx_y = i * 2 * (self.K - 1) + 2 * k + 1
                A_jerk[row_idx_y, self._accel_idx(i, k+1, 1)] = 1/self.h
                A_jerk[row_idx_y, self._accel_idx(i, k, 1)] = -1/self.h
                l_bounds.append(self.jmin[1])
                u_bounds.append(self.jmax[1])
        constraints.append(A_jerk.tocsc())

        # === 2. Initial Acceleration Constraint (a_0 for each vehicle) ===
        # N vehicles, 2 coords
        A_init_accel = sp.lil_matrix((2 * self.N, self.num_accel_vars))
        for i in range(self.N):
            A_init_accel[i*2, self._accel_idx(i, 0, 0)] = 1
            A_init_accel[i*2+1, self._accel_idx(i, 0, 1)] = 1
            l_bounds.append(self.initial_states['accelerations'][i, 0])
            u_bounds.append(self.initial_states['accelerations'][i, 0])
            l_bounds.append(self.initial_states['accelerations'][i, 1])
            u_bounds.append(self.initial_states['accelerations'][i, 1])
        constraints.append(A_init_accel.tocsc())

        # === 3. Acceleration Limits (a_k in [a_min, a_max]) ===
        # N vehicles, K steps, 2 coords
        A_accel_limits = sp.eye(self.num_accel_vars, format='csc')
        for _ in range(self.N * self.K):
            l_bounds.extend(self.amin)
            u_bounds.extend(self.amax)
        constraints.append(A_accel_limits)

        # === 4. Final Position Constraint (p_K for each vehicle) ===
        # p_K = p_0 + v_0 * T + sum_{j=0}^{K-1} h^2 * (K-j-0.5) * a_j
        # N vehicles, 2 coords
        A_final_pos = sp.lil_matrix((2 * self.N, self.num_accel_vars))
        for i in range(self.N):
            p0_x, p0_y = self.initial_states['positions'][i]
            v0_x, v0_y = self.initial_states['velocities'][i]
            pf_x, pf_y = self.final_states['positions'][i]

            # x-component
            row_idx_x = i * 2
            const_term_x = p0_x + v0_x * self.h * self.K
            for j in range(self.K):
                A_final_pos[row_idx_x, self._accel_idx(i, j, 0)] = self.h**2 * (self.K - j - 0.5)
            l_bounds.append(pf_x - const_term_x)
            u_bounds.append(pf_x - const_term_x)
            
            # y-component
            row_idx_y = i * 2 + 1
            const_term_y = p0_y + v0_y * self.h * self.K
            for j in range(self.K):
                A_final_pos[row_idx_y, self._accel_idx(i, j, 1)] = self.h**2 * (self.K - j - 0.5)
            l_bounds.append(pf_y - const_term_y)
            u_bounds.append(pf_y - const_term_y)
        constraints.append(A_final_pos.tocsc())

        # === 5. Final Velocity Constraint (v_K for each vehicle) ===
        # v_K = v_0 + sum_{j=0}^{K-1} h * a_j
        # N vehicles, 2 coords
        A_final_vel = sp.lil_matrix((2 * self.N, self.num_accel_vars))
        for i in range(self.N):
            v0_x, v0_y = self.initial_states['velocities'][i]
            vf_x, vf_y = self.final_states['velocities'][i]

            # x-component
            row_idx_x = i * 2
            const_term_x = v0_x
            for j in range(self.K):
                A_final_vel[row_idx_x, self._accel_idx(i, j, 0)] = self.h
            l_bounds.append(vf_x - const_term_x)
            u_bounds.append(vf_x - const_term_x)
            
            # y-component
            row_idx_y = i * 2 + 1
            const_term_y = v0_y
            for j in range(self.K):
                A_final_vel[row_idx_y, self._accel_idx(i, j, 1)] = self.h
            l_bounds.append(vf_y - const_term_y)
            u_bounds.append(vf_y - const_term_y)
        constraints.append(A_final_vel.tocsc())

        # === 6. Position Bounding Box Constraints (p_k in [p_min, p_max]) ===
        # This one is tricky as p_k is a sum of accelerations and initial state.
        # It's actually: p_k = p_0 + v_0 * k*h + sum_{j=0}^{k-1} h^2 * (k-j-0.5) * a_j
        # N vehicles, K steps (excluding initial p_0 which is fixed), 2 coords
        A_pos_bounds = sp.lil_matrix((2 * self.N * self.K, self.num_accel_vars))
        for i in range(self.N):
            p0_x, p0_y = self.initial_states['positions'][i]
            v0_x, v0_y = self.initial_states['velocities'][i]
            
            for k_pos in range(self.K): # k_pos from 0 to K-1 (corresponding to acceleration index)
                # Position p_{k_pos+1} depends on accelerations up to a_{k_pos}
                # But here we are indexing by a_k, which has K steps (a_0 to a_{K-1})
                # We need positions p_0 to p_K. So, for p_k, the last acceleration is a_{k-1}.
                # The total time steps for position is K+1
                
                # Let's align with p_k being the position at time k*h, where k goes from 0 to K.
                # The accelerations are a_0 to a_{K-1}. So p_K uses all accelerations.
                
                # For positions at time step 'k_time' (0 to K)
                # p_k_time = p_0 + v_0 * k_time * h + sum_{j=0}^{k_time-1} h^2 * (k_time - j - 0.5) * a_j
                
                # Iterate for position at time k_time from 0 to K
                k_time = k_pos # Represents p_{k_pos}
                
                # We need positions for k_time = 1 to K, because p_0 is fixed.
                if k_time == 0:
                    # p_0 is already fixed by initial_states, not part of variable constraints.
                    # Instead, we just add dummy constraints for p_0 that are always true.
                    # Or, even better, we formulate p_k for k=0 to K.
                    # The number of position values is K+1.
                    # Let's redefine K for position indices: 0 to K.
                    # Number of accels is K (0 to K-1).
                    # A_pos_bounds matrix has 2 * N * (K+1) rows.
                    continue # Skip initial position as it's not a variable.
                
                # Position at time (k_pos * h)
                p_k_x = p0_x + v0_x * k_time * self.h
                p_k_y = p0_y + v0_y * k_time * self.h
                
                for j in range(min(k_time, self.K)): # Accelerations up to a_{k_time-1} contribute
                    A_pos_bounds[i * 2 * self.K + 2 * k_time, self._accel_idx(i, j, 0)] = self.h**2 * (k_time - j - 0.5)
                    A_pos_bounds[i * 2 * self.K + 2 * k_time + 1, self._accel_idx(i, j, 1)] = self.h**2 * (k_time - j - 0.5)
                
                l_bounds.append(self.pmin[0] - p_k_x)
                u_bounds.append(self.pmax[0] - p_k_x)
                l_bounds.append(self.pmin[1] - p_k_y)
                u_bounds.append(self.pmax[1] - p_k_y)

        # Let's correct this and define position indices properly.
        # Positions p_k (k=0...K_pos) depend on accelerations a_j (j=0...K_accel-1)
        # Here: K_accel = self.K. K_pos = self.K.
        # This means p_K = p_0 + v_0 * K * h + sum_{j=0}^{K-1} h^2 * (K-j-0.5) * a_j
        # So we have K+1 position values.
        
        # We need to construct a mapping from accelerations to positions:
        # P_k = P_0 + k*h*V_0 + H_pos(k) @ A
        # Where A is [a_0 ... a_{K-1}]
        
        # The number of position constraints is 2 * N * (self.K+1)
        A_pos_bounds_corrected = sp.lil_matrix((2 * self.N * (self.K + 1), self.num_accel_vars))
        for i in range(self.N):
            p0_x, p0_y = self.initial_states['positions'][i]
            v0_x, v0_y = self.initial_states['velocities'][i]
            
            for k_time in range(self.K + 1): # For each discrete time step from 0 to K
                # Constant term from initial position and velocity
                const_term_x = p0_x + v0_x * k_time * self.h
                const_term_y = p0_y + v0_y * k_time * self.h
                
                # Coefficients for accelerations
                for j in range(min(k_time, self.K)): # Accelerations a_0 to a_{k_time-1}
                    coeff = self.h**2 * (k_time - j - 0.5)
                    A_pos_bounds_corrected[i * 2 * (self.K + 1) + 2 * k_time, self._accel_idx(i, j, 0)] = coeff
                    A_pos_bounds_corrected[i * 2 * (self.K + 1) + 2 * k_time + 1, self._accel_idx(i, j, 1)] = coeff
                
                l_bounds.append(self.pmin[0] - const_term_x)
                u_bounds.append(self.pmax[0] - const_term_x)
                l_bounds.append(self.pmin[1] - const_term_y)
                u_bounds.append(self.pmax[1] - const_term_y)
        constraints.append(A_pos_bounds_corrected.tocsc())

        self.A_base = sp.vstack(constraints).tocsc()
        self.l_base = np.array(l_bounds)
        self.u_base = np.array(u_bounds)
        #print(f"Base constraints built. A_base shape: {self.A_base.shape}")
        #print(f"l_base shape: {self.l_base.shape}, u_base shape: {self.u_base.shape}")


    def _build_collision_constraints(self, prev_accelerations):
        """
        Builds linearized collision avoidance constraints for OSQP based on a previous solution.
        The constraints are of the form:
            (p_i_prev - p_j_prev)/||p_i_prev - p_j_prev||_2^T * (p_i_curr - p_j_curr) >= R + safety_margin
        Linearized around p_i_prev, p_j_prev.
        """
        prev_positions, _ = self._accelerations_to_positions_velocities(prev_accelerations)

        # Number of collision constraints:
        # One for each pair of vehicles at each time step (K+1 position time steps)
        n_pairs = (self.N * (self.N - 1)) // 2
        num_collision_constraints = n_pairs * (self.K + 1) # K+1 time steps for positions (0 to K)

        A_collision = sp.lil_matrix((num_collision_constraints, self.num_accel_vars))
        l_collision = np.zeros(num_collision_constraints)
        u_collision = np.ones(num_collision_constraints) * np.inf

        row_idx = 0
        effective_min_distance = self.R + self.safety_margin
        
        for k_time in range(self.K + 1): # For each discrete time step from 0 to K
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    p_i_prev = prev_positions[i, k_time]
                    p_j_prev = prev_positions[j, k_time]

                    diff_vector = p_i_prev - p_j_prev
                    dist_prev = np.linalg.norm(diff_vector)

                    # Handle numerical stability if vehicles are too close in the previous iteration
                    if dist_prev < 1e-6:
                        # If overlapping, push them apart in a fixed direction (e.g., x-axis)
                        eta = np.array([1.0, 0.0])
                        # A small positive distance for linearization base
                        dist_prev = 1e-6 
                    else:
                        eta = diff_vector / dist_prev

                    # Constraint for vehicle i (p_i_curr): eta^T * (p_i_curr)
                    # p_i_curr = p_0_i + v_0_i * k_time * h + sum_{s=0}^{k_time-1} h^2 * (k_time - s - 0.5) * a_{i,s}
                    
                    # For a given vehicle i at time k_time, and its acceleration a_{i,s}
                    # Coeff of a_{i,s} in p_i_curr_x is h^2 * (k_time - s - 0.5)
                    # Coeff of a_{i,s} in p_i_curr_y is h^2 * (k_time - s - 0.5)
                    
                    # So, eta^T * p_i_curr = eta_x * p_i_curr_x + eta_y * p_i_curr_y
                    # The coeff for a_{i,s,x} will be eta_x * h^2 * (k_time - s - 0.5)
                    # The coeff for a_{i,s,y} will be eta_y * h^2 * (k_time - s - 0.5)
                    
                    for s in range(min(k_time, self.K)): # Accelerations up to a_{k_time-1}
                        coeff = self.h**2 * (k_time - s - 0.5)
                        A_collision[row_idx, self._accel_idx(i, s, 0)] = eta[0] * coeff
                        A_collision[row_idx, self._accel_idx(i, s, 1)] = eta[1] * coeff
                        
                        A_collision[row_idx, self._accel_idx(j, s, 0)] = -eta[0] * coeff
                        A_collision[row_idx, self._accel_idx(j, s, 1)] = -eta[1] * coeff

                    # Right-hand side of the linearized constraint:
                    # R_eff + eta^T * (p_i_prev - p_j_prev) - eta^T * (p_i_prev - p_j_prev) + eta^T * (p_i_prev - p_j_prev)
                    # No, it's: R_eff + eta^T * (p_i_prev - p_j_prev) - (eta^T * p_i_ref - eta^T * p_j_ref)
                    # where p_i_ref is p_i_0 + v_i_0 * k_time * h
                    
                    p0_i_x, p0_i_y = self.initial_states['positions'][i]
                    v0_i_x, v0_i_y = self.initial_states['velocities'][i]
                    p0_j_x, p0_j_y = self.initial_states['positions'][j]
                    v0_j_x, v0_j_y = self.initial_states['velocities'][j]

                    p_i_ref_x = p0_i_x + v0_i_x * k_time * self.h
                    p_i_ref_y = p0_i_y + v0_i_y * k_time * self.h
                    p_j_ref_x = p0_j_x + v0_j_x * k_time * self.h
                    p_j_ref_y = p0_j_y + v0_j_y * k_time * self.h
                    
                    rhs = effective_min_distance \
                          + np.dot(eta, diff_vector) \
                          - np.dot(eta, np.array([p_i_ref_x - p_j_ref_x, p_i_ref_y - p_j_ref_y]))
                    
                    l_collision[row_idx] = rhs
                    row_idx += 1
        
        return A_collision.tocsc(), l_collision, u_collision

    def generate_trajectories(self, max_iterations=15, convergence_threshold=1e-3):
        """Main method to generate collision-free trajectories using SCP with OSQP."""
        if self.initial_states is None or self.final_states is None:
            raise ValueError("Initial and final states must be set before generating trajectories")
        
        start_time = time.time()
        
        # Clear any previous debug data
        self.iteration_data = []
        
        # Build base constraints once
        self._build_base_constraints()
        
        # Initial guess: linear interpolation of accelerations, or zeros
        # A simple initial guess: all zeros
        accelerations = np.zeros(self.num_accel_vars)
        
        # Solve initial trajectory without avoidance (a warm start for SCP)
        print("Solving initial trajectory without collision avoidance...")
        # Create a problem instance for OSQP
        prob = osqp.OSQP()
        prob.setup(P=self.P, q=self.q, A=self.A_base, l=self.l_base, u=self.u_base,
                   verbose=False, warm_start=True, max_iter=10000)
        
        result_initial = prob.solve()
        if result_initial.info.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Initial trajectory solver failed with status {result_initial.info.status}. Using zero accelerations.")
            # If initial solve fails, use zero accelerations as a fallback
            accelerations = np.zeros(self.num_accel_vars)
        else:
            accelerations = result_initial.x
        
        # Store the initial solution for debugging
        positions, velocities = self._accelerations_to_positions_velocities(accelerations)
        self.iteration_data.append({
            'positions': positions.copy(),
            'accelerations': accelerations.copy(),
            'objective': np.sum(accelerations**2) # Calculate objective manually
        })
        
        # SCP iterations
        iteration = 0
        converged = False
        prev_objective = float('inf')
        
        while iteration < max_iterations and not converged:
            print(f"\nSCP Iteration {iteration+1}")
            
            # Build collision constraints based on previous solution
            A_collision, l_collision, u_collision = self._build_collision_constraints(accelerations)
            
            # Combine all constraints
            A_combined = sp.vstack([self.A_base, A_collision]).tocsc()
            l_combined = np.hstack([self.l_base, l_collision])
            u_combined = np.hstack([self.u_base, u_collision])
            
            # Update OSQP problem
            prob.update(Ax=A_combined.data, Ai=A_combined.indices, Ap=A_combined.indptr,
                        l=l_combined, u=u_combined, x=accelerations) # Warm start with previous solution
            
            # Solve the problem
            result = prob.solve()
            
            # Check for solver errors
            if result.info.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Solver failed at iteration {iteration+1}: {result.info.status}. Skipping update.")
                # Try to continue with previous solution or a more relaxed approach if this happens
                if self.safety_margin < 1.0: # Try increasing safety margin for next iteration
                    self.safety_margin += 0.1
                    print(f"Increased safety margin to {self.safety_margin:.2f}")
                iteration += 1
                continue # Skip the rest of this iteration and try again

            new_accelerations = result.x
            objective = np.sum(new_accelerations**2) # Calculate objective manually
            
            # Store this iteration's solution for debugging
            positions, velocities = self._accelerations_to_positions_velocities(new_accelerations)
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
                converged = False # If not satisfied, force more iterations
                
            # Perform continuous collision checking
            continuous_avoidance_satisfied = self._optimized_check_continuous_avoidance(
                positions, velocities, new_accelerations
            )
            if continuous_avoidance_satisfied:
                print("Continuous avoidance constraints satisfied.")
            else:
                print("Warning: Continuous avoidance constraints not satisfied.")
                if converged and iteration < max_iterations - 1:
                    print("Continuing iterations to resolve continuous collisions.")
                    converged