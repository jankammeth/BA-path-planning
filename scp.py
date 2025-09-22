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
        self.R = min_distance
        self.space_dims = space_dims

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

        # Velocity and acceleration limits with margins for solver stability
        self.vel_min = -3.0
        self.vel_max = 3.0
        self.vel_margin = 0.002

        self.acc_min = -105.0
        self.acc_max = 105.0
        self.acc_margin = 0.02

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

    def generate_trajectories(self, max_iterations=15, convergence_threshold=1e-3):
        """Main method to generate collision-free trajectories using SCP."""
        start_time = time.time()

        self.precompute_optimization_matrices()
        accelerations = self._solve_initial_trajectory()



        positions, velocities = self._compute_positions_velocities(accelerations)

        self.trajectories = {
            'positions': positions,  # Shape (N, K, 2)
            'velocities': velocities,  # Shape (N, K, 2)
            'accelerations': accelerations  # Shape (N, K, 2)
        }

        # SCP iterations
        iteration = 0
        converged = False
        prev_objective = float('inf')

        
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
    
    def _solve_initial_trajectory(self):
        """
        Solve the initial trajectory optimization problem without avoidance constraints.
        This serves as the starting point for the SCP iterations.
        
        Optimized to solve for all vehicles at once to avoid repeated problem setup.
        """

        problem = osqp.OSQP()

        P = np.eye(2*self.N*self.K)

        A = sp.vstack([self.A_jerk, self.A_accel_initial, self.A_accel_final, self.A_accel_constraints, self.A_vel_constraints, self.A_pos_constraints], format="csc")
        l = np.hstack([self.l_jerk, self.l_accel_initial, self.l_accel_final, self.l_accel_constraints, self.l_vel_constraints, self.l_pos_constraints])
        u = np.hstack([self.u_jerk, self.u_accel_initial, self.u_accel_final, self.u_accel_constraints, self.u_vel_constraints, self.u_pos_constraints])
        q = np.array([0] * 2 * self.N * self.K)

        # convert into sparse matrixes for OSQP
        P = sp.csc_matrix(P)

        problem.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

        results = problem.solve()
        if results.info.status_val not in (1, 2):  # Solved / Solved Inaccurate
            raise RuntimeError(f"OSQP failed: {results.info.status}")

        acc_flat = results.x
        accelerations = results.x.reshape(self.N, self.K, 2)
        
        return accelerations

    def _compute_positions_velocities(self, accelerations):
        """
        Compute positions and velocities for all vehicles at all time steps
        given the accelerations. Uses the stored flattened initial
        positions and velocities.
        
        Args:
            accelerations: (N, K, 2) array from optimization
            
        Returns:
            positions: (N, K, 2)
            velocities: (N, K, 2)
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
                # velocity update
                velocities[i, k] = init_vel[i] + self.h * np.sum(accelerations[i, :k], axis=0)
                
                # position update
                positions[i, k] = init_pos[i] + self.h * k * init_vel[i]
                for j in range(k):
                    positions[i, k] += (self.h**2) * (k - j - 0.5) * accelerations[i, j]
        
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
