import time

import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp
from matplotlib.patches import Circle


def _jerk_matrix_sparse(N, K, h):
    """(2N(K-1)) x (2NK) first-difference operator per vehicle/axis, scaled by 1/h."""
    rows, cols, vals = [], [], []
    r = 0
    for i in range(N):
        base = 2 * i * K
        for k in range(K - 1):
            # x
            rows += [r, r]
            cols += [base + 2 * k, base + 2 * (k + 1)]
            vals += [-1.0 / h, 1.0 / h]
            r += 1
            # y
            rows += [r, r]
            cols += [base + 2 * k + 1, base + 2 * (k + 1) + 1]
            vals += [-1.0 / h, 1.0 / h]
            r += 1
    m, n = 2 * N * (K - 1), 2 * N * K
    return sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsc()


class SCP:
    def __init__(
        self,
        n_vehicles=5,
        time_horizon=3.0,
        time_step=0.1,
        min_distance=0.1,
        space_dims=None,
    ):
        self.N = n_vehicles
        self.T = time_horizon
        self.h = time_step
        self.K = int(self.T / self.h)
        self.R = min_distance

        # bad practice to pass immutable object, hence pass None as default
        if space_dims is None:
            space_dims = [0, 0, 20, 20]
        self.space_dims = space_dims

        # Hyperparameters
        self.convergence_tolerance = 1.5e-2

        # Initialization: Solution trajectory & Equality constraints
        self.trajectories = None

        self.initial_positions = None
        self.initial_velocities = None
        self.final_positions = None
        self.final_velocities = None

        # Initialization: Inequality Contraints
        self.pos_min = np.array([space_dims[0], space_dims[1]])
        self.pos_max = np.array([space_dims[2], space_dims[3]])

        # Limits to match first implementation
        self.vel_min = -2
        self.vel_max = 2

        self.acc_min = -15.0
        self.acc_max = 15.0

        self.jerk_min = -20
        self.jerk_max = 20

        # Constraint matrices
        self.C_jerk = None
        self.l_jerk = []
        self.u_jerk = []

        self.C_acc = None
        self.l_acc = []
        self.u_acc = []

        self.C_vel = None
        self.l_vel = []
        self.u_vel = []

        self.C_pos = None
        self.l_pos = []
        self.u_pos = []

        print("---=== SCP Problem initialized ===---")
        print(f"Number of timesteps: {self.K}")
        print(f"Timestep: {self.h}")
        print(f"Minimum distance between vehicles: {self.R}")
        print(f"Space dimensions: {self.space_dims}")

    def set_initial_states(self, positions, velocities=None):
        """Set initial states for all vehicles in flat format."""
        if velocities is None:
            velocities = np.zeros((self.N, 2))

        # Conversion to flatt vector
        self.initial_positions = positions.flatten()
        self.initial_velocities = velocities.flatten()

        assert len(self.initial_positions) == len(self.initial_velocities) == 2 * self.N, (
            f"Initial states mismatch"
            f"positions={len(self.initial_positions)}, "
            f"velocities={len(self.initial_velocities)}, "
            f"expected={2*self.N}"
        )

    def set_final_states(self, positions, velocities=None):
        """Set final states for all vehicles in flat format."""
        if velocities is None:
            velocities = np.zeros((self.N, 2))

        # Convert to flat vectors
        self.final_positions = positions.flatten()
        self.final_velocities = velocities.flatten()

        assert len(self.final_positions) == len(self.final_velocities) == 2 * self.N, (
            f"Final states mismatch"
            f"positions={len(self.final_positions)}, "
            f"velocities={len(self.final_velocities)}, "
            f"expected={2*self.N}"
        )

    def generate_trajectories(self, max_iterations=15):
        """Main method to generate collision-free trajectories using SCP."""
        is_feasible = False

        start_time = time.time()

        self._precompute_constraint_matrices()
        accelerations_flat = self._solve_initial_trajectory()

        init_guess_positions, _ = self._compute_positions_velocities(
            accelerations_flat.reshape(self.N, self.K, 2)
        )

        is_feasible = self._fast_check_avoidance_constraints(init_guess_positions)

        # SCP iterations
        iteration = 0
        converged = False

        # TODO: fast check --> exit if already no collision

        while iteration < max_iterations and not converged and not is_feasible:
            print(f"SCP Iteration {iteration+1}")

            new_accelerations_flat = self._solve_with_avoidance_constraints(accelerations_flat)

            rel_step_norm = np.linalg.norm(
                new_accelerations_flat - accelerations_flat
            ) / np.linalg.norm(accelerations_flat)
            print(rel_step_norm)
            if rel_step_norm <= self.convergence_tolerance:
                converged = True
                print(f"Converged after {iteration+1} iterations.")

            accelerations_flat = new_accelerations_flat
            iteration += 1

        accelerations = accelerations_flat.reshape(self.N, self.K, 2)
        positions, velocities = self._compute_positions_velocities(accelerations)

        self.trajectories = {
            "positions": positions,  # Shape (N, K, 2)
            "velocities": velocities,  # Shape (N, K, 2)
            "accelerations": accelerations,  # Shape (N, K, 2)
        }

        end_time = time.time()
        print(f"Trajectory generation completed in {end_time - start_time:.3f} seconds")

        return self.trajectories

    def _precompute_constraint_matrices(self):
        N, K, h = self.N, self.K, self.h
        I2 = sp.eye(2, format="csc")
        IN = sp.eye(N, format="csc")

        # -------- jerk --------
        self.C_jerk = _jerk_matrix_sparse(N, K, h)
        self.l_jerk = np.full(2 * N * (K - 1), self.jerk_min, dtype=float)
        self.u_jerk = np.full(2 * N * (K - 1), self.jerk_max, dtype=float)

        # -------- acceleration --------
        self.C_acc = sp.eye(2 * N * K, format="csc")
        self.l_acc = np.full(2 * N * K, self.acc_min, dtype=float)
        self.u_acc = np.full(2 * N * K, self.acc_max, dtype=float)

        # -------- velocities --------
        diags_T = [np.ones(K - d, dtype=float) for d in range(0, K)]
        offs_T = [-d for d in range(0, K)]
        T = sp.diags(diags_T, offs_T, shape=(K, K), format="csc")  # KxK, incl. diag

        C_vel_block = h * sp.kron(T, I2, format="csc")  # 2K x 2K
        self.C_vel = sp.kron(IN, C_vel_block, format="csc")  # 2NK x 2NK

        # Bounds: for k=0..K-2 -> box; k=K-1 -> equality
        self.l_vel = np.empty(2 * N * K, dtype=float)
        self.u_vel = np.empty(2 * N * K, dtype=float)

        v0 = self.initial_velocities.reshape(N, 2)
        vf = self.final_velocities.reshape(N, 2)

        for i in range(N):
            base = 2 * i * K
            for k_idx in range(K):
                rx = base + 2 * k_idx
                ry = base + 2 * k_idx + 1
                if k_idx < K - 1:
                    self.l_vel[rx] = self.vel_min - v0[i, 0]
                    self.u_vel[rx] = self.vel_max - v0[i, 0]
                    self.l_vel[ry] = self.vel_min - v0[i, 1]
                    self.u_vel[ry] = self.vel_max - v0[i, 1]
                else:  # final equality
                    self.l_vel[rx] = self.u_vel[rx] = vf[i, 0] - v0[i, 0]
                    self.l_vel[ry] = self.u_vel[ry] = vf[i, 1] - v0[i, 1]

        # -------- positions (new) --------
        diags_S = [np.full(K - d, h * h * (d + 0.5), dtype=float) for d in range(0, K)]
        offs_S = [-d for d in range(0, K)]
        S = sp.diags(diags_S, offs_S, shape=(K, K), format="csc")  # KxK, incl. diag

        C_pos_block = sp.kron(S, I2, format="csc")  # 2K x 2K
        self.C_pos = sp.kron(IN, C_pos_block, format="csc")  # 2NK x 2NK

        self.l_pos = np.empty(2 * N * K, dtype=float)
        self.u_pos = np.empty(2 * N * K, dtype=float)

        p0 = self.initial_positions.reshape(N, 2)
        v0 = self.initial_velocities.reshape(N, 2)  # reuse
        pf = self.final_positions.reshape(N, 2)
        pmin, pmax = self.pos_min, self.pos_max  # shape (2,)

        for i in range(N):
            base = 2 * i * K
            for k_idx in range(K):
                # known offset p0 + h*k*v0 (two components)
                off_x = p0[i, 0] + h * (k_idx + 1) * v0[i, 0]
                off_y = p0[i, 1] + h * (k_idx + 1) * v0[i, 1]
                rx = base + 2 * k_idx
                ry = base + 2 * k_idx + 1
                if k_idx < K - 1:
                    self.l_pos[rx] = pmin[0] - off_x
                    self.u_pos[rx] = pmax[0] - off_x
                    self.l_pos[ry] = pmin[1] - off_y
                    self.u_pos[ry] = pmax[1] - off_y
                else:  # final equality
                    self.l_pos[rx] = self.u_pos[rx] = pf[i, 0] - off_x
                    self.l_pos[ry] = self.u_pos[ry] = pf[i, 1] - off_y

        expected_nnz_acc = 2 * N * K
        expected_nnz_jerk = 4 * N * (K - 1)
        expected_nnz_vel = N * K * (K + 1)
        expected_nnz_pos = N * K * (K + 1)

        # --- Size checks ---
        assert (
            self.C_acc.shape == (2 * N * K, 2 * N * K)
            and self.l_acc.shape == (2 * N * K,)
            and self.u_acc.shape == (2 * N * K,)
        ), (
            f"C_acc expects {(2*N*K, 2*N*K)}, got {self.C_acc.shape}; "
            f"l_acc {(2*N*K,)}, got {self.l_acc.shape}; u_acc {(2*N*K,)}, got {self.u_acc.shape}"
        )

        assert (
            self.C_jerk.shape == (2 * N * (K - 1), 2 * N * K)
            and self.l_jerk.shape == (2 * N * (K - 1),)
            and self.u_jerk.shape == (2 * N * (K - 1),)
        ), (
            f"C_jerk expects {(2*N*(K-1), 2*N*K)}, got {self.C_jerk.shape}; "
            f"l_jerk {(2*N*(K-1),)}, got {self.l_jerk.shape}; u_jerk {(2*N*(K-1),)}, got {self.u_jerk.shape}"
        )

        assert (
            self.C_vel.shape == (2 * N * K, 2 * N * K)
            and self.l_vel.shape == (2 * N * K,)
            and self.u_vel.shape == (2 * N * K,)
        ), (
            f"C_vel expects {(2*N*K, 2*N*K)}, got {self.C_vel.shape}; "
            f"l_vel {(2*N*K,)}, got {self.l_vel.shape}; u_vel {(2*N*K,)}, got {self.u_vel.shape}"
        )

        assert (
            self.C_pos.shape == (2 * N * K, 2 * N * K)
            and self.l_pos.shape == (2 * N * K,)
            and self.u_pos.shape == (2 * N * K,)
        ), (
            f"C_pos expects {(2*N*K, 2*N*K)}, got {self.C_pos.shape}; "
            f"l_pos {(2*N*K,)}, got {self.l_pos.shape}; u_pos {(2*N*K,)}, got {self.u_pos.shape}"
        )

        # --- nnz checks ---
        assert (
            self.C_acc.nnz == expected_nnz_acc
        ), f"C_acc nnz={self.C_acc.nnz}, expected {expected_nnz_acc}"
        assert (
            self.C_jerk.nnz == expected_nnz_jerk
        ), f"C_jerk nnz={self.C_jerk.nnz}, expected {expected_nnz_jerk}"
        assert (
            self.C_vel.nnz == expected_nnz_vel
        ), f"C_vel nnz={self.C_vel.nnz}, expected {expected_nnz_vel}"
        assert (
            self.C_pos.nnz == expected_nnz_pos
        ), f"C_pos nnz={self.C_pos.nnz}, expected {expected_nnz_pos}"

        # --- format checks (CSC is best for OSQP) ---
        assert (
            sp.isspmatrix_csc(self.C_acc)
            and sp.isspmatrix_csc(self.C_jerk)
            and sp.isspmatrix_csc(self.C_vel)
            and sp.isspmatrix_csc(self.C_pos)
        ), "All constraint matrices must be CSC for OSQP."

    def _solve_initial_trajectory(self):
        """Solve initial trajectory without avoidance constraints."""

        problem = osqp.OSQP()

        # OSQP takes the format: 1/2 x^T P x => factor 2.0
        P = sp.csc_matrix(np.eye(2 * self.N * self.K)) * 2.0
        q = np.array([0] * 2 * self.N * self.K)

        C = sp.vstack(
            [
                self.C_jerk,
                self.C_acc,
                self.C_vel,
                self.C_pos,
            ],
            format="csc",
        )

        l_vec = np.hstack(
            [
                self.l_jerk,
                self.l_acc,
                self.l_vel,
                self.l_pos,
            ]
        )

        u_vec = np.hstack(
            [
                self.u_jerk,
                self.u_acc,
                self.u_vel,
                self.u_pos,
            ]
        )

        problem.setup(P=P, q=q, A=C, l=l_vec, u=u_vec, verbose=False)

        results = problem.solve()
        if results.info.status_val not in (1, 2):  # Solved / Solved Inaccurate
            print("not feasible")
            raise RuntimeError(f"OSQP failed: {results.info.status}")

        accelerations_flat = results.x

        return accelerations_flat

    def _compute_positions_velocities(self, accelerations):
        """
        Compute positions and velocities from accelerations via affine constraints.
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

        if not sp.issparse(A_collision):
            A_collision = sp.csc_matrix(A_collision)

        P = sp.csc_matrix(np.eye(2 * self.N * self.K)) * 2.0
        q = np.array([0] * 2 * self.N * self.K)

        A = sp.vstack(
            [
                self.C_jerk,
                self.C_acc,
                self.C_vel,
                self.C_pos,
                A_collision,
            ],
            format="csc",
        )

        l_vec = np.hstack(
            [
                self.l_jerk,
                self.l_acc,
                self.l_vel,
                self.l_pos,
                l_collision,
            ]
        )

        u = np.hstack(
            [
                self.u_jerk,
                self.u_acc,
                self.u_vel,
                self.u_pos,
                u_collision,
            ]
        )

        problem = osqp.OSQP()
        problem.setup(P=P, q=q, A=A, l=l_vec, u=u, verbose=False, warm_start=True, max_iter=10000)
        problem.warm_start(x=accelerations_flat)

        result = problem.solve()
        if result.info.status_val not in (1, 2):
            print(f"Warning: OSQP status {result.info.status}")

        new_accelerations_flat = result.x

        return new_accelerations_flat

    def _add_collision_constraints(self, previous_solution):
        """
        Sparse + faster version.
        Builds A_collision as CSC directly via (rows, cols, vals) lists.
        Keeps identical functionality and row ordering as the original.
        """
        N, K, h = self.N, self.K, self.h
        Rm = self.R

        # 1) Recompute previous positions (same as before)
        prev_positions, _ = self._accelerations_to_positions_velocities(
            previous_solution
        )  # (N, K, 2)

        # 2) Counts
        n_pairs = (N * (N - 1)) // 2
        n_constraints = n_pairs * K
        n_vars = 2 * N * K

        # 3) Pre-fetch initial states into 2D arrays for faster access
        init_pos = self.initial_positions.reshape(N, 2)
        init_vel = self.initial_velocities.reshape(N, 2)

        # 4) Sparse builders
        rows, cols, vals = [], [], []
        l_collision = np.full(n_constraints, Rm, dtype=float)
        u_collision = np.full(n_constraints, np.inf, dtype=float)

        row_idx = 0

        # Helper: column bases per vehicle (start index of that vehicle's 2K accel vars)
        base_cols = np.arange(N, dtype=int) * (2 * K)

        # Main loops: keep exact ordering: for k in 0..K-1, for i, for j>i
        for k in range(K):
            # Precompute the weights for all a[0..k-1] at this time k: h^2 * (k - m - 0.5)
            if k > 0:
                m_idx = np.arange(k)
                w = (h * h) * (k - m_idx - 0.5)  # shape (k,)
            else:
                w = None  # no accel contribution at k=0

            for i in range(N):
                for j in range(i + 1, N):
                    # ----- Linearization direction from previous positions -----
                    pi_prev = prev_positions[i, k]  # (2,)
                    pj_prev = prev_positions[j, k]  # (2,)
                    diff = pi_prev - pj_prev
                    dist = np.hypot(diff[0], diff[1])

                    if dist < 1e-6:
                        # Avoid numerical issues by picking a random direction
                        angle = np.random.uniform(0.0, 2.0 * np.pi)
                        eta = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0
                    else:
                        eta = diff / dist  # (2,)

                    # ----- Fill sparse coeffs for this constraint row -----
                    if k > 0:
                        # vehicle i columns for m=0..k-1
                        base_i = base_cols[i]
                        # x columns are base_i + 2*m, y columns are base_i + 2*m + 1
                        # Append x coefficients (eta[0] * w[m])
                        rows.extend([row_idx] * k)
                        cols.extend(base_i + 2 * m_idx)
                        vals.extend(eta[0] * w)

                        # Append y coefficients (eta[1] * w[m])
                        rows.extend([row_idx] * k)
                        cols.extend(base_i + 2 * m_idx + 1)
                        vals.extend(eta[1] * w)

                        # vehicle j columns (negative contributions)
                        base_j = base_cols[j]
                        rows.extend([row_idx] * k)
                        cols.extend(base_j + 2 * m_idx)
                        vals.extend(-eta[0] * w)

                        rows.extend([row_idx] * k)
                        cols.extend(base_j + 2 * m_idx + 1)
                        vals.extend(-eta[1] * w)

                    # ----- RHS (same formula as your dense version) -----
                    # Initial position + velocity contributions
                    p_init_i = init_pos[i]  # (2,)
                    p_init_j = init_pos[j]  # (2,)
                    v_init_i = init_vel[i]  # (2,)
                    v_init_j = init_vel[j]  # (2,)

                    init_pos_contrib = eta @ (p_init_i - p_init_j)
                    init_vel_contrib = eta @ (v_init_i - v_init_j) * (k * h)

                    # Linearization term (ηᵀ(p_i_prev - p_j_prev) - ||p_i_prev - p_j_prev||)
                    linearization_term = eta @ (pi_prev - pj_prev) - dist

                    rhs = Rm + linearization_term - (init_pos_contrib + init_vel_contrib)
                    l_collision[row_idx] = rhs

                    row_idx += 1

        # Build sparse matrix
        A_collision = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars)).tocsc()

        return A_collision, l_collision, u_collision

    def _accelerations_to_positions_velocities(self, accelerations_flat):
        """Convert flat acceleration array to positions and velocities."""
        N, K, h = self.N, self.K, self.h

        # Initialize arrays
        positions = np.zeros((N, K, 2))
        velocities = np.zeros((N, K, 2))

        # Set initial conditions
        for i in range(N):
            positions[i, 0, 0] = self.initial_positions[2 * i]
            positions[i, 0, 1] = self.initial_positions[2 * i + 1]
            velocities[i, 0, 0] = self.initial_velocities[2 * i]
            velocities[i, 0, 1] = self.initial_velocities[2 * i + 1]

        # Compute trajectories using the exact same method as first implementation
        for i in range(N):
            for k in range(1, K):
                # Velocity: v[k] = v[0] + h * sum(a[j] for j in 0..k-1)
                velocities[i, k, 0] = velocities[i, 0, 0]
                velocities[i, k, 1] = velocities[i, 0, 1]
                for j in range(k):
                    ax = accelerations_flat[2 * i * K + 2 * j]
                    ay = accelerations_flat[2 * i * K + 2 * j + 1]
                    velocities[i, k, 0] += h * ax
                    velocities[i, k, 1] += h * ay

                # Position: p[k] = p[0] + k*h*v[0] + sum(h^2*(k-j-0.5)*a[j] for j in 0..k-1)
                positions[i, k, 0] = positions[i, 0, 0] + k * h * velocities[i, 0, 0]
                positions[i, k, 1] = positions[i, 0, 1] + k * h * velocities[i, 0, 1]
                for j in range(k):
                    ax = accelerations_flat[2 * i * K + 2 * j]
                    ay = accelerations_flat[2 * i * K + 2 * j + 1]
                    positions[i, k, 0] += h * h * (k - j - 0.5) * ax
                    positions[i, k, 1] += h * h * (k - j - 0.5) * ay

        return positions, velocities

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
                for j in range(i + 1, self.N):
                    dist = np.linalg.norm(pos_k[i] - pos_k[j])
                    if dist < self.R - 0.01:
                        print(
                            f"Avoidance constraint violation at timestep {k} between vehicles {i} and {j}: distance = {dist:.3f}"
                        )
                        return False
        return True

        # Put this helper inside class SCP (anywhere above visualize_trajectories)

    def _quadrant_colors(self, center=(10.0, 10.0)):
        """
        Assign a color to each craft based on the quadrant of its initial position.
        Quadrants (relative to `center`):
        Q0: top-right, Q1: top-left, Q2: bottom-left, Q3: bottom-right
        Returns: list of RGB tuples length N.
        """
        cx, cy = center
        # Same palette as position_generator.quadrant_colors
        palette = [(0.17, 0.28, 0.46), (0.54, 0.31, 0.56), (1.00, 0.39, 0.38), (1.00, 0.65, 0.00)]
        init_pos = self.initial_positions.reshape(self.N, 2)
        colors = []
        for x, y in init_pos:
            if x >= cx and y >= cy:
                q = 0
            elif x < cx and y >= cy:
                q = 1
            elif x < cx and y < cy:
                q = 2
            else:
                q = 3
            colors.append(palette[q])
        return colors

    # Replace your existing visualize_trajectories with this version
    def visualize_trajectories(self, show_animation=False, save_path="trajectories.pdf"):
        """Visualize the generated 2D trajectories with same styling as position_generator."""
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")

        import matplotlib as mpl
        from matplotlib.patches import Circle, Rectangle

        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 16,
                "axes.titlesize": 20,
                "axes.labelsize": 18,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
            }
        )

        positions = self.trajectories["positions"]  # (N, K, 2)
        colors = self._quadrant_colors(center=(10.0, 10.0))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")

        # Bounds (draw the 20x20 boundary box like in position_generator)
        xmin, ymin, xmax, ymax = self.space_dims
        ax.set_xlim(xmin - 1, xmax + 1)
        ax.set_ylim(ymin - 1, ymax + 1)
        boundary = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
        ax.add_patch(boundary)

        # Corner circles (centers at 3.5, 16.5; radius = 2.5)
        circle_centers = np.array(
            [
                [xmin + 3.5, ymin + 3.5],
                [xmax - 3.5, ymin + 3.5],
                [xmin + 3.5, ymax - 3.5],
                [xmax - 3.5, ymax - 3.5],
            ]
        )
        for c in circle_centers:
            ax.add_patch(
                Circle(c, 2.5, linewidth=1.5, edgecolor="grey", facecolor="none", alpha=0.7)
            )

        # Central diamond (square side 6m rotated 45°)
        diamond_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        diamond_size = 6.0 / np.sqrt(2.0)  # distance center->vertex
        diamond_vertices = np.array(
            [
                [diamond_center[0], diamond_center[1] + diamond_size],
                [diamond_center[0] + diamond_size, diamond_center[1]],
                [diamond_center[0], diamond_center[1] - diamond_size],
                [diamond_center[0] - diamond_size, diamond_center[1]],
            ]
        )
        dx = np.append(diamond_vertices[:, 0], diamond_vertices[0, 0])
        dy = np.append(diamond_vertices[:, 1], diamond_vertices[0, 1])
        ax.plot(dx, dy, linewidth=1.5, color="grey", alpha=0.7)

        # Plot start markers (triangles), end markers (squares), and paths
        for i in range(self.N):
            # Start
            ax.scatter(
                positions[i, 0, 0],
                positions[i, 0, 1],
                marker="o",
                s=100,
                color=colors[i],
                label=None,
            )
            ax.add_patch(Circle(positions[i, 0], self.R, color=colors[i], alpha=0.1, fill=True))
            # End
            ax.scatter(
                positions[i, -1, 0],
                positions[i, -1, 1],
                marker="s",
                s=100,
                color=colors[i],
                label=None,
            )
            ax.add_patch(Circle(positions[i, -1], self.R, color=colors[i], alpha=0.1, fill=True))
            # Full trajectory
            ax.plot(
                positions[i, :, 0], positions[i, :, 1], color=colors[i], linewidth=1.5, alpha=0.8
            )

        # Legend (simplified start vs stop like in position_generator)
        import matplotlib.lines as mlines

        start_handle = mlines.Line2D(
            [], [], color="black", marker="o", linestyle="None", markersize=8, label="Start"
        )
        stop_handle = mlines.Line2D(
            [], [], color="black", marker="s", linestyle="None", markersize=8, label="Stop"
        )
        ax.legend(
            handles=[start_handle, stop_handle],
            loc="lower right",
            bbox_to_anchor=(0.95, 0.05),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.6,
            borderpad=1.2,
            labelspacing=1.0,
            handlelength=1.8,
            handletextpad=0.8,
            fontsize=11,
        )

        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_title("2D Collision-Free Trajectories")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight", format="pdf")
        plt.show()

        return fig, ax

    def visualize_time_snapshots(self, num_snapshots=5, save_path=None):
        """
        Visualize trajectories as a series of time snapshots.
        Colors now match quadrant-based colors used elsewhere.
        """
        if self.trajectories is None:
            raise ValueError("Trajectories not generated yet")

        positions = self.trajectories["positions"]
        K = self.K

        # Use the same quadrant-based colors as other plots
        colors = self._quadrant_colors(center=(10.0, 10.0))

        # Select frames to visualize evenly spaced in time
        frame_indices = np.linspace(0, K - 1, num_snapshots, dtype=int)

        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_snapshots, figsize=(15, 3))

        # Plot each frame
        for f, frame_idx in enumerate(frame_indices):
            ax = axes[f]

            # Set equal aspect ratio
            ax.set_aspect("equal")

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

            # Plot circles and paths with the quadrant colors
            for i in range(self.N):
                pos = positions[i, frame_idx]  # (x, y)

                # Vehicle radius: 0.20m
                vehicle_circle = Circle(pos, 0.20, color=colors[i], alpha=0.7)
                safety_circle = Circle(pos, self.R, color=colors[i], alpha=0.1, fill=True)
                ax.add_patch(vehicle_circle)
                ax.add_patch(safety_circle)

                # Trajectory up to current frame
                if frame_idx > 0:
                    traj_x = positions[i, : frame_idx + 1, 0]
                    traj_y = positions[i, : frame_idx + 1, 1]
                    ax.plot(traj_x, traj_y, "-", color=colors[i], alpha=0.7, linewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")

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
        space_dims=[-5, -5, 500, 200],
    )

    # Set initial positions (spread out)
    initial_positions = np.array([[-2, -2], [0, -2], [2, -2]])

    # Set final positions (crossed over)
    final_positions = np.array([[2, 2], [0, 2], [-2, 2]])

    # Set the states
    planner.set_initial_states(initial_positions)
    planner.set_final_states(final_positions)

    # Generate trajectories
    trajectories = planner.generate_trajectories(max_iterations=10)

    # Visualize
    planner.visualize_trajectories(show_animation=False)
    planner.visualize_time_snapshots(num_snapshots=4)
