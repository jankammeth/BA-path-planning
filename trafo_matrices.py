# pip install sympy
import sympy as sp

def axis_vel_transform(K, h):
    """K x K lower-triangular matrix T_v with (T_v @ a)[k] = sum_{j<k} h a[j]"""
    T = sp.zeros(K, K)
    for k in range(1, K):
        for j in range(k):
            T[k, j] = h
    return T

def axis_pos_transform(K, h):
    """K x K lower-triangular matrix T_p with (T_p @ a)[k] = sum_{j<k} h^2*(k-j-1/2) a[j]"""
    T = sp.zeros(K, K)
    for k in range(1, K):
        for j in range(k):
            T[k, j] = h**2 * (k - j - sp.Rational(1,2))
    return T

import sympy as sp

def interleave_xy_block(T_axis):
    # Returns a 2K x 2K block for interleaved [ax0, ay0, ax1, ay1, ...]
    return sp.kronecker_product(T_axis, sp.eye(2))

def blockdiag_over_vehicles(B_per_vehicle, N):
    """Block-diagonal over N vehicles."""
    return sp.diag(*([B_per_vehicle] * N))

def decision_symbols(N, K):
    """Return the symbolic decision vector x (size 2NK) with interleaved (ax, ay)."""
    entries = []
    for i in range(N):
        for k in range(K):
            entries.append(sp.symbols(f'a_x^{i}_{k}'))
            entries.append(sp.symbols(f'a_y^{i}_{k}'))
    x = sp.Matrix(entries)
    return x

def jerk_matrix(N, K, h):
    """
    Build A_jerk so that (A_jerk @ x) gives [ (a_x[k+1]-a_x[k])/h , (a_y[k+1]-a_y[k])/h ] for each i,k.
    Size: (2*N*(K-1)) x (2*N*K)
    """
    rows = 2*N*(K-1)
    cols = 2*N*K
    A = sp.zeros(rows, cols)
    r = 0
    for i in range(N):
        base = 2*i*K
        for k in range(K-1):
            # x
            A[r, base + 2*k    ] = -1/h
            A[r, base + 2*(k+1)] =  1/h
            r += 1
            # y
            A[r, base + 2*k    +1] = -1/h
            A[r, base + 2*(k+1)+1] =  1/h
            r += 1
    return A

def accel_identity(N, K):
    """A_accel_constraints = I (2NK x 2NK)"""
    return sp.eye(2*N*K)

def accel_initial_selector(N, K):
    """
    Selects a^{(i)}[0] (x and y) for each vehicle.
    Size: (2N) x (2NK). Row layout: [ax^0[0], ay^0[0], ax^1[0], ay^1[0], ..., ax^{N-1}[0], ay^{N-1}[0]]
    """
    A = sp.zeros(2*N, 2*N*K)
    for i in range(N):
        base = 2*i*K
        A[2*i  , base + 0] = 1  # ax at k=0
        A[2*i+1, base + 1] = 1  # ay at k=0
    return A

def accel_final_selector(N, K):
    """Selects a^{(i)}[K-1] (x and y) for each vehicle. Size: (2N) x (2NK)"""
    A = sp.zeros(2*N, 2*N*K)
    for i in range(N):
        base = 2*i*K
        A[2*i  , base + 2*(K-1)  ] = 1  # ax at k=K-1
        A[2*i+1, base + 2*(K-1)+1] = 1  # ay at k=K-1
    return A

def build_transforms(N, K, h):
    """
    Returns:
      A_vel  : (2NK) x (2NK) mapping x -> stacked [v_x[k]-v_x[0], v_y[k]-v_y[0]] for all i,k
      A_pos  : (2NK) x (2NK) mapping x -> stacked [p_x[k]-p_x[0]-k h v_x[0], ...]
      A_jerk : (2N(K-1)) x (2NK)
      A_acc0 : (2N) x (2NK), A_accT : (2N) x (2NK), A_acc_box : (2NK) x (2NK)
    """
    Tv = axis_vel_transform(K, h)   # K x K
    Tp = axis_pos_transform(K, h)   # K x K

    Bv = interleave_xy_block(Tv)    # 2K x 2K
    Bp = interleave_xy_block(Tp)    # 2K x 2K

    A_vel  = blockdiag_over_vehicles(Bv, N)  # (2NK) x (2NK)
    A_pos  = blockdiag_over_vehicles(Bp, N)  # (2NK) x (2NK)
    A_jerk = jerk_matrix(N, K, h)            # (2N(K-1)) x (2NK)
    A_acc0 = accel_initial_selector(N, K)    # (2N) x (2NK)
    A_accT = accel_final_selector(N, K)      # (2N) x (2NK)
    A_acc_box = accel_identity(N, K)         # (2NK) x (2NK)

    return A_vel, A_pos, A_jerk, A_acc0, A_accT, A_acc_box

# ----- Example usage (choose integers for N, K so SymPy can size matrices) -----
N, K = 2, 4
h = sp.symbols('h', positive=True)

x = decision_symbols(N, K)                  # symbolic decision vector
A_vel, A_pos, A_jerk, A_acc0, A_accT, A_acc_box = build_transforms(N, K, h)

# Pretty-print one small block to verify shapes:
print("A_vel shape:", A_vel.shape)
print("A_pos shape:", A_pos.shape)
print("A_jerk shape:", A_jerk.shape)
print("A_acc0 shape:", A_acc0.shape, "A_accT shape:", A_accT.shape)
from sympy import pprint

# Show a few matrices symbolically
print("Decision vector x:")
pprint(x)

print("\nA_vel:")
pprint(A_vel)

print("\nA_pos:")
pprint(A_pos)

print("\nA_jerk:")
pprint(A_jerk)

print("\nA_acc0:")
pprint(A_acc0)

print("\nA_accT:")
pprint(A_accT)

from sympy import latex
#print("chi (LaTeX):")
#print(latex(x))

#print("A_jerk (LaTeX):")
#print(latex(A_jerk))

#print("A_vel (LaTeX):")
#print(latex(A_vel))

print("A_pos (LaTeX):")
print(latex(A_pos))

print("A_acc0 (LaTeX):")
print(latex(A_acc0))

print("A_accT (LaTeX):")
print(latex(A_accT))