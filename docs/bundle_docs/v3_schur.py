"""Normal equations, block structure, and exactness of the Schur complement."""

import numpy as np
import pyceres
import docs.bundle_docs.toy as toy
from scipy.sparse import csr_matrix

np.set_printoptions(precision=4, suppress=True, linewidth=130)

s = toy.build(n_cam=5, n_pt=40)
prob, pts, poses = s["problem"], s["points"], s["poses"]

# Column order we impose: all points first, then all camera poses.
opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks(pts + poses)
crs = prob.evaluate_jacobian(opts)
J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
f = np.asarray(prob.evaluate_residuals(opts))

np_, nc = 3 * len(pts), 3 * len(poses)  # tangent sizes (rotation held const.)
print(f"J: {J.shape[0]} residuals x {J.shape[1]} params  ({np_} point + {nc} camera)")
print(f"J nnz {J.nnz} -> density {100 * J.nnz / np.prod(J.shape):.3f}%")
print(f"cost C(x) = 0.5*||f||^2 = {0.5 * f @ f:.6f}")

H = (J.T @ J).toarray()
g = J.T @ f
lam = 1e-2
D2 = np.diag(np.diag(H))  # Ceres' JACOBI-scaled damping D^T D
A = H + lam * D2

# blocks: rows/cols [0:np_] = points (C), [np_:] = cameras (B)
C = A[:np_, :np_]
E = A[np_:, :np_]
B = A[np_:, np_:]
gp, gc = g[:np_], g[np_:]
off = np.abs(C - np.diag(np.diag(C)))
blk = np.zeros_like(C, dtype=bool)
for i in range(len(pts)):
    blk[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = True
print(
    f"C (point-point) is block diagonal: max |off-block| = {np.abs(C[~blk]).max():.3e}"
)

# --- (1) direct solve of the full damped normal equations
dx_full = np.linalg.solve(A, -g)

# --- (2) Schur: S = B - E C^-1 E^T, solve cameras, back-substitute points
Cinv = np.zeros_like(C)
for i in range(len(pts)):
    sl = slice(3 * i, 3 * i + 3)
    Cinv[sl, sl] = np.linalg.inv(C[sl, sl])  # 3x3 inverses only
S = B - E @ Cinv @ E.T
rhs = -gc + E @ Cinv @ gp
dc = np.linalg.solve(S, rhs)
dp = Cinv @ (-gp - E.T @ dc)
dx_schur = np.concatenate([dp, dc])

print(
    f"\nS is {S.shape[0]}x{S.shape[0]} (reduced camera system) vs full {A.shape[0]}x{A.shape[0]}"
)
print(
    f"max |dx_full - dx_schur| = {np.max(np.abs(dx_full - dx_schur)):.3e}"
    f"   relative {np.max(np.abs(dx_full - dx_schur)) / np.max(np.abs(dx_full)):.3e}"
)

# --- predicted vs actual cost decrease for this step (GN model quality)
pred = -(g @ dx_schur + 0.5 * dx_schur @ H @ dx_schur)
print(f"\npredicted decrease from the quadratic model: {pred:.6f}")
