"""Iterative (matrix-free) Schur: PCG on S without ever forming S."""

import numpy as np
import pyceres
import docs.bundle_docs.toy as toy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg

s = toy.build(n_cam=8, n_pt=150)
prob, pts, poses = s["problem"], s["points"], s["poses"]
opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks(pts + poses)
crs = prob.evaluate_jacobian(opts)
J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
f = np.asarray(prob.evaluate_residuals(opts))
H = (J.T @ J).toarray()
g = J.T @ f
lam = 1e-2
A = H + lam * np.diag(np.diag(H))
n = 3 * len(pts)
C, E, B = A[:n, :n], A[n:, :n], A[n:, n:]
gp, gc = g[:n], g[n:]
Cinv = np.zeros_like(C)
for i in range(len(pts)):
    sl = slice(3 * i, 3 * i + 3)
    Cinv[sl, sl] = np.linalg.inv(C[sl, sl])

S = B - E @ Cinv @ E.T
rhs = -gc + E @ Cinv @ gp
dc_direct = np.linalg.solve(S, rhs)

# matrix-free:  S v = B v - E (C^-1 (E^T v)),  S never assembled
mv = lambda v: B @ v - E @ (Cinv @ (E.T @ v))
op = LinearOperator(S.shape, matvec=mv, dtype=float)
Mjac = np.diag(1.0 / np.diag(S))  # SCHUR_JACOBI preconditioner (block-diag of S)
M = LinearOperator(S.shape, matvec=lambda v: Mjac @ v, dtype=float)
it = [0]
dc_cg, info = cg(
    op, rhs, rtol=1e-12, M=M, callback=lambda xk: it.__setitem__(0, it[0] + 1)
)
print(
    f"reduced camera system S: {S.shape[0]}x{S.shape[0]}, full system {A.shape[0]}x{A.shape[0]}"
)
print(f"cond(S) = {np.linalg.cond(S):.3e}")
print(f"PCG (Jacobi) iterations: {it[0]}, info={info}")
print(
    f"max |dc_direct - dc_pcg| = {np.max(np.abs(dc_direct - dc_cg)):.3e}"
    f"  (relative {np.max(np.abs(dc_direct - dc_cg)) / np.max(np.abs(dc_direct)):.3e})"
)
print(f"S dense storage {S.nbytes / 1024:.1f} kB vs full A {A.nbytes / 1024:.1f} kB")
