"""Sparsity of S: is S_ij nonzero iff cameras i and j share a point?"""

import numpy as np
import pyceres
import toy
from scipy.sparse import csr_matrix

s = toy.build(n_cam=10, n_pt=200, seed=3)
prob, pts, poses, obs = s["problem"], s["points"], s["poses"], s["obs"]
opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks(pts + poses)
crs = prob.evaluate_jacobian(opts)
J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
f = np.asarray(prob.evaluate_residuals(opts))
H = (J.T @ J).toarray()
A = H + 1e-2 * np.diag(np.diag(H))
n = 3 * len(pts)
C, E, B = A[:n, :n], A[n:, :n], A[n:, n:]
Cinv = np.zeros_like(C)
for i in range(len(pts)):
    sl = slice(3 * i, 3 * i + 3)
    Cinv[sl, sl] = np.linalg.inv(C[sl, sl])
S = B - E @ Cinv @ E.T
nc = len(poses)
Sblk = np.array([
    [np.abs(S[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]).max() > 1e-12 for j in range(nc)]
    for i in range(nc)
])
seen = [set() for _ in range(nc)]
for ci, pi, _ in obs:
    seen[ci].add(pi)
share = np.array([[len(seen[i] & seen[j]) > 0 for j in range(nc)] for i in range(nc)])
print("cameras:", nc, " points:", len(pts), " observations:", len(obs))
print("S block pattern == 'share >=1 point' pattern:", bool((Sblk == share).all()))
print("S block fill: %d / %d nonzero blocks" % (Sblk.sum(), nc * nc))
print(
    "H (full) would be %dx%d; S is %dx%d"
    % (A.shape[0], A.shape[0], S.shape[0], S.shape[0])
)
