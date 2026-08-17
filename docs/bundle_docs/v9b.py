"""Same check on a strip block, where cameras genuinely share only some points."""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf
from scipy.sparse import csr_matrix

rng = np.random.default_rng(7)
cam = pycolmap.Camera.create_from_model_id(
    1, pycolmap.CameraModelId.PINHOLE, 1000.0, 1000, 800
)
cam.params = np.array([1000.0, 1000.0, 500.0, 400.0])
n_cam, n_pt = 10, 300
pts = np.column_stack([
    rng.uniform(0, 20, n_pt),
    rng.uniform(-2, 2, n_pt),
    rng.uniform(9, 11, n_pt),
])
poses = []
for i in range(n_cam):  # nadir-ish strip, 2 m spacing
    C = np.array([2.0 * i, 0.0, 0.0])
    R = pycolmap.Rotation3d(np.array([0.0, 0.0, 0.0]))
    poses.append(pycolmap.Rigid3d(R, -(R.matrix() @ C)))
pt_blocks = [p + rng.normal(0, 0.02, 3) for p in pts]
pose_blocks = [np.array(T.params, float) for T in poses]
par = np.array(cam.params, float)
prob = pyceres.Problem()
keep = []
obs = []
for ci, T in enumerate(poses):
    for pi, X in enumerate(pts):
        Xc = T * X
        if Xc[2] <= 0:
            continue
        uv = cam.img_from_cam(Xc)
        if not (0 <= uv[0] < 1000 and 0 <= uv[1] < 800):
            continue
        c = cf.ReprojErrorCost(cam.model, uv + rng.normal(0, 0.5, 2))
        keep.append(c)
        prob.add_residual_block(c, None, [pt_blocks[pi], pose_blocks[ci], par])
        obs.append((ci, pi))
for b in pose_blocks:
    prob.set_manifold(b, pyceres.SubsetManifold(7, [0, 1, 2, 3]))
prob.set_parameter_block_constant(par)

opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks(pt_blocks + pose_blocks)
crs = prob.evaluate_jacobian(opts)
J = csr_matrix((crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols))
H = (J.T @ J).toarray()
A = H + 1e-2 * np.diag(np.diag(H))
n = 3 * n_pt
C, E, B = A[:n, :n], A[n:, :n], A[n:, n:]
Cinv = np.zeros_like(C)
for i in range(n_pt):
    sl = slice(3 * i, 3 * i + 3)
    blk = C[sl, sl]
    Cinv[sl, sl] = np.linalg.inv(blk) if np.linalg.cond(blk) < 1e14 else 0.0
S = B - E @ Cinv @ E.T
Sblk = np.array([
    [np.abs(S[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]).max() > 1e-9 for j in range(n_cam)]
    for i in range(n_cam)
])
seen = [set() for _ in range(n_cam)]
for ci, pi in obs:
    seen[ci].add(pi)
share = np.array([
    [len(seen[i] & seen[j]) > 0 for j in range(n_cam)] for i in range(n_cam)
])
print("observations:", len(obs))
print("S block pattern == co-visibility pattern:", bool((Sblk == share).all()))
print("S block fill: %d/%d = %.0f%%" % (Sblk.sum(), n_cam**2, 100 * Sblk.mean()))
print("co-visibility (bandedness):")
print(share.astype(int))
