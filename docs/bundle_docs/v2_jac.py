"""Check COLMAP's reprojection Jacobian against central finite differences."""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf

camera = pycolmap.Camera.create(1, "SIMPLE_RADIAL", 1000.0, 1000, 800)
camera.params = np.array([1000.0, 500.0, 400.0, -0.02])
X = np.array([0.7, -0.4, 10.0])
R = pycolmap.Rotation3d(np.array([0.05, -0.03, 0.02]))
T = pycolmap.Rigid3d(R, np.array([0.2, -0.1, 0.05]))
uv_obs = camera.img_from_cam(T * X) + np.array([0.4, -0.3])

pose = np.array(T.params, float)  # (qx,qy,qz,qw, tx,ty,tz)
cam = np.array(camera.params, float)
pt = X.copy()

prob = pyceres.Problem()
cost = cf.ReprojErrorCost(camera.model, uv_obs)
rid = prob.add_residual_block(cost, None, [pt, pose, cam])


def residual():
    return np.asarray(prob.evaluate_residuals(pyceres.EvaluateOptions()))


opts = pyceres.EvaluateOptions()
opts.set_parameter_blocks([pt, pose, cam])
crs = prob.evaluate_jacobian(opts)
from scipy.sparse import csr_matrix

J = csr_matrix(
    (crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols)
).toarray()

blocks = [("point3D", pt), ("cam_from_world", pose), ("camera_params", cam)]
Jfd = np.zeros_like(J)
col = 0
for _, b in blocks:
    for k in range(b.size):
        h = 1e-6 * max(1.0, abs(b[k]))
        old = b[k]
        b[k] = old + h
        rp = residual()
        b[k] = old - h
        rm = residual()
        b[k] = old
        Jfd[:, col] = (rp - rm) / (2 * h)
        col += 1

print("residual f(x) (px):", residual())
print("J shape:", J.shape, " (2 residuals x [3 point + 7 pose + 4 intrinsics])")
np.set_printoptions(precision=4, suppress=True, linewidth=140)
print("\nanalytic J (from Ceres):\n", J)
print("\nmax |J_ceres - J_finite_difference| =", np.max(np.abs(J - Jfd)))
print("relative:", np.max(np.abs(J - Jfd)) / np.max(np.abs(J)))
