"""Covariance weighting: is the residual really the Mahalanobis whitening?"""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf

cam = pycolmap.Camera.create_from_model_id(
    1, pycolmap.CameraModelId.SIMPLE_RADIAL, 1000.0, 1000, 800
)
cam.params = np.array([1000.0, 500.0, 400.0, -0.02])
X = np.array([0.7, -0.4, 10.0])
T = pycolmap.Rigid3d(
    pycolmap.Rotation3d(np.array([0.05, -0.03, 0.02])), np.array([0.2, -0.1, 0.05])
)
uv_obs = cam.img_from_cam(T * X) + np.array([1.3, -0.7])
cov = np.array([[0.36, 0.12], [0.12, 0.81]])  # anisotropic, correlated


def resid(cost):
    p = pyceres.Problem()
    pt = X.copy()
    pose = np.array(T.params, float)
    par = np.array(cam.params, float)
    p.add_residual_block(cost, None, [pt, pose, par])
    o = pyceres.EvaluateOptions()
    o.apply_loss_function = False
    return np.asarray(p.evaluate_residuals(o))


raw = resid(cf.ReprojErrorCost(cam.model, uv_obs))
wht = resid(cf.ReprojErrorCost(cam.model, cov, uv_obs))
L = np.linalg.cholesky(np.linalg.inv(cov))  # inv(cov) = L L^T
pred = L.T @ raw
print("raw residual        f      =", raw)
print("weighted residual   r      =", wht)
print("chol(cov^-1)^T f           =", pred, "   max diff", np.max(np.abs(wht - pred)))
print("f^T cov^-1 f               =", raw @ np.linalg.solve(cov, raw))
print("||r||^2 (Mahalanobis)      =", wht @ wht)
