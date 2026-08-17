"""Small synthetic BA on COLMAP's own cost functions, for structural checks."""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf

rng = np.random.default_rng(0)


def build(seed=0, n_cam=4, n_pt=30, noise_px=0.5, perturb=True):
    rng = np.random.default_rng(seed)
    cam = pycolmap.Camera.create(1, "SIMPLE_RADIAL", 1000.0, 1000, 800)
    cam.params = np.array([1000.0, 500.0, 400.0, -0.02])
    pts = rng.uniform([-2, -2, 8], [2, 2, 12], size=(n_pt, 3))
    poses = []
    for i in range(n_cam):
        C = np.array([1.5 * i - 2.0, 0.3 * (-1) ** i, 0.0])
        R = pycolmap.Rotation3d(np.array([0.02 * i, 0.01 * i, 0.005 * i]))
        poses.append(pycolmap.Rigid3d(R, -(R.matrix() @ C)))
    obs = []  # (cam_idx, pt_idx, uv)
    for ci, T in enumerate(poses):
        for pi, X in enumerate(pts):
            Xc = T * X
            uv = cam.img_from_cam(Xc[None, :2] / Xc[2])[0]
            if 0 <= uv[0] < 1000 and 0 <= uv[1] < 800:
                obs.append((ci, pi, uv + rng.normal(0, noise_px, 2)))
    # parameter blocks (mutable copies)
    pt_blocks = [pts[i].copy() for i in range(n_pt)]
    pose_blocks = [np.array(T.params, dtype=float) for T in poses]
    cam_block = np.array(cam.params, dtype=float)
    if perturb:
        for b in pt_blocks:
            b += rng.normal(0, 0.05, 3)
        for b in pose_blocks:
            b[4:] += rng.normal(0, 0.01, 3)
    prob = pyceres.Problem()
    keep = []
    for ci, pi, uv in obs:
        c = cf.ReprojErrorCost(cam.model_id, uv)
        keep.append(c)
        prob.add_residual_block(c, None, [pt_blocks[pi], pose_blocks[ci], cam_block])
    for b in pose_blocks:
        prob.set_manifold(b, pyceres.QuaternionManifold())  # placeholder, fixed below
    return dict(
        prob=prob,
        keep=keep,
        pts=pt_blocks,
        poses=pose_blocks,
        cam=cam_block,
        obs=obs,
        camera=cam,
    )
