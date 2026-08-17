"""Synthetic BA assembled from COLMAP's own ReprojErrorCost, via pyceres."""

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions as cf
from scipy.sparse import csr_matrix


def build(
    seed=0,
    n_cam=5,
    n_pt=40,
    noise_px=0.5,
    sigma_perturb=0.05,
    fix_rotation=True,
    loss=None,
):
    rng = np.random.default_rng(seed)
    camera = pycolmap.Camera.create(1, "SIMPLE_RADIAL", 1000.0, 1000, 800)
    camera.params = np.array([1000.0, 500.0, 400.0, -0.02])
    pts = rng.uniform([-3, -3, 8], [3, 3, 13], size=(n_pt, 3))
    poses = []
    for i in range(n_cam):
        C = np.array([1.5 * i - 3.0, 0.3 * (-1) ** i, 0.0])
        R = pycolmap.Rotation3d(np.array([0.03 * i, 0.02 * i, 0.01 * i]))
        poses.append(pycolmap.Rigid3d(R, -(R.matrix() @ C)))
    obs = []
    for ci, T in enumerate(poses):
        for pi, X in enumerate(pts):
            Xc = T * X
            if Xc[2] <= 0:
                continue
            uv = camera.img_from_cam(Xc)
            if 0 <= uv[0] < 1000 and 0 <= uv[1] < 800:
                obs.append((ci, pi, uv + rng.normal(0, noise_px, 2)))
    pt_blocks = [p + rng.normal(0, sigma_perturb, 3) for p in pts]
    pose_blocks = [np.array(T.params, float) for T in poses]
    for b in pose_blocks:
        b[4:] += rng.normal(0, 0.02, 3)
    cam_block = np.array(camera.params, float)

    prob = pyceres.Problem()
    keep, rb = [], []
    for ci, pi, uv in obs:
        c = cf.ReprojErrorCost(camera.model, uv)
        keep.append(c)
        rb.append(
            prob.add_residual_block(
                c, loss, [pt_blocks[pi], pose_blocks[ci], cam_block]
            )
        )
    if fix_rotation:  # exactly COLMAP's SubsetManifold(7,{0,1,2,3}) for const rotation
        for b in pose_blocks:
            prob.set_manifold(b, pyceres.SubsetManifold(7, [0, 1, 2, 3]))
    prob.set_parameter_block_constant(cam_block)
    return dict(
        problem=prob,
        keep=keep,
        blocks_rb=rb,
        points=pt_blocks,
        poses=pose_blocks,
        cam=cam_block,
        camera=camera,
        obs=obs,
    )


def jac(problem, opts=None):
    crs = problem.evaluate_jacobian(opts or pyceres.EvaluateOptions())
    return csr_matrix(
        (crs.values, crs.cols, crs.rows), shape=(crs.num_rows, crs.num_cols)
    )
