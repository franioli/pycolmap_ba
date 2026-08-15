"""Backend: extra residual blocks that pycolmap does not provide.

COLMAP's own bundle adjuster handles tie points. This module adds constraints
to *that same* Ceres problem - marker (GCP / check-point) observations, marker
world coordinates, camera pose priors - as plain residual blocks over the
parameter blocks COLMAP already owns. Nothing here touches `Point3D` or `Track`
structures.

Dataset-agnostic on purpose: inputs are plain dicts, DataFrames and arrays.

Weighting
---------
All sigmas are given in **native units** (pixels, metres, degrees). COLMAP adds
its tie-point reprojection residuals *unweighted*, i.e. with an implicit sigma
of 1 px, so every other sigma is internally divided by `tie_points_sigma_px` to
put the whole problem in one consistent whitened least-squares system. Callers
never do that arithmetic themselves. See `docs/gcp_bundle_adjustment.md` §4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions

from .common import cov_in_tie_units

__all__ = [
    "CameraPoseConstraints",
    "add_camera_pose_constraints",
]


@dataclass
class CameraPoseConstraints:
    """Prior poses for the cameras, typically GNSS or GNSS/INS.

    Positions alone reproduce what COLMAP's native pose priors do; adding
    `sigma_rot_deg` also constrains the rotations, which the native priors
    cannot. Be aware that once the cameras are constrained, the resulting
    camera-pose residual is a *fit* to data the bundle was given, not an
    independent accuracy figure.

    Attributes:
        positions: Prior camera **centres** (not translation vectors) per image
            name, in world coordinates, metres.
        sigma_m: 1-sigma position accuracy, metres. Scalar or (x, y, z).
        cam_from_world: Prior full poses per image name. Required when
            `sigma_rot_deg` is set, ignored otherwise.
        sigma_rot_deg: 1-sigma attitude accuracy, degrees. None keeps the
            constraint position-only.
        robust_loss_scale: Cauchy scale for the priors. None = plain quadratic.
    """

    positions: dict[str, np.ndarray]
    sigma_m: float | Sequence[float] = 0.10
    cam_from_world: dict[str, pycolmap.Rigid3d] | None = None
    sigma_rot_deg: float | None = None
    robust_loss_scale: float | None = None

    def __post_init__(self):
        if self.sigma_rot_deg is not None and self.cam_from_world is None:
            raise ValueError("sigma_rot_deg requires cam_from_world prior poses")


def add_camera_pose_constraints(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    priors: CameraPoseConstraints,
    tie_points_sigma_px: float,
    retained: list[Any],
) -> int:
    """Constrain camera poses to prior positions, optionally with rotations.

    COLMAP ships a native position-prior bundle adjuster
    (`create_pose_prior_ceres_bundle_adjuster` with
    `PosePriorBundleAdjustmentOptions`). This is the flexible alternative: the
    priors live in the same problem as any other constraint added here, and the
    rotations can be constrained too.

    Args:
        problem: The Ceres problem from the single bundle adjuster.
        reconstruction: Provides the pose parameter blocks.
        priors: Prior poses and their sigmas.
        tie_points_sigma_px: Reference sigma for the weighting.
        retained: Keep-alive list for the cost functions.

    Returns:
        Number of residual blocks added (one per registered image with a prior).
    """
    pos_cov = cov_in_tie_units(priors.sigma_m, tie_points_sigma_px, 3)
    if priors.sigma_rot_deg is not None:
        # AbsolutePosePriorCost's residual is [rotation(3, rad), position(3, m)],
        # so the rotation sigma enters in radians. Its position component is on
        # the raw translation t = -R @ C, not on the centre C, so an anisotropic
        # sigma_m given in world axes must be rotated into that frame first:
        # dt = -R @ dC  =>  Cov(t) = R @ Cov(C) @ R^T. Built per-image below,
        # since each prior pose has its own rotation.
        rot_cov = cov_in_tie_units(
            np.deg2rad(priors.sigma_rot_deg), tie_points_sigma_px, 3
        )
    prior_poses = priors.cam_from_world or {}
    robust = (
        pyceres.CauchyLoss(priors.robust_loss_scale)
        if priors.robust_loss_scale is not None
        else None
    )

    n_added = 0
    for image in reconstruction.images.values():
        if priors.sigma_rot_deg is not None:
            prior_pose = prior_poses.get(image.name)
            if prior_pose is None:
                continue
            if not image.is_ref_in_frame:
                raise NotImplementedError(
                    "6-DoF pose priors assume one camera per frame; image "
                    f"{image.name!r} is not the frame reference sensor"
                )
            # Cov(t) = R @ Cov(C) @ R^T for this image's own prior rotation R
            # (see note above pos_cov/rot_cov construction).
            R = prior_pose.rotation.matrix()
            pose_cov = np.block(
                [
                    [rot_cov, np.zeros((3, 3))],
                    [np.zeros((3, 3)), R @ pos_cov @ R.T],
                ]
            )
            cost = cost_functions.AbsolutePosePriorCost(pose_cov, prior_pose)
        else:
            if image.name not in priors.positions:
                continue
            # AbsolutePosePositionPriorCost, NOT NormalPrior. COLMAP's pose
            # parameter block is rig_from_world = (quaternion, t), and the camera
            # centre is C = -R^T t. This cost's residual evaluates to
            # C_prior - C_est directly in world axes, so the sigma really is
            # metres on the CENTRE, in world axes. A NormalPrior on the same
            # block would constrain t instead, which is a different quantity
            # and silently wrong.
            cost = cost_functions.AbsolutePosePositionPriorCost(
                pos_cov, np.asarray(priors.positions[image.name], dtype=float)
            )
        retained.append(cost)
        problem.add_residual_block(
            cost,
            robust,
            [reconstruction.frames[image.frame_id].rig_from_world.params],
        )
        n_added += 1
    return n_added
