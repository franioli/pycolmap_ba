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
import pandas as pd
import pyceres
import pycolmap
from pycolmap import cost_functions

from .common import cov_in_tie_units

__all__ = [
    "MarkerConstraints",
    "add_marker_constraints",
]


@dataclass
class MarkerConstraints:
    """Marker (GCP / check point) observations and world coordinates.

    A marker contributes two independent kinds of information: its image
    measurements (always used, for every marker) and its surveyed world
    coordinate (used only for markers listed in `control_names`). Markers left
    out of `control_names` become **check points**: they are triangulated by the
    bundle but never constrained, so their error is an unbiased accuracy
    estimate.

    Attributes:
        observations: Columns `image_name`, `marker_name`, `u`, `v`. Image
            coordinates in pixels, in the same convention as the keypoints
            (COLMAP stores them at full resolution).
        xyz: Surveyed / initial world coordinates per marker name, metres. Used
            to initialise every marker and to constrain the control ones.
        control_names: Markers to constrain with their world coordinate. None
            means *all* markers are control (no check points).
        sigma_px: 1-sigma accuracy of the image measurements, pixels. Marking a
            target by hand is typically far less precise than an automatic tie
            point, so this is usually larger than `tie_points_sigma_px`.
        sigma_m: 1-sigma accuracy of the surveyed coordinates, metres. Scalar
            or (x, y, z); height is commonly looser than planimetry. Set it to
            what the survey actually achieved - tighter is not safer, it just
            forces the block onto survey noise.
        robust_loss_scale: Cauchy scale for the marker *image* residuals, to
            absorb mis-identified targets. None disables robustification. The
            world priors are always plain quadratic.
    """

    observations: pd.DataFrame
    xyz: dict[str, np.ndarray]
    control_names: Sequence[str] | None = None
    sigma_px: float = 1.0
    sigma_m: float | Sequence[float] = (0.02, 0.02, 0.03)
    robust_loss_scale: float | None = 2.8


def add_marker_constraints(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    markers: MarkerConstraints,
    tie_points_sigma_px: float,
    retained: list[Any],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Add marker image residuals and world priors to an existing problem.

    Each marker becomes a 3-vector parameter block owned by this module, so the
    reconstruction's Point3D / Track structures are never touched. Image
    residuals tie that block to the poses and intrinsics; world priors tie it to
    the surveyed coordinate.

    Args:
        problem: The Ceres problem from the *single* bundle adjuster.
        reconstruction: Provides images, cameras and pose parameter blocks.
        markers: Observations, coordinates, control list and sigmas.
        tie_points_sigma_px: Reference sigma for the weighting (see module doc).
        retained: List that keeps cost functions alive - pyceres does not take
            ownership, and a garbage-collected cost function leaves the solver
            reading freed memory.

    Returns:
        `(marker_xyz, n_image_blocks, n_world_blocks)` where `marker_xyz` maps
        marker name to the parameter block the solver will optimise.
    """
    image_cov = cov_in_tie_units(markers.sigma_px, tie_points_sigma_px, 2)
    world_cov = cov_in_tie_units(markers.sigma_m, tie_points_sigma_px, 3)

    # Own, mutable copies: these are the parameter blocks Ceres writes into.
    marker_xyz = {name: np.array(xyz, dtype=float) for name, xyz in markers.xyz.items()}
    by_name = {im.name: im for im in reconstruction.images.values()}
    robust = (
        pyceres.CauchyLoss(markers.robust_loss_scale)
        if markers.robust_loss_scale is not None
        else None
    )

    n_image = 0
    for obs in markers.observations.itertuples():
        image = by_name.get(obs.image_name)
        if image is None or obs.marker_name not in marker_xyz:
            continue  # marker not visible in this reconstruction
        if not image.is_ref_in_frame:
            raise NotImplementedError(
                "marker image residuals assume one camera per frame; image "
                f"{image.name!r} is not the frame reference sensor"
            )
        camera = reconstruction.cameras[image.camera_id]
        cost = cost_functions.ReprojErrorCost(
            camera.model, image_cov, np.array([obs.u, obs.v])
        )
        retained.append(cost)
        # Block order is positional: [point3D(3), cam_from_world(7), camera(N)].
        # For a single-camera rig cam_from_world == rig_from_world, and these are
        # live views into COLMAP's memory, so they are the same parameter blocks
        # COLMAP optimises - with the quaternion manifold already set.
        problem.add_residual_block(
            cost,
            robust,
            [
                marker_xyz[obs.marker_name],
                reconstruction.frames[image.frame_id].rig_from_world.params,
                camera.params,
            ],
        )
        n_image += 1

    # Only control markers get a world constraint; the rest stay free and act as
    # check points.
    control = (
        list(markers.control_names)
        if markers.control_names is not None
        else list(marker_xyz)
    )
    n_world = 0
    for name in control:
        if name not in marker_xyz:
            continue
        # NormalPrior(mean, covariance) whitens internally: the residual is
        # chol(cov^-1)^T (x - mean), i.e. the Mahalanobis distance. Use the C++
        # factor, never a Python one: a Python cost function holds the GIL and
        # serialises Ceres's multithreaded pass over the tie-point residuals.
        cost = pyceres.factors.NormalPrior(markers.xyz[name], world_cov)
        retained.append(cost)
        problem.add_residual_block(cost, None, [marker_xyz[name]])
        n_world += 1

    return marker_xyz, n_image, n_world
