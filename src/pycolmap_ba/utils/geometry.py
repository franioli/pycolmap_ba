"""Helpers: similarity alignment and pose error metrics.

A bundle adjustment is a local optimiser, so a free network must be brought
close to the world frame before any world constraint can act on it - that is
what the alignment here is for. The error functions are the ones needed to
decide whether the alignment succeeded.
"""

from __future__ import annotations

import numpy as np
import pycolmap

__all__ = [
    "align_to_positions",
    "camera_center_rmse",
    "rotation_error_deg",
    "umeyama",
]


def umeyama(source: np.ndarray, target: np.ndarray) -> pycolmap.Sim3d:
    """Fit a least-squares similarity mapping `source` -> `target`.

    Prefer this over `pycolmap.align_reconstruction_to_locations` on aerial
    blocks: that one RANSACs over 3-point samples, which is badly conditioned
    when the camera centres are near-coplanar (constant flying height) and can
    return transforms tilted tens of degrees off. A plain least-squares fit over
    every correspondence is deterministic and well behaved.

    Args:
        source: (N, 3) points to transform.
        target: (N, 3) corresponding points.

    Returns:
        The fitted `pycolmap.Sim3d`.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    mean_s, mean_t = source.mean(0), target.mean(0)
    x, y = source - mean_s, target - mean_t
    u, s, vt = np.linalg.svd(x.T @ y / len(source))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = (u @ d @ vt).T
    scale = (s * np.diag(d)).sum() / (x**2).sum() * len(source)
    return pycolmap.Sim3d(
        scale, pycolmap.Rotation3d(rotation), mean_t - scale * rotation @ mean_s
    )


def align_to_positions(
    reconstruction: pycolmap.Reconstruction,
    xyz: dict[str, np.ndarray],
    verbose: bool = True,
) -> pycolmap.Sim3d:
    """Transform a reconstruction onto known camera centres with a Sim3.

    A starting point only - a similarity absorbs uniform scale but *not*
    anisotropic deformation, so a large residual afterwards means the free
    network is distorted (usually a runaway self-calibration) and no amount of
    control will rescue it. Check with `camera_center_rmse`.

    When the target is itself a reconstruction, use COLMAP's
    `align_reconstructions_via_proj_centers` / `..._via_points` instead.

    Args:
        reconstruction: Transformed in place.
        xyz: Reference camera centres per image name, metres.
        verbose: Print the fitted scale.

    Returns:
        The applied `pycolmap.Sim3d`.
    """
    images = [im for im in reconstruction.images.values() if im.name in xyz]
    sim3 = umeyama(
        np.array([im.projection_center() for im in images]),
        np.array([xyz[im.name] for im in images]),
    )
    reconstruction.transform(sim3)
    if verbose:
        print(f"  Sim3 over {len(images)} centres, scale = {sim3.scale:.6f}")
    return sim3


def camera_center_rmse(
    reconstruction: pycolmap.Reconstruction, xyz: dict[str, np.ndarray]
) -> float:
    """Compute the 3D RMSE of the camera centres against `xyz`, in metres.

    Args:
        reconstruction: Source of the estimated centres.
        xyz: Reference centres per image name.

    Returns:
        The RMSE over the images present in both.
    """
    errors = [
        np.linalg.norm(im.projection_center() - xyz[im.name])
        for im in reconstruction.images.values()
        if im.name in xyz
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Compute the angle of the rotation taking `reference` onto `estimated`.

    The single number that summarises an attitude difference: a residual tilt of
    the whole block is invisible in the camera-centre residuals alone.

    Args:
        estimated: (3, 3) estimated rotation matrix.
        reference: (3, 3) reference rotation matrix, same convention.

    Returns:
        The angle in degrees, in [0, 180].
    """
    delta = np.asarray(estimated, float) @ np.asarray(reference, float).T
    return float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
