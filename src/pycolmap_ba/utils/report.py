"""Helpers: printed quality and accuracy reports.

Two distinct questions, deliberately kept apart:

* `block_report` - **internal** quality. The numbers available on a real project:
  they say whether the block is well conditioned and self-consistent, and
  nothing about absolute accuracy.
* `accuracy_report` - **external** accuracy against a `GroundTruth`. Per-axis
  bias and 3D RMSE, because planimetry and height behave differently in an
  aerial block and a systematic bias means something quite different from a
  large spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pycolmap

from .geometry import rotation_error_deg

__all__ = [
    "GroundTruth",
    "accuracy_report",
    "banner",
    "block_report",
    "compare_to_reconstruction",
    "nmad",
    "stats",
]


@dataclass
class GroundTruth:
    """Reference values to score a reconstruction against.

    Only `camera_xyz` and `marker_xyz` are required; every other field enables
    one more row of `accuracy_report`, so a partial truth is fine.

    Attributes:
        camera_xyz: True camera centres per image name, metres.
        marker_xyz: True marker coordinates per marker name, metres.
        control_names: Markers that were used as control. Their residual is a
            fit, not an accuracy.
        check_names: Markers held out of the adjustment. Their residual is the
            only unbiased accuracy estimate.
        cam_from_world: True poses per image name, for the rotation check.
        focal_px: Reference focal length in pixels, for the intrinsics check.
    """

    camera_xyz: dict[str, np.ndarray]
    marker_xyz: dict[str, np.ndarray]
    control_names: Sequence[str] | None = None
    check_names: Sequence[str] | None = None
    cam_from_world: dict[str, pycolmap.Rigid3d] | None = None
    focal_px: float | None = None


def banner(text: str) -> None:
    """Print a section header.

    Args:
        text: The header text.
    """
    print(f"\n{'=' * 76}\n{text}\n{'=' * 76}")


def stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Summarise a 1-D sample.

    Args:
        values: The sample.

    Returns:
        Its mean, median, RMS, p90, p99 and max.
    """
    v = np.asarray(values, dtype=float)
    return {
        "mean": v.mean(),
        "median": np.median(v),
        "rms": np.sqrt((v**2).mean()),
        "p90": np.percentile(v, 90),
        "p99": np.percentile(v, 99),
        "max": v.max(),
    }


def nmad(values: Sequence[float] | np.ndarray) -> float:
    """Compute the normalised median absolute deviation - a robust sigma.

    Args:
        values: The sample.

    Returns:
        1.4826 * median(|x - median(x)|), which equals the standard deviation
        for Gaussian data but ignores outliers.
    """
    v = np.asarray(values, dtype=float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def block_report(
    reconstruction: pycolmap.Reconstruction,
    marker_observations: pd.DataFrame | None = None,
    marker_xyz: dict[str, np.ndarray] | None = None,
) -> None:
    """Print the internal quality of a block.

    Args:
        reconstruction: The block to describe.
        marker_observations: Optional marker marks (`image_name`, `marker_name`,
            `u`, `v`), scored separately from the tie points because they are
            weighted differently and are how the control acts on the block.
        marker_xyz: Marker coordinates matching those observations.
    """
    banner("BLOCK QUALITY")

    camera = next(iter(reconstruction.cameras.values()))
    n_img = reconstruction.num_reg_images()
    print(
        f"  images            {n_img} registered / {len(reconstruction.images)} in "
        f"block | {len(reconstruction.cameras)} camera(s) {camera.model.name}"
    )
    print(f"  intrinsics        {np.round(camera.params, 3)}")

    points = list(reconstruction.points3D.values())
    track = np.array([p.track.length() for p in points])
    error = np.array([p.error for p in points])
    n_obs = int(track.sum())
    print(f"  points3D          {len(points):,}      observations {n_obs:,}")

    # The histogram matters more than the mean: 2-ray points are the weak ones.
    bins = [(2, 2), (3, 5), (6, 9), (10, 10**6)]
    hist = "   ".join(
        f"{lo}{'' if lo == hi else ('+' if hi > 1000 else f'-{hi}')}: "
        f"{((track >= lo) & (track <= hi)).sum():,} "
        f"({100 * ((track >= lo) & (track <= hi)).mean():.1f}%)"
        for lo, hi in bins
    )
    print(
        f"  track length      mean {track.mean():.2f}   median {int(np.median(track))}"
    )
    print(f"                    {hist}")

    per_image = np.array([im.num_points3D for im in reconstruction.images.values()])
    print(
        f"  observations/img  min {per_image.min():,}   "
        f"mean {per_image.mean():,.0f}   max {per_image.max():,}"
    )

    s = stats(error)
    print("\n  tie-point reprojection error (px)")
    print(
        f"    mean {s['mean']:.3f}   median {s['median']:.3f}   RMS {s['rms']:.3f}   "
        f"p90 {s['p90']:.3f}   p99 {s['p99']:.3f}   max {s['max']:.2f}"
    )
    print(f"    > 2 px: {(error > 2).sum():,} points ({100 * (error > 2).mean():.1f}%)")

    # Standard photogrammetric redundancy: each observation gives 2 equations,
    # each image 6 pose unknowns, each point 3, plus the intrinsics.
    n_unknowns = 6 * n_img + 3 * len(points) + len(camera.params)
    print(
        f"\n  redundancy        {2 * n_obs:,} measurements - {n_unknowns:,} "
        f"unknowns  ->  r = {2 * n_obs - n_unknowns:,}"
    )

    if marker_observations is None or marker_xyz is None:
        return
    residuals, per_marker = _marker_residuals(
        reconstruction, marker_observations, marker_xyz
    )
    if not residuals:
        return
    s = stats(residuals)
    print(
        f"\n  marker observations ({len(residuals)} in "
        f"{marker_observations.image_name.nunique()} images)"
    )
    print(
        f"    reprojection  mean {s['mean']:.2f} px   "
        f"median {s['median']:.2f}   max {s['max']:.2f}"
    )
    worst = sorted(per_marker.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("    worst:  " + ",  ".join(f"{n} {np.mean(v):.2f} px" for n, v in worst))


def _marker_residuals(
    reconstruction: pycolmap.Reconstruction,
    marker_observations: pd.DataFrame,
    marker_xyz: dict[str, np.ndarray],
) -> tuple[list[float], dict[str, list[float]]]:
    """Reproject the marker coordinates and measure the image residuals.

    Args:
        reconstruction: Provides the poses and intrinsics.
        marker_observations: Columns `image_name`, `marker_name`, `u`, `v`.
        marker_xyz: Marker coordinates, metres.

    Returns:
        `(residuals, per_marker)` in pixels; observations whose image or marker
        is missing, or that project behind the camera, are skipped.
    """
    by_name = {im.name: im for im in reconstruction.images.values()}
    residuals: list[float] = []
    per_marker: dict[str, list[float]] = {}
    for obs in marker_observations.itertuples():
        image = by_name.get(obs.image_name)
        if image is None or obs.marker_name not in marker_xyz:
            continue
        camera = reconstruction.cameras[image.camera_id]
        uv = camera.img_from_cam(image.cam_from_world() * marker_xyz[obs.marker_name])
        if uv is None:
            continue
        distance = float(np.hypot(uv[0] - obs.u, uv[1] - obs.v))
        residuals.append(distance)
        per_marker.setdefault(obs.marker_name, []).append(distance)
    return residuals, per_marker


def accuracy_report(
    reconstruction: pycolmap.Reconstruction,
    marker_xyz: dict[str, np.ndarray],
    truth: GroundTruth,
    label: str = "",
    camera_priors_used: bool = False,
) -> None:
    """Print the external accuracy of a block against ground truth.

    Args:
        reconstruction: The adjusted block.
        marker_xyz: Adjusted marker coordinates, metres. For a controlled bundle
            these are `BundleResult.marker_xyz`, not the surveyed input.
        truth: The reference values.
        label: Suffix for the header, e.g. "after GCP bundle".
        camera_priors_used: Whether the bundle was given camera pose priors. If
            so the camera-centre row is a *fit* to those priors and is flagged as
            such, so it is never read as an accuracy figure.
    """
    banner(f"ACCURACY vs GROUND TRUTH{f' - {label}' if label else ''}")
    print(f"  {'':<22s}{'bias E':>8s}{'bias N':>9s}{'bias U':>9s}{'RMSE 3D':>11s}")
    print(f"  {'':<22s}{'(cm)':>8s}{'(cm)':>9s}{'(cm)':>9s}{'(cm)':>11s}")

    def axis_row(name: str, errors: list, note: str = "") -> None:
        e = np.asarray(errors, dtype=float).reshape(-1, 3)
        bias = e.mean(axis=0) * 100
        rmse3d = np.sqrt((e**2).sum(axis=1).mean()) * 100
        print(
            f"  {name:<22s}{bias[0]:+8.2f}{bias[1]:+9.2f}{bias[2]:+9.2f}"
            f"{rmse3d:11.2f}   {note}"
        )

    images = [
        im for im in reconstruction.images.values() if im.name in truth.camera_xyz
    ]
    axis_row(
        f"camera centres n={len(images)}",
        [im.projection_center() - truth.camera_xyz[im.name] for im in images],
        "(fit - priors used)" if camera_priors_used else "",
    )

    if truth.control_names is None and truth.check_names is None:
        print("\n  no control/check split - all markers scored together")
        present = [n for n in marker_xyz if n in truth.marker_xyz]
        if present:
            axis_row(
                f"markers n={len(present)}",
                [marker_xyz[n] - truth.marker_xyz[n] for n in present],
            )
    else:
        for name, names, note in [
            ("control (GCP)", truth.control_names, "(fit - always flattering)"),
            ("check (CP)", truth.check_names, "<- TRUE ACCURACY"),
        ]:
            present = [n for n in names or [] if n in marker_xyz]
            if present:
                axis_row(
                    f"{name} n={len(present)}",
                    [marker_xyz[n] - truth.marker_xyz[n] for n in present],
                    note,
                )

    if truth.cam_from_world is not None:
        angles = [
            rotation_error_deg(
                im.cam_from_world().rotation.matrix(),
                truth.cam_from_world[im.name].rotation.matrix(),
            )
            for im in reconstruction.images.values()
            if im.name in truth.cam_from_world
        ]
        print(
            f"\n  camera rotations       mean {np.mean(angles):.4f} deg   "
            f"max {np.max(angles):.4f} deg"
        )

    if truth.focal_px is not None:
        focal = next(iter(reconstruction.cameras.values())).params[0]
        print(
            f"  focal                  {focal:.3f}  vs reference "
            f"{truth.focal_px:.3f}  ({focal - truth.focal_px:+.3f})"
        )


def compare_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    reference: pycolmap.Reconstruction,
    max_proj_center_error: float = 0.1,
    verbose: bool = True,
) -> Any:
    """Compare against a reference reconstruction, aligning on camera centres.

    The right tool whenever the truth is available as a COLMAP model rather than
    as tables: `pycolmap.compare_reconstructions` estimates the similarity and
    then reports per-image `rotation_error_deg` and `proj_center_error`.

    Args:
        reconstruction: The block to score.
        reference: The reference block, same image names.
        max_proj_center_error: Inlier threshold for the alignment, metres.
        verbose: Print the RMSE of both error types.

    Returns:
        The dict returned by `pycolmap.compare_reconstructions`, or None if the
        alignment failed.
    """
    result = pycolmap.compare_reconstructions(
        reconstruction,
        reference,
        alignment_error="proj_center",
        max_proj_center_error=max_proj_center_error,
    )
    if result is None:
        if verbose:
            print("  alignment failed - reconstructions are not comparable")
        return None
    if verbose:
        rotations = [e.rotation_error_deg for e in result["errors"]]
        centres = [e.proj_center_error for e in result["errors"]]
        print(
            f"  {len(rotations)} common images: "
            f"rotation RMSE {np.sqrt(np.mean(np.square(rotations))):.4f} deg, "
            f"centre RMSE {np.sqrt(np.mean(np.square(centres))) * 100:.2f} cm"
        )
    return result
