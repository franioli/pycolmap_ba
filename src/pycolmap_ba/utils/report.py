"""Helpers: printed quality, process and accuracy reports.

Three distinct questions, deliberately kept apart:

* `block_report` - **internal** quality. The numbers available on a real
  project: they say whether the block is well conditioned and self-consistent,
  and nothing about absolute accuracy. Passing `result` also unlocks a
  **process** section - solver outcome, block composition, robust-loss
  behaviour, correspondence funnel, per-pass convergence - none of which needs
  a ground truth, only the solved `BundleResult`.
* `accuracy_report` - **external** accuracy against a `GroundTruth`. Per-axis
  bias and 3D RMSE, because planimetry and height behave differently in an
  aerial block and a systematic bias means something quite different from a
  large spread. Every "vs. reference" flavour - markers, camera, reference
  reconstruction, reference point cloud - goes through the same `_axis_row`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pycolmap

from .geometry import rotation_error_deg

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from ..core.bundle import BundleResult, SolverOptions
    from ..core.ref_cloud import ReferenceCloud

__all__ = [
    "GroundTruth",
    "accuracy_report",
    "banner",
    "block_report",
    "compare_to_reconstruction",
    "nmad",
    "stats",
]

# Only used for the "% below gate" triangulation-angle statistic - matches
# `FilterOptions.min_tri_angle_deg`'s default, kept in sync by eye since the
# two live in different modules for good reason (this one never deletes
# anything, it only reports).
_MIN_TRI_ANGLE_DEG = 1.5


@dataclass
class GroundTruth:
    """Reference values to score a reconstruction against.

    Every field is optional - each one just enables one more row of
    `accuracy_report`, so a partial truth is fine.

    Attributes:
        camera_xyz: True camera centres per image name, metres.
        marker_xyz: True marker coordinates per marker name, metres.
        control_names: Markers that were used as control. Their residual is a
            fit, not an accuracy.
        check_names: Markers held out of the adjustment. Their residual is the
            only unbiased accuracy estimate.
        cam_from_world: True poses per image name, for the rotation check.
        focal_px: Reference focal length in pixels, for the intrinsics check.
        reconstruction: A full reference reconstruction (e.g. a more precise
            processing of the same block), scored on common image names as an
            alternative/complement to `camera_xyz`.
        cloud: A reference point cloud (LiDAR / TLS), scored as signed
            point-to-plane distance from the sparse tie points.
        cloud_max_dist: Correspondence gate for the `cloud` row, metres.
    """

    camera_xyz: dict[str, np.ndarray] | None = None
    marker_xyz: dict[str, np.ndarray] | None = None
    control_names: Sequence[str] | None = None
    check_names: Sequence[str] | None = None
    cam_from_world: dict[str, pycolmap.Rigid3d] | None = None
    focal_px: float | None = None
    reconstruction: pycolmap.Reconstruction | None = None
    cloud: ReferenceCloud | None = None
    cloud_max_dist: float = 1.0


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
        Its mean, median, RMS, p90, p99, max, count and NMAD.
    """
    v = np.asarray(values, dtype=float)
    return {
        "mean": v.mean(),
        "median": np.median(v),
        "rms": np.sqrt((v**2).mean()),
        "p90": np.percentile(v, 90),
        "p99": np.percentile(v, 99),
        "max": v.max(),
        "n": float(v.size),
        "nmad": nmad(v),
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


def _triangulation_angles_deg(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Best pairwise triangulation angle of every multi-ray point, degrees.

    The one geometric-strength number track length is only a proxy for: two
    rays crossing at a shallow angle triangulate a poorly-determined depth
    however many images see the point. Matches the definition COLMAP's own
    `filter_all_points3D` uses internally (best angle over all ray pairs), but
    computed here since that value is not otherwise exposed.

    Args:
        reconstruction: Source of the points and camera centres.

    Returns:
        One angle per point with at least two valid rays, degrees.
    """
    centers = {
        image_id: image.projection_center()
        for image_id, image in reconstruction.images.items()
    }
    angles: list[float] = []
    for point in reconstruction.points3D.values():
        elements = point.track.elements
        if len(elements) < 2:
            continue
        rays = np.array([
            centers[e.image_id] - point.xyz for e in elements if e.image_id in centers
        ])
        if len(rays) < 2:
            continue
        norm = np.maximum(
            np.linalg.norm(rays, axis=1, keepdims=True), np.finfo(float).tiny
        )
        unit = rays / norm
        cos = np.clip(unit @ unit.T, -1.0, 1.0)
        np.fill_diagonal(cos, 1.0)  # exclude the zero self-angle from the min
        angles.append(float(np.degrees(np.arccos(cos.min()))))
    return np.asarray(angles)


def _free_camera_param_count(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult | None,
    solver: SolverOptions | None,
) -> int:
    """Number of intrinsics parameters actually left free.

    Prefers `result.options` - the `pycolmap.BundleAdjustmentOptions` that was
    actually solved with - over `solver`, since `SolverOptions.extra` can
    override a refine_* flag after `solver` was built. Falling back to neither
    (no result and no solver given) assumes "every parameter free", the same
    assumption the old, buggy redundancy formula made unconditionally - kept
    here only as the least-wrong default when the caller cannot say, and
    printed with a caveat by the caller.

    Args:
        reconstruction: Source of the cameras.
        result: The solved `BundleResult`, if available.
        solver: The settings the bundle was run with, if `result` isn't.

    Returns:
        Total free intrinsics parameters, summed over all cameras.
    """
    result_options = result.options if result is not None else None
    source = result_options if result_options is not None else solver
    if source is None:
        return sum(len(camera.params) for camera in reconstruction.cameras.values())
    if not (
        source.refine_focal_length
        or source.refine_principal_point
        or source.refine_extra_params
    ):
        return 0
    total = 0
    for camera in reconstruction.cameras.values():
        if source.refine_focal_length:
            total += len(list(camera.focal_length_idxs()))
        if source.refine_principal_point:
            total += len(list(camera.principal_point_idxs()))
        if source.refine_extra_params:
            total += len(list(camera.extra_params_idxs()))
    return total


def _redundancy(
    reconstruction: pycolmap.Reconstruction,
    n_obs: int,
    n_points: int,
    result: BundleResult | None,
    solver: SolverOptions | None,
) -> tuple[int, int]:
    """Photogrammetric redundancy `r = n_eq - n_unk + n_datum`.

    Fixes the previous formula, which ignored held-constant intrinsics, the
    marker/camera-pose/reference-cloud constraint equations, and gauge-fixing
    datum DOF.

    Args:
        reconstruction: Source of the images and cameras.
        n_obs: Tie-point observations (`sum(track lengths)`).
        n_points: 3D points in the block.
        result: The solved bundle, if available - supplies the added
            constraint-equation counts and whether the gauge was fixed.
        solver: The solver settings used, if available - decides how many
            intrinsics parameters were actually free.

    Returns:
        `(redundancy, n_datum)`.
    """
    n_img = reconstruction.num_reg_images()
    n_blocks = result.n_blocks if result is not None else {}
    n_eq = 2 * n_obs + 3 * (
        n_blocks.get("marker_world", 0)
        + n_blocks.get("camera_pose", 0)
        + n_blocks.get("reference_cloud", 0)
    )
    n_free_camera = _free_camera_param_count(reconstruction, result, solver)
    n_unk = 6 * n_img + 3 * n_points + n_free_camera
    n_datum = 7 if (result is not None and result.gauge is not None) else 0
    return n_eq - n_unk + n_datum, n_datum


def _sigma0(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult | None,
    redundancy: int,
) -> float:
    """Overall variance factor sigma-hat-0, in tie-point-sigma units.

    `sigma0^2 = sum(whitened residuals^2) / redundancy`; sigma0 near 1 says the
    assumed sigmas were about right, well above 1 says they were too tight (or
    the model does not fit), well below says too loose.

    Tie-point residuals are approximated from `Point3D.error` - COLMAP's own
    per-point mean reprojection error - since their per-observation residual
    block ids are not retained (COLMAP owns those blocks and never hands them
    back): `sum_i r_i^2 ~= track_length * error_point^2`. Every added
    constraint group's residual is exact, via `result.residuals`.

    Args:
        reconstruction: Source of the points.
        result: The solved bundle, if available.
        redundancy: The corrected redundancy (see `_redundancy`).

    Returns:
        `sigma0_hat`, or NaN if the redundancy is not positive.
    """
    if redundancy <= 0:
        return float("nan")
    points = list(reconstruction.points3D.values())
    track = np.array([p.track.length() for p in points], dtype=float)
    error = np.array([p.error for p in points], dtype=float)
    total_sq = float(np.sum(track * error**2))
    if result is not None:
        for group in ("marker_image", "marker_world", "camera_pose", "reference_cloud"):
            r = result.residuals(group)
            if r.size:
                total_sq += float((r**2).sum())
    return float(np.sqrt(total_sq / redundancy))


def _process_section(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult,
    n_obs: int,
    n_points: int,
    redundancy: int,
) -> None:
    """Print solver outcome, block composition, robust-loss and per-pass data.

    Everything here comes off `result` alone - no ground truth needed. See
    `BundleResult.residuals`/`loss_weights` for how the per-group numbers are
    computed; both go through `pyceres.Problem.evaluate_residuals`, never the
    solve path itself.

    Args:
        reconstruction: The adjusted block.
        result: The solved bundle.
        n_obs: Tie-point observations, already computed by the caller.
        n_points: 3D points in the block, already computed by the caller.
        redundancy: The overall redundancy `block_report` already computed,
            apportioned below by observation-equation count.
    """
    outcome = "CONVERGED" if result.converged else "NOT CONVERGED"
    print(f"\n  solver            {outcome}  (usable: {result.usable})")
    if result.summary is not None:
        print(f"                    {result.summary.termination_type}")

    print("  block composition")
    for kind, n in result.n_blocks.items():
        print(f"    {kind:<17}{n:,}")
    if result.n_filtered:
        print(f"    filtered removed {result.n_filtered:,}")

    # Apportioned by observation-equation count: the exact split needs the
    # full redundancy-matrix diagonal, which is not computed here - printed
    # with that caveat, once.
    total_eq = 2 * n_obs
    dims: dict[str, int] = {}
    for group, blocks in result.residual_blocks.items():
        if not blocks:
            continue
        r = result.residuals(group)
        dim = r.shape[1] if r.size else 0
        dims[group] = dim
        total_eq += dim * len(blocks)

    if dims:
        print("  per-group residuals (apportioned redundancy, approximate)")
        for group, dim in dims.items():
            blocks = result.residual_blocks[group]
            r = result.residuals(group)
            msr = float((r**2).sum() / r.size) if r.size else float("nan")
            eq_g = dim * len(blocks)
            r_g = eq_g / total_eq * redundancy if total_eq else float("nan")
            line = (
                f"    {group:<17}n={len(blocks):,}  mean sq whitened resid={msr:.3f}"
                f"  (r_g~={r_g:.0f})"
            )
            weights = result.loss_weights(group)
            if weights.size:
                # kappa_j = ||r_loss||^2 / ||r||^2 = rho'(s_j), the IRLS weight
                # Ceres's Corrector actually applies - confirmed against the
                # analytic Cauchy weight on a toy problem (c=2, s=25): Ceres
                # gives ||r_loss||^2 = 3.448 = rho'(25)*25, so the ratio here
                # is rho'(25) = 0.1379 exactly, matching 1/(1+s/c^2) to
                # floating-point precision. No fallback needed.
                line += (
                    f"  | kappa: median {np.median(weights):.2f}, "
                    f"<0.5: {(weights < 0.5).mean():.1%}, "
                    f"<0.1: {(weights < 0.1).mean():.1%}"
                )
            print(line)

    if result.correspondences is not None:
        print(f"\n  {result.correspondences.summary()}")
        # Datum observability needs the reference cloud's own covariances,
        # which are not on `result` - see `block_report`, which calls
        # `_datum_section` separately when it has `cloud` in scope.

    if result.passes:
        print("\n  per-pass (gate schedule)")
        for record in result.passes:
            print(
                f"    {record['pass']}: gate {record['max_dist']:.2f} m, "
                f"{record['n_used']:,} constrained, "
                f"{record['changed_fraction']:.1%} changed, "
                f"median to plane {record['median_surface_distance']:.3f} m, "
                f"{record['seconds']:.1f} s"
            )

    if result.compatibility is not None:
        print(f"\n  {result.compatibility.summary()}")
    if result.prealign is not None:
        print(f"  {result.prealign.summary()}")

    if result.covariance is not None:
        _posterior_sigma_summary(reconstruction, result.covariance)


def _posterior_sigma_summary(
    reconstruction: pycolmap.Reconstruction, covariance: Any
) -> None:
    """Print a one-line summary of the posterior camera-position sigmas.

    Best-effort: `pycolmap.BACovariance`'s per-image matrix layout is not
    pinned down here, so this only reports the position sub-block trace and
    says nothing about rotations or points. Silently skipped if the API
    disagrees with that assumption.

    Args:
        reconstruction: Source of the registered images.
        covariance: The `pycolmap.BACovariance` from `estimate_covariance`.
    """
    sigmas = []
    for image_id in reconstruction.reg_image_ids():
        try:
            cov = np.asarray(covariance.get_cam_cov_from_world(image_id))
        except Exception:
            continue
        if cov.ndim != 2 or cov.shape[0] < 3:
            continue
        position = cov[-3:, -3:]  # translation block, by convention last
        sigmas.append(float(np.sqrt(np.mean(np.diag(position)))))
    if sigmas:
        arr = np.asarray(sigmas) * 100
        print(
            f"\n  posterior camera-position sigma (n={len(arr)}): "
            f"median {np.median(arr):.2f} cm, max {arr.max():.2f} cm"
        )
    else:
        print("\n  posterior covariance available (per-image sigmas not extracted)")


def block_report(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult | None = None,
    *,
    marker_observations: pd.DataFrame | None = None,
    marker_xyz: dict[str, np.ndarray] | None = None,
    cloud: ReferenceCloud | None = None,
    solver: SolverOptions | None = None,
    tie_points_sigma_px: float = 0.5,
    triangulation_angles: bool = True,
    label: str = "",
) -> dict[str, float]:
    """Print the internal quality - and, with `result`, the process - of a block.

    Args:
        reconstruction: The block to describe.
        result: The solved `BundleResult`, if available. Unlocks the process
            section: solver outcome, block composition, per-group whitened
            residuals and robust-loss weights, correspondence funnel, per-pass
            convergence, compatibility/prealign summaries, sigma0.
        marker_observations: Optional marker marks (`image_name`, `marker_name`,
            `u`, `v`), scored separately from the tie points because they are
            weighted differently and are how the control acts on the block.
        marker_xyz: Marker coordinates matching those observations.
        cloud: The reference cloud, if the bundle used one - enables the datum
            observability line.
        solver: The solver settings used, if available - lets the redundancy
            count only the intrinsics parameters that were actually free.
        tie_points_sigma_px: The sigma the bundle was assembled with. Not
            currently used beyond documenting the units sigma0 is reported in
            (tie-point-sigma units, same as `BundleResult.residuals`).
        triangulation_angles: Print the triangulation-angle section. Default on
            - sub-second on this project's blocks.
        label: Suffix for the header, e.g. "after GCP bundle".

    Returns:
        Headline scalars: `sigma0`, `redundancy`, `reproj_rms_px`,
        `median_tri_angle_deg`, `n_points`, `n_constrained`.
    """
    banner(f"BLOCK QUALITY{f' - {label}' if label else ''}")

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
    n_points = len(points)
    print(f"  points3D          {n_points:,}      observations {n_obs:,}")

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

    reproj_stats = stats(error)
    print("\n  tie-point reprojection error (px)")
    print(
        f"    mean {reproj_stats['mean']:.3f}   median {reproj_stats['median']:.3f}   "
        f"RMS {reproj_stats['rms']:.3f}   p90 {reproj_stats['p90']:.3f}   "
        f"p99 {reproj_stats['p99']:.3f}   max {reproj_stats['max']:.2f}"
    )
    print(f"    > 2 px: {(error > 2).sum():,} points ({100 * (error > 2).mean():.1f}%)")

    median_tri_angle = float("nan")
    if triangulation_angles:
        angles = _triangulation_angles_deg(reconstruction)
        if angles.size:
            median_tri_angle = float(np.median(angles))
            print("\n  triangulation angle (deg, best pairwise ray angle per point)")
            below = angles < _MIN_TRI_ANGLE_DEG
            p10 = np.percentile(angles, 10)
            print(
                f"    median {median_tri_angle:.2f}   p10 {p10:.2f}   "
                f"< {_MIN_TRI_ANGLE_DEG:.1f} deg: {below.sum():,} points "
                f"({100 * below.mean():.1f}%)"
            )

    redundancy, n_datum = _redundancy(reconstruction, n_obs, n_points, result, solver)
    n_free_camera = _free_camera_param_count(reconstruction, result, solver)
    n_eq = 2 * n_obs + 3 * (
        (result.n_blocks.get("marker_world", 0) if result else 0)
        + (result.n_blocks.get("camera_pose", 0) if result else 0)
        + (result.n_blocks.get("reference_cloud", 0) if result else 0)
    )
    n_unk = 6 * n_img + 3 * n_points + n_free_camera
    caveat = (
        ""
        if solver is not None or (result is not None and result.options is not None)
        else "  (no solver given - assumed all intrinsics free)"
    )
    print(
        f"\n  redundancy        {n_eq:,} equations - {n_unk:,} unknowns"
        f"{f' + {n_datum} datum' if n_datum else ''}  ->  r = {redundancy:,}{caveat}"
    )
    sigma0 = _sigma0(reconstruction, result, redundancy)
    print(f"  sigma0_hat        {sigma0:.3f}  (1.0 = assumed sigmas about right)")

    n_constrained = 0
    if result is not None:
        _process_section(reconstruction, result, n_obs, n_points, redundancy)
        if result.correspondences is not None:
            n_constrained = result.correspondences.n_used
            if cloud is not None and n_constrained:
                _datum_section(reconstruction, cloud, result)

    if marker_observations is not None and marker_xyz is not None:
        residuals, per_marker = _marker_residuals(
            reconstruction, marker_observations, marker_xyz
        )
        if residuals:
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
            worst_str = ",  ".join(f"{n} {np.mean(v):.2f} px" for n, v in worst)
            print(f"    worst:  {worst_str}")

    return {
        "sigma0": sigma0,
        "redundancy": float(redundancy),
        "reproj_rms_px": float(reproj_stats["rms"]),
        "median_tri_angle_deg": median_tri_angle,
        "n_points": float(n_points),
        "n_constrained": float(n_constrained),
    }


def _datum_section(
    reconstruction: pycolmap.Reconstruction, cloud: ReferenceCloud, result: BundleResult
) -> None:
    """Print datum observability for the points this solve actually constrained.

    Args:
        reconstruction: The adjusted block.
        cloud: The reference cloud the bundle was constrained to.
        result: The solved bundle - supplies which points were used and their
            matched reference patches.
    """
    # Local import: avoids a circular import at module load time (`core.bundle`
    # imports `utils.covariance`/`utils.filtering`, which triggers this
    # package's `__init__.py`, before `core.ref_cloud` finishes importing).
    from ..core.ref_cloud import datum_information

    stats_c = result.correspondences
    # .tolist(): pycolmap's points3D map's `__contains__` requires a plain
    # Python int - a bare numpy.int64 (what point3D_ids holds) silently
    # compares unequal to every key and this would filter out everything.
    present = set(reconstruction.points3D.keys())
    ids = [pid for pid in stats_c.point3D_ids.tolist() if pid in present]
    if len(ids) != stats_c.n_used:
        return  # some constrained points were filtered away since; skip, don't mislead
    xyz = np.array([reconstruction.points3D[pid].xyz for pid in ids])
    covs = cloud.covariances(stats_c.reference_ids)
    info = datum_information(xyz, covs)
    print(f"\n  {info.summary()}")


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


def _empty_row(name: str, note: str) -> None:
    """Print a placeholder row when a flavour has no data to score."""
    dash = f"{'--':>8s}{'--':>9s}{'--':>9s}{'--':>11s}{0:>6d}{'--':>9s}"
    print(f"  {name:<28s}{dash}   {note}")


def _axis_row(
    name: str, errors: Sequence[np.ndarray] | np.ndarray, note: str = ""
) -> None:
    """Print one row of the accuracy table: bias per axis, RMSE 3D, n, NMAD.

    Shared by every "vs. reference" flavour - markers, camera, reference
    reconstruction, reference cloud - so they read as one consistent table
    rather than separate bolted-on sections.

    Args:
        name: Row label, e.g. "control (GCP) n=18".
        errors: `(n, 3)` array of `(dE, dN, dU)` errors, for a full 3D
            comparison; or `(n,)` (or anything reshaping to it) for a
            reference-cloud row, where only the signed point-to-plane distance
            is meaningful and E/N are printed as `n/a` - the constraint is
            tangentially slack, so a horizontal number there would be false
            precision.
        note: Trailing annotation, e.g. "(fit - always flattering)".
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim == 1 or (e.ndim == 2 and e.shape[-1] != 3):
        u = e.reshape(-1)
        if u.size == 0:
            _empty_row(name, note)
            return
        bias_u = float(np.mean(u)) * 100
        rmse = float(np.sqrt(np.mean(u**2))) * 100
        nmad_u = nmad(u) * 100
        print(
            f"  {name:<28s}{'n/a':>8s}{'n/a':>9s}{bias_u:+9.2f}"
            f"{rmse:11.2f}{u.size:6d}{nmad_u:9.2f}   {note}"
        )
        return

    e = e.reshape(-1, 3)
    if e.shape[0] == 0:
        _empty_row(name, note)
        return
    bias = e.mean(axis=0) * 100
    rmse3d = np.sqrt((e**2).sum(axis=1).mean()) * 100
    nmad3d = nmad(np.sqrt((e**2).sum(axis=1))) * 100
    print(
        f"  {name:<28s}{bias[0]:+8.2f}{bias[1]:+9.2f}{bias[2]:+9.2f}"
        f"{rmse3d:11.2f}{e.shape[0]:6d}{nmad3d:9.2f}   {note}"
    )


def _cloud_rows(
    reconstruction: pycolmap.Reconstruction,
    cloud: ReferenceCloud,
    max_dist: float,
    result: BundleResult | None,
) -> None:
    """Print the reference-cloud accuracy row(s), split by constrained/not.

    Args:
        reconstruction: The adjusted block.
        cloud: The reference cloud to measure against.
        max_dist: Correspondence gate, metres.
        result: The solved bundle, if available - `result.correspondences`
            says which points actually got a residual block, splitting
            constrained (pulled onto the surface) from unconstrained (matched
            but outside the final gate, or not queried at all).
    """
    ids = np.array(list(reconstruction.points3D.keys()), dtype=np.int64)
    xyz = np.array([reconstruction.points3D[i].xyz for i in ids], dtype=float)
    idx, dist = cloud.query(xyz, max_dist)
    matched = idx >= 0
    if not matched.any():
        print(f"  {'reference cloud':<28s}no points within {max_dist:.2f} m")
        return
    safe = np.where(matched, idx, 0)
    signed = np.einsum("mi,mi->m", xyz - cloud.mu[safe], cloud.normals[safe])
    signed = signed[matched]
    matched_ids = ids[matched]

    correspondences = result.correspondences if result is not None else None
    if correspondences is not None and correspondences.n_used:
        constrained = set(correspondences.point3D_ids.tolist())
        is_constrained = np.array([pid in constrained for pid in matched_ids])
        n_constrained = int(is_constrained.sum())
        n_unconstrained = int((~is_constrained).sum())
        _axis_row(f"cloud constrained n={n_constrained}", signed[is_constrained])
        _axis_row(f"cloud unconstrained n={n_unconstrained}", signed[~is_constrained])
    else:
        _axis_row(f"cloud (all matched) n={len(signed)}", signed)


def accuracy_report(
    reconstruction: pycolmap.Reconstruction,
    marker_xyz: dict[str, np.ndarray] | None,
    truth: GroundTruth,
    label: str = "",
    camera_priors_used: bool = False,
    *,
    result: BundleResult | None = None,
) -> None:
    """Print the external accuracy of a block against ground truth.

    One table, every "vs. reference" flavour as a row: control/check markers,
    camera (centres/rotations/focal), reference reconstruction (if
    `truth.reconstruction`), reference cloud (if `truth.cloud`).

    Args:
        reconstruction: The adjusted block.
        marker_xyz: Adjusted marker coordinates, metres. For a controlled bundle
            these are `BundleResult.marker_xyz`, not the surveyed input.
        truth: The reference values.
        label: Suffix for the header, e.g. "after GCP bundle".
        camera_priors_used: Whether the bundle was given camera pose priors. If
            so the camera-centre row is a *fit* to those priors and is flagged as
            such, so it is never read as an accuracy figure.
        result: The solved `BundleResult`, if available - only used to split
            the reference-cloud row into constrained/unconstrained.
    """
    banner(f"ACCURACY vs GROUND TRUTH{f' - {label}' if label else ''}")
    print(
        f"  {'':<28s}{'bias E':>8s}{'bias N':>9s}{'bias U':>9s}{'RMSE 3D':>11s}"
        f"{'n':>6s}{'NMAD':>9s}"
    )
    print(
        f"  {'':<28s}{'(cm)':>8s}{'(cm)':>9s}{'(cm)':>9s}{'(cm)':>11s}"
        f"{'':>6s}{'(cm)':>9s}"
    )

    if truth.camera_xyz:
        images = [
            im for im in reconstruction.images.values() if im.name in truth.camera_xyz
        ]
        _axis_row(
            f"camera centres n={len(images)}",
            [im.projection_center() - truth.camera_xyz[im.name] for im in images],
            "(fit - priors used)" if camera_priors_used else "",
        )

    if truth.marker_xyz and marker_xyz:
        if truth.control_names is None and truth.check_names is None:
            present = [n for n in marker_xyz if n in truth.marker_xyz]
            if present:
                _axis_row(
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
                    _axis_row(
                        f"{name} n={len(present)}",
                        [marker_xyz[n] - truth.marker_xyz[n] for n in present],
                        note,
                    )

    if truth.reconstruction is not None:
        ref_by_name = {i.name: i for i in truth.reconstruction.images.values()}
        pairs = [
            (im, ref_by_name[im.name])
            for im in reconstruction.images.values()
            if im.name in ref_by_name
        ]
        if pairs:
            _axis_row(
                f"reference reconstruction n={len(pairs)}",
                [a.projection_center() - b.projection_center() for a, b in pairs],
            )

    if truth.cloud is not None:
        _cloud_rows(reconstruction, truth.cloud, truth.cloud_max_dist, result)

    if truth.cam_from_world is not None:
        angles = [
            rotation_error_deg(
                im.cam_from_world().rotation.matrix(),
                truth.cam_from_world[im.name].rotation.matrix(),
            )
            for im in reconstruction.images.values()
            if im.name in truth.cam_from_world
        ]
        if angles:
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
