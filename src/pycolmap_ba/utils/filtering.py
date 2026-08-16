"""Helpers: post-solve removal of points the adjustment cannot support.

Separate from `bundle` because this is cleanup, not estimation: it runs after a
solve, changes no parameter, and is equally useful after a triangulation or an
import. Everything goes through COLMAP's `ObservationManager` so the internal
correspondence bookkeeping stays consistent - `Reconstruction.delete_point3D`
does not maintain it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pycolmap

from ..core.covariance import std_norm

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from ..core.bundle import BundleResult

__all__ = [
    "FilterOptions",
    "filter_points_by_covariance",
    "filter_reconstruction",
    "surviving_blocks",
]


@dataclass
class FilterOptions:
    """Post-solve filtering, applied through COLMAP's `ObservationManager`.

    Unlike `SolverOptions.min_track_length`, these actually delete points, and
    they go through the observation manager so COLMAP's internal correspondence
    bookkeeping stays consistent (raw `Reconstruction.delete_point3D` does not
    maintain it).

    Attributes:
        max_reproj_error: Drop points above this reprojection error, pixels.
        min_tri_angle_deg: Drop points whose best pairwise triangulation angle
            is below this, **degrees** (COLMAP converts internally). On a block
            with good overlap this is typically a no-op.
        min_track_length: Drop points with fewer observations. 0 disables.
    """

    max_reproj_error: float = 4.0
    min_tri_angle_deg: float = 1.5
    min_track_length: int = 0


def filter_reconstruction(
    observation_manager: pycolmap.ObservationManager | None = None,
    reconstruction: pycolmap.Reconstruction | None = None,
    filters: FilterOptions | None = None,
    verbose: bool = True,
) -> int:
    """Delete weak points through COLMAP's `ObservationManager`.

    Uses the observation manager rather than `Reconstruction.delete_point3D` so
    COLMAP's correspondence bookkeeping stays consistent.

    Args:
        observation_manager: If provided, used instead of creating a new one.
        reconstruction: Modified in place. If not provided, uses the one from
            `observation_manager`.
        filters: Thresholds to apply. If not provided, uses the defaults in
            `FilterOptions`.
        verbose: Print how many observations were removed.

    Returns:
        Number of filtered observations.
    """
    if observation_manager is None:
        if reconstruction is None:
            raise ValueError(
                "Either `observation_manager` or `reconstruction` must be provided."
            )
        elif reconstruction is not None:
            observation_manager = pycolmap.ObservationManager(reconstruction)

    if filters is None:
        filters = FilterOptions()

    # Covers reprojection error, negative depth and triangulation angle.
    n_filtered = observation_manager.filter_all_points3D(
        filters.max_reproj_error, filters.min_tri_angle_deg if filters else 1.5
    )
    if filters.min_track_length > 0:
        n_filtered += observation_manager.filter_points3D_with_short_tracks(
            filters.min_track_length
        )

    if verbose:
        print(
            f"  filtered {n_filtered} observations -> "
            f"{reconstruction.num_points3D() if reconstruction else None:,}points remain"
        )
    return n_filtered


def surviving_blocks(
    reconstruction: pycolmap.Reconstruction,
    point3D_ids: Sequence[int] | np.ndarray,
    blocks: Sequence[Any],
) -> list[Any]:
    """Residual blocks whose point still exists, in the same order as `blocks`.

    Deleting a `Point3D` after a solve leaves any residual block that
    referenced it holding a pointer into freed memory - COLMAP owns that
    memory, and `Reconstruction`/`ObservationManager` deletion does not know
    about blocks this package added on top. Call this right after any
    point-deleting step on `BundleResult.residual_blocks["reference_cloud"]`
    (or any other per-point group) to drop the ones that no longer apply.

    Args:
        reconstruction: The block *after* deletion.
        point3D_ids: One id per entry of `blocks`, same order - e.g.
            `BundleResult.correspondences.point3D_ids`.
        blocks: The residual-block ids to filter.

    Returns:
        The surviving blocks, in their original order.
    """
    # .tolist(): pycolmap's points3D map's `__contains__` requires a plain
    # Python int - a bare numpy.int64 (what an array of ids often holds)
    # silently compares unequal to every key and this would drop everything.
    ids = point3D_ids.tolist() if hasattr(point3D_ids, "tolist") else list(point3D_ids)
    present = reconstruction.points3D
    return [block for block, pid in zip(blocks, ids, strict=True) if pid in present]


def filter_points_by_covariance(
    result: BundleResult,
    *,
    max_std_norm: float = 0.10,
    scale_by_sigma0: bool = True,
    sigma0: float | None = None,
    verbose: bool = True,
) -> int:
    """Delete points whose posterior `std_norm` exceeds a threshold, metres.

    `std_norm = sqrt(trace(Sigma))` (see `core.covariance.std_norm`), scaled by
    `sigma0^2` by default so the threshold is in real metres rather than the
    raw cofactor's arbitrary units - see `core.covariance.variance_factor`.
    Reuses `result.point_covariances` rather than recomputing anything, and
    goes through `pycolmap.ObservationManager.delete_point3D` per point, like
    `filter_reconstruction`, so the correspondence bookkeeping stays
    consistent.

    Points with no covariance - excluded from the problem by
    `SolverOptions.min_track_length`, or otherwise held constant - are kept,
    not deleted: no information is not evidence of a bad point, and silently
    deleting every such point would be a much bigger, different filter than
    the caller asked for.

    Args:
        result: The solved bundle - needs `result.BACovariance` from
            `run_bundle_adjustment(..., covariance=True)`.
        max_std_norm: Deletion threshold on `std_norm`, metres.
        scale_by_sigma0: Rescale the raw cofactor matrix by `sigma0^2` first -
            almost always what you want; False compares the threshold against
            the raw, unscaled cofactor trace instead.
        sigma0: Override the internally computed sigma0.
        verbose: Print how many points were removed / kept without a
            covariance.

    Returns:
        Number of points deleted.
    """
    if result.BACovariance is None:
        if verbose:
            print("  no covariance on this result - nothing to filter")
        return 0

    covs = result.point_covariances(scale_by_sigma0=scale_by_sigma0, sigma0=sigma0)
    manager = pycolmap.ObservationManager(result.reconstruction)
    n_deleted = 0
    n_no_cov = 0
    for point3D_id, cov in covs.items():
        if cov is None:
            n_no_cov += 1
            continue
        if std_norm(cov) > max_std_norm:
            manager.delete_point3D(point3D_id)
            n_deleted += 1

    if n_deleted and result.correspondences is not None:
        result.residual_blocks["reference_cloud"] = surviving_blocks(
            result.reconstruction,
            result.correspondences.point3D_ids,
            result.residual_blocks["reference_cloud"],
        )

    if verbose:
        print(
            f"  filtered {n_deleted} points by covariance (std_norm > "
            f"{max_std_norm:.3f} m) -> {result.reconstruction.num_points3D():,} "
            f"points remain ({n_no_cov} kept without covariance)"
        )
    return n_deleted
