"""Helpers: post-solve removal of points the adjustment cannot support.

Separate from `bundle` because this is cleanup, not estimation: it runs after a
solve, changes no parameter, and is equally useful after a triangulation or an
import. Everything goes through COLMAP's `ObservationManager` so the internal
correspondence bookkeeping stays consistent - `Reconstruction.delete_point3D`
does not maintain it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pycolmap

__all__ = ["FilterOptions", "filter_reconstruction"]


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
    reconstruction: pycolmap.Reconstruction,
    filters: FilterOptions,
    verbose: bool = True,
) -> int:
    """Delete weak points through COLMAP's `ObservationManager`.

    Uses the observation manager rather than `Reconstruction.delete_point3D` so
    COLMAP's correspondence bookkeeping stays consistent.

    Args:
        reconstruction: Modified in place.
        filters: Thresholds to apply.
        verbose: Print how many observations were removed.

    Returns:
        Number of filtered observations.
    """
    manager = pycolmap.ObservationManager(reconstruction)
    # Covers reprojection error, negative depth and triangulation angle.
    n_filtered = manager.filter_all_points3D(
        filters.max_reproj_error, filters.min_tri_angle_deg
    )
    if filters.min_track_length > 0:
        n_filtered += manager.filter_points3D_with_short_tracks(
            filters.min_track_length
        )
    if verbose:
        print(
            f"  filtered {n_filtered} observations -> "
            f"{reconstruction.num_points3D():,} points remain"
        )
    return n_filtered
