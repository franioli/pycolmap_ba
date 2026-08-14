"""Helpers: posterior covariances of an adjusted block.

Separate from `bundle` because this is a diagnostic on a solved problem rather
than part of solving it, and it is the one step that is allowed to fail without
the adjustment being wrong (see the note in `estimate_covariance`).
"""

from __future__ import annotations

from typing import Any

import pycolmap

__all__ = ["estimate_covariance"]


def estimate_covariance(
    reconstruction: pycolmap.Reconstruction,
    bundle_adjuster: Any,
    options: Any = None,
) -> Any:
    """Estimate posterior covariances for the adjusted parameters.

    Thin wrapper on `pycolmap.estimate_ba_covariance_from_problem`, which needs
    a problem whose structure suits the Schur complement trick. The marker
    parameter blocks added by `add_marker_constraints` are extra 3-vectors that
    are *not* registered as COLMAP points, so the estimation can legitimately
    fail here - hence the None return rather than an exception.

    Args:
        reconstruction: The adjusted reconstruction.
        bundle_adjuster: The adjuster that was solved.
        options: A `pycolmap.BACovarianceOptions`. None uses the defaults.

    Returns:
        A `pycolmap.BACovariance`, or None if the estimation failed.
    """
    options = options or pycolmap.BACovarianceOptions()
    return pycolmap.estimate_ba_covariance_from_problem(
        options, reconstruction, bundle_adjuster.problem
    )
