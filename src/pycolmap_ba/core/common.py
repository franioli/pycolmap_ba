"""Backend: weighting shared by every constraint module.

COLMAP adds its tie-point reprojection residuals *unweighted*, i.e. with an
implicit sigma of 1 px. Only the ratios between sigmas matter, so multiplying
the whole objective by `sigma_tie^2` leaves COLMAP's own blocks matching exactly
and rescales every other sigma by `1 / sigma_tie`. Both helpers here do that,
differing only in what they are handed.

See `docs/gcp_bundle_adjustment.md` section 4.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["cov_in_tie_units", "covs_in_tie_units"]


def cov_in_tie_units(
    sigma: float | Sequence[float] | np.ndarray, tie_points_sigma_px: float, size: int
) -> np.ndarray:
    """Build a diagonal covariance for `sigma`, in tie-point sigma units.

    For constraints specified as independent per-axis sigmas - marker world
    coordinates, camera positions, image measurements.

    Args:
        sigma: Scalar or per-axis 1-sigma, in native units (px or m).
        tie_points_sigma_px: Reference tie-point sigma, pixels.
        size: Dimension of the residual (2 for image, 3 for world).

    Returns:
        A `size` x `size` diagonal covariance matrix.
    """
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (size,))
    return np.diag((s / tie_points_sigma_px) ** 2)


def covs_in_tie_units(cov: np.ndarray, tie_points_sigma_px: float) -> np.ndarray:
    """Rescale full covariance matrices into tie-point sigma units.

    The matrix counterpart of `cov_in_tie_units`, for constraints whose
    covariance is not diagonal - a reference-cloud GICP covariance, say. Scaling
    a covariance by `1 / sigma_tie^2` is the same operation as dividing its
    sigmas by `sigma_tie`.

    Args:
        cov: `(..., n, n)` covariance matrices in native units.
        tie_points_sigma_px: Reference tie-point sigma, pixels.

    Returns:
        The same array scaled by `1 / tie_points_sigma_px^2`.
    """
    return np.asarray(cov, dtype=float) / tie_points_sigma_px**2
