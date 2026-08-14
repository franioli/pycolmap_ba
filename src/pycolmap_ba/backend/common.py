from __future__ import annotations

import numpy as np


def cov_in_tie_units(cov: np.ndarray, tie_points_sigma_px: float) -> np.ndarray:
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
