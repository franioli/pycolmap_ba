"""Backend: bundle adjustment machinery that pycolmap does not provide.

`constraints` builds the extra residual blocks (marker/GCP observations and
world priors, camera pose priors); `bundle` assembles them on top of COLMAP's
own Ceres problem, solves it and filters the result.
"""

from .bundle import (
    BundleResult,
    FilterOptions,
    SolverOptions,
    build_config,
    build_options,
    estimate_covariance,
    filter_reconstruction,
    run_bundle_adjustment,
    solve_to_convergence,
)
from .constraints import (
    CameraPoseConstraints,
    MarkerConstraints,
    add_camera_pose_constraints,
    add_marker_constraints,
    cov_in_tie_units,
)

__all__ = [
    "BundleResult",
    "CameraPoseConstraints",
    "FilterOptions",
    "MarkerConstraints",
    "SolverOptions",
    "add_camera_pose_constraints",
    "add_marker_constraints",
    "build_config",
    "build_options",
    "cov_in_tie_units",
    "estimate_covariance",
    "filter_reconstruction",
    "run_bundle_adjustment",
    "solve_to_convergence",
]
