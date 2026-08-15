"""Core: the bundle-adjustment machinery pycolmap does not provide.

Each module has one job, so a change to how a constraint is weighted, how the
block is registered, or how the result is filtered lands in exactly one place.

Constraints - extra residual blocks on COLMAP's own Ceres problem:

* `gcp`          - marker image observations and surveyed world coordinates
* `camera_pose`  - GNSS / INS pose priors
* `cloud`        - LiDAR / TLS reference-surface constraints on the tie points
* `common`       - the weighting convention every one of them shares

Solving:

* `bundle`       - assemble and solve **one** adjustment
* `cloud_bundle` - alternate solving with reference-cloud re-correspondence

The steps around a solve - `utils.prealign` (global ICP registration, to get
the block in range first), `utils.filtering` (post-solve removal of
unsupportable points) and `utils.covariance` (posterior covariances of the
result) - live in `utils` instead, since none of them touches a Ceres
problem. Not re-exported here: import them from `utils` directly.
"""

from .bundle import (
    BundleResult,
    SolverOptions,
    build_config,
    build_options,
    run_bundle_adjustment,
    solve_to_convergence,
)
from .camera_pose import CameraPoseConstraints, add_camera_pose_constraints
from .common import cov_in_tie_units, covs_in_tie_units
from .gcp import MarkerConstraints, add_marker_constraints
from .ref_cloud import (
    CorrespondenceStats,
    DatumInformation,
    ReferenceCloud,
    ReferenceCloudConstraints,
    add_reference_cloud_constraints,
    datum_information,
)
from .ref_cloud_iter_bundle import (
    CloudAlignmentError,
    run_cloud_constrained_bundle_adjustment,
)

__all__ = [
    "BundleResult",
    "CameraPoseConstraints",
    "CloudAlignmentError",
    "CorrespondenceStats",
    "DatumInformation",
    "MarkerConstraints",
    "ReferenceCloud",
    "ReferenceCloudConstraints",
    "SolverOptions",
    "add_camera_pose_constraints",
    "add_marker_constraints",
    "add_reference_cloud_constraints",
    "build_config",
    "build_options",
    "cov_in_tie_units",
    "covs_in_tie_units",
    "datum_information",
    "run_bundle_adjustment",
    "run_cloud_constrained_bundle_adjustment",
    "solve_to_convergence",
]
