"""Core: the bundle-adjustment machinery pycolmap does not provide.

Each module has one job, so a change to how a constraint is weighted, how the
block is registered, or how the result is filtered lands in exactly one place.

Constraints - extra residual blocks on COLMAP's own Ceres problem:

* `gcp`          - marker image observations and surveyed world coordinates
* `camera_pose`  - GNSS / INS pose priors
* `cloud`        - LiDAR / TLS reference-surface constraints on the tie points

Solving:

* `bundle`       - assemble and solve **one** adjustment
* `cloud_bundle` - alternate solving with reference-cloud re-correspondence

`covariance` covers both ends of the weighting convention every constraint
above shares: `cov_in_tie_units`/`covs_in_tie_units` whiten a constraint going
in, `variance_factor` and the `*_covariances` functions correctly scale what
comes out (`pycolmap.BACovariance` posterior). It lives here too, rather than
in `utils`, because every function in it takes or is a method of
`BundleResult` - as tightly coupled to the solve as anything else on this
list. `utils.prealign` (global ICP registration, to get the block in range
first) and `utils.filtering` (post-solve removal of unsupportable points)
still live in `utils`, since neither touches a Ceres problem or a
`BundleResult`. Not re-exported here: import them from `utils` directly.
"""

from .bundle import (
    BundleResult,
    SolverOptions,
    _solve_to_convergence,
    build_config,
    run_bundle_adjustment,
)
from .camera_pose import CameraPoseConstraints, add_camera_pose_constraints
from .covariance import (
    VarianceFactor,
    camera_center_covariances,
    camera_pose_covariances,
    cov_in_tie_units,
    covs_in_tie_units,
    estimate_covariance,
    marker_covariances,
    point_covariances,
    std_norm,
    variance_factor,
)
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
    "VarianceFactor",
    "add_camera_pose_constraints",
    "add_marker_constraints",
    "add_reference_cloud_constraints",
    "build_config",
    "camera_center_covariances",
    "camera_pose_covariances",
    "cov_in_tie_units",
    "covs_in_tie_units",
    "datum_information",
    "estimate_covariance",
    "marker_covariances",
    "point_covariances",
    "run_bundle_adjustment",
    "run_cloud_constrained_bundle_adjustment",
    "_solve_to_convergence",
    "std_norm",
    "variance_factor",
]
