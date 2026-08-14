"""Helpers: alignment geometry, perturbation, printed reports and export.

Also home to the steps around a bundle adjustment that are not part of
assembling and solving one - `prealign` (global ICP registration, to get the
block in range first), `filtering` (post-solve removal of unsupportable
points) and `covariance` (posterior covariances of the result). They live
here rather than in `core` because none of them touches a Ceres problem;
`core.__init__` re-exports them for convenience since they are still
conceptually part of the bundle-adjustment workflow.
"""

from .covariance import estimate_covariance
from .filtering import FilterOptions, filter_reconstruction
from .geometry import (
    align_to_positions,
    camera_center_rmse,
    rotation_error_deg,
    umeyama,
)
from .io import export_reconstruction, save_point_cloud_as_ply, save_points_as_ply
from .perturb import Perturbation, PerturbationReport, perturb_reconstruction
from .prealign import PrealignResult, prealign_to_reference, residual_rms
from .report import (
    GroundTruth,
    accuracy_report,
    banner,
    block_report,
    compare_to_reconstruction,
    nmad,
    stats,
)

__all__ = [
    "FilterOptions",
    "GroundTruth",
    "Perturbation",
    "PerturbationReport",
    "PrealignResult",
    "accuracy_report",
    "align_to_positions",
    "banner",
    "block_report",
    "camera_center_rmse",
    "compare_to_reconstruction",
    "estimate_covariance",
    "export_reconstruction",
    "filter_reconstruction",
    "nmad",
    "perturb_reconstruction",
    "prealign_to_reference",
    "residual_rms",
    "rotation_error_deg",
    "save_point_cloud_as_ply",
    "save_points_as_ply",
    "stats",
    "umeyama",
]
