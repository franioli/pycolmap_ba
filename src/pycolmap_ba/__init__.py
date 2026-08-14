"""Custom bundle adjustment on top of pycolmap.

Three layers, kept apart on purpose:

* `pycolmap_ba.core` - what pycolmap does not provide: extra residual blocks
  (marker/GCP observations and world priors, camera pose priors, LiDAR / TLS
  reference-surface constraints) added to COLMAP's own Ceres problem, plus the
  registration, assembly, solve, filtering and covariance around them.
* `pycolmap_ba.frontend` - thin wrappers over native pycolmap calls, only where
  the option plumbing is tedious. Anything not here should be called on
  `pycolmap` directly.
* `pycolmap_ba.utils` - alignment geometry, perturbation, printed reports,
  export.

The names below are re-exported for convenience; the submodules remain the
canonical import path.
"""

from . import core, frontend, utils
from .core import (
    BundleResult,
    CameraPoseConstraints,
    FilterOptions,
    MarkerConstraints,
    ReferenceCloud,
    ReferenceCloudConstraints,
    SolverOptions,
    run_bundle_adjustment,
    run_cloud_constrained_bundle_adjustment,
)
from .utils import GroundTruth, accuracy_report, banner, block_report

__all__ = [
    "BundleResult",
    "CameraPoseConstraints",
    "FilterOptions",
    "GroundTruth",
    "MarkerConstraints",
    "ReferenceCloud",
    "ReferenceCloudConstraints",
    "SolverOptions",
    "accuracy_report",
    "banner",
    "block_report",
    "core",
    "frontend",
    "run_bundle_adjustment",
    "run_cloud_constrained_bundle_adjustment",
    "utils",
]
