"""Custom bundle adjustment on top of pycolmap.

Three layers, kept apart on purpose:

* `pycolmap_ba.backend` - what pycolmap does not provide: extra residual blocks
  (marker/GCP observations and world priors, camera pose priors) added to
  COLMAP's own Ceres problem, plus the assembly, solve and filtering around them.
* `pycolmap_ba.frontend` - thin wrappers over native pycolmap calls, only where
  the option plumbing is tedious. Anything not here should be called on
  `pycolmap` directly.
* `pycolmap_ba.utils` - alignment geometry, printed reports, export.

The names below are re-exported for convenience; the submodules remain the
canonical import path.
"""

from . import backend, frontend, utils
from .backend import (
    BundleResult,
    CameraPoseConstraints,
    FilterOptions,
    MarkerConstraints,
    SolverOptions,
    run_bundle_adjustment,
)
from .utils import GroundTruth, accuracy_report, banner, block_report

__all__ = [
    "BundleResult",
    "CameraPoseConstraints",
    "FilterOptions",
    "GroundTruth",
    "MarkerConstraints",
    "SolverOptions",
    "accuracy_report",
    "backend",
    "banner",
    "block_report",
    "frontend",
    "run_bundle_adjustment",
    "utils",
]
