"""Helpers: alignment geometry, perturbation, printed reports and export."""

from .geometry import (
    align_to_positions,
    camera_center_rmse,
    rotation_error_deg,
    umeyama,
)
from .io import export_reconstruction, save_point_cloud_as_ply, save_points_as_ply
from .perturb import Perturbation, PerturbationReport, perturb_reconstruction
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
    "GroundTruth",
    "Perturbation",
    "PerturbationReport",
    "accuracy_report",
    "align_to_positions",
    "banner",
    "block_report",
    "camera_center_rmse",
    "compare_to_reconstruction",
    "export_reconstruction",
    "nmad",
    "perturb_reconstruction",
    "rotation_error_deg",
    "save_point_cloud_as_ply",
    "save_points_as_ply",
    "stats",
    "umeyama",
]
