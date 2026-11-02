"""All three constraint kinds in one problem."""

from __future__ import annotations

import pycolmap
from synthetic import (
    TOLERANCE,
    camera_centres,
    marker_positions,
    point_errors,
    project_markers,
    reference_cloud,
    rmse,
)

from pycolmap_ba import (
    CameraPoseConstraints,
    MarkerConstraints,
    ReferenceCloudConstraints,
    SolverOptions,
    run_cloud_constrained_bundle_adjustment,
)
from pycolmap_ba.utils.compatibility import CompatibilityOptions
from pycolmap_ba.utils.perturb import Perturbation, perturb_reconstruction


def test_markers_poses_and_cloud_solve_together(
    truth: pycolmap.Reconstruction, solver: SolverOptions
) -> None:
    """Every block lands in the same Ceres problem, and the solve recovers truth."""
    xyz = marker_positions()
    block = pycolmap.Reconstruction(truth)
    perturb_reconstruction(
        block, Perturbation(shift_m=0.03, rot_deg=0.5, scale_ppm=5_000, seed=1)
    )

    result = run_cloud_constrained_bundle_adjustment(
        block,
        reference_cloud(truth),
        solver=solver,
        markers=MarkerConstraints(
            observations=project_markers(truth, xyz),
            xyz=xyz,
            control_names=["M0", "M1", "M2"],
            sigma_m=0.01,
        ),
        camera_poses=CameraPoseConstraints(
            positions=camera_centres(truth), sigma_m=0.05
        ),
        reference_options=ReferenceCloudConstraints(max_dist=0.05),
        max_dist_schedule=[0.1, 0.05],
        on_mismatch="raise",
        compatibility=CompatibilityOptions(max_dist=0.1, cell_m=0.25),
        verbose=False,
    )

    # run_bundle_adjustment asserts the problem holds exactly these blocks.
    blocks = result.n_blocks
    assert blocks["marker_world"] == 3
    assert blocks["marker_image"] == len(project_markers(truth, xyz))
    assert blocks["camera_pose"] == truth.num_images()
    assert blocks["reference_cloud"] == truth.num_points3D()
    assert result.converged
    assert rmse(point_errors(block, truth)) < TOLERANCE
    for name in ("M3", "M4", "M5"):  # check points, never constrained
        assert ((result.marker_xyz[name] - xyz[name]) ** 2).sum() ** 0.5 < TOLERANCE
