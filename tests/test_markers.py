"""Marker (GCP) constraints: control fixes the datum, check points stay free."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from synthetic import (
    TOLERANCE,
    camera_centres,
    marker_positions,
    point_errors,
    project_markers,
    rmse,
)

from pycolmap_ba import MarkerConstraints, SolverOptions, run_bundle_adjustment
from pycolmap_ba.utils import camera_center_rmse


def test_control_markers_recover_datum(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """Six control markers pull a moved, rotated, scaled block back onto truth."""
    xyz = marker_positions()
    markers = MarkerConstraints(
        observations=project_markers(truth, xyz), xyz=xyz, sigma_m=0.001
    )
    assert camera_center_rmse(block, camera_centres(truth)) > 0.1

    result = run_bundle_adjustment(block, solver=solver, markers=markers, verbose=False)

    assert result.converged
    assert result.n_blocks["marker_world"] == len(xyz)
    assert result.n_blocks["marker_image"] == len(markers.observations)
    assert camera_center_rmse(block, camera_centres(truth)) < TOLERANCE
    assert rmse(point_errors(block, truth)) < TOLERANCE


def test_check_points_ignore_their_survey(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """A check point is triangulated, never pulled to its (here wrong) survey."""
    true_xyz = marker_positions()
    control = ["M0", "M1", "M2", "M3"]
    check = ["M4", "M5"]
    # The surveyed value only initialises a check point; 0.3 m off must not matter.
    surveyed = {
        name: xyz + (0.3 if name in check else 0.0) for name, xyz in true_xyz.items()
    }
    markers = MarkerConstraints(
        observations=project_markers(truth, true_xyz),
        xyz=surveyed,
        control_names=control,
        sigma_m=0.001,
    )

    result = run_bundle_adjustment(block, solver=solver, markers=markers, verbose=False)

    assert result.n_blocks["marker_world"] == len(control)
    for name in check:
        assert np.linalg.norm(result.marker_xyz[name] - true_xyz[name]) < TOLERANCE


def test_marker_world_residuals_are_whitened_in_tie_units(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """World residual = tie_sigma * (adjusted - surveyed) / sigma, per axis."""
    true_xyz = marker_positions()
    sigma = np.array([0.01, 0.02, 0.04])
    rng = np.random.default_rng(0)
    surveyed = {name: xyz + rng.normal(0.0, sigma) for name, xyz in true_xyz.items()}
    markers = MarkerConstraints(
        observations=project_markers(truth, true_xyz), xyz=surveyed, sigma_m=sigma
    )
    tie_sigma = 0.5

    result = run_bundle_adjustment(
        block,
        solver=solver,
        markers=markers,
        tie_points_sigma_px=tie_sigma,
        verbose=False,
    )

    # World blocks are added in marker order, so rows line up with `surveyed`.
    expected = np.array([
        tie_sigma * (result.marker_xyz[name] - surveyed[name]) / sigma
        for name in surveyed
    ])
    residuals = result.residuals("marker_world")
    assert np.abs(residuals).max() > 0.01  # the noisy survey left a real misfit
    np.testing.assert_allclose(residuals, expected, rtol=1e-6, atol=1e-9)


def test_markers_reject_non_reference_rig_cameras(solver: SolverOptions) -> None:
    """Marker residuals assume one camera per frame and must say so on a rig."""
    pycolmap.set_random_seed(0)
    rig = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=1, num_cameras_per_rig=2, num_frames_per_rig=4, num_points3D=50
        )
    )
    xyz = marker_positions()
    markers = MarkerConstraints(observations=project_markers(rig, xyz), xyz=xyz)

    with pytest.raises(NotImplementedError, match="one camera per frame"):
        run_bundle_adjustment(rig, solver=solver, markers=markers, verbose=False)
