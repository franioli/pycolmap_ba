"""Camera pose priors: positions fix the datum, rotations add what they cannot."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from synthetic import TOLERANCE, camera_centres, point_errors, rmse

from pycolmap_ba import CameraPoseConstraints, SolverOptions, run_bundle_adjustment
from pycolmap_ba.utils import camera_center_rmse, rotation_error_deg


def test_position_priors_recover_datum(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """Camera-centre priors alone pull a moved, rotated, scaled block back."""
    priors = CameraPoseConstraints(positions=camera_centres(truth), sigma_m=0.001)
    assert camera_center_rmse(block, camera_centres(truth)) > 0.1

    result = run_bundle_adjustment(
        block, solver=solver, camera_poses=priors, verbose=False
    )

    assert result.converged
    assert result.n_blocks["camera_pose"] == truth.num_images()
    assert camera_center_rmse(block, camera_centres(truth)) < TOLERANCE
    assert rmse(point_errors(block, truth)) < TOLERANCE


def test_position_priors_constrain_the_camera_centre(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """Priors shifted by d move the whole block by d, in world axes.

    The prior is on the centre C = -R^T t, not on the translation t. A prior on t
    would turn the same world offset into a different, rotation-dependent
    displacement per camera, which no rigid block motion can satisfy.
    """
    offset = np.array([0.3, -0.2, 0.1])
    priors = CameraPoseConstraints(
        positions={name: c + offset for name, c in camera_centres(truth).items()},
        sigma_m=0.001,
    )

    run_bundle_adjustment(block, solver=solver, camera_poses=priors, verbose=False)

    centres, true_centres = camera_centres(block), camera_centres(truth)
    for name, centre in centres.items():
        np.testing.assert_allclose(centre - true_centres[name], offset, atol=TOLERANCE)
    np.testing.assert_allclose(
        point_errors(block, truth).mean(axis=0), offset, atol=TOLERANCE
    )


def test_rotation_priors_fix_a_two_camera_datum(
    block: pycolmap.Reconstruction,
    truth: pycolmap.Reconstruction,
    solver: SolverOptions,
) -> None:
    """Two positions leave a roll about their baseline free; rotations fix it."""
    names = sorted(camera_centres(truth))[:2]
    images = {image.name: image for image in truth.images.values()}
    priors = CameraPoseConstraints(
        positions={name: images[name].projection_center() for name in names},
        cam_from_world={name: images[name].cam_from_world() for name in names},
        sigma_m=0.001,
        sigma_rot_deg=0.001,
    )

    result = run_bundle_adjustment(
        block, solver=solver, camera_poses=priors, verbose=False
    )

    # One 6-vector per prior: rotation (3) and position (3).
    assert result.residuals("camera_pose").shape == (2, 6)
    assert camera_center_rmse(block, camera_centres(truth)) < TOLERANCE
    for image in block.images.values():
        reference = images[image.name].cam_from_world().rotation.matrix()
        estimated = image.cam_from_world().rotation.matrix()
        assert rotation_error_deg(estimated, reference) < 1e-3


def test_rotation_sigma_requires_prior_poses() -> None:
    """A rotation sigma without prior rotations is a configuration error."""
    with pytest.raises(ValueError, match="cam_from_world"):
        CameraPoseConstraints(positions={}, sigma_rot_deg=1.0)
