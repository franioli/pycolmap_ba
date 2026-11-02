"""Reference-cloud constraints: surfaces fix the datum only if their normals vary."""

from __future__ import annotations

import numpy as np
import pycolmap
from synthetic import (
    TOLERANCE,
    camera_centres,
    planar_patches,
    point_errors,
    random_normals,
    reference_cloud,
    rmse,
)

from pycolmap_ba import (
    ReferenceCloud,
    ReferenceCloudConstraints,
    SolverOptions,
    run_bundle_adjustment,
    run_cloud_constrained_bundle_adjustment,
)
from pycolmap_ba.core import datum_information
from pycolmap_ba.utils import camera_center_rmse
from pycolmap_ba.utils.compatibility import CompatibilityOptions
from pycolmap_ba.utils.perturb import Perturbation, perturb_reconstruction

# The correspondence search needs the block within reach of the right patch, so
# the cloud tests use a smaller perturbation than the shared `block` fixture:
# 44 mm at the block edge, under the 0.1 m first gate.
NEAR = Perturbation(shift_m=0.03, rot_deg=0.5, scale_ppm=5_000, seed=1)

# The footprint check erodes the survey by one grid cell; its 10 m default
# cell is larger than this whole 2 m block. A few tie-point spacings instead.
COMPATIBILITY = CompatibilityOptions(max_dist=0.1, cell_m=0.25)


def _truth_xyz(truth: pycolmap.Reconstruction) -> np.ndarray:
    return np.array([point.xyz for point in truth.points3D.values()])


def test_cloud_recovers_datum(
    truth: pycolmap.Reconstruction, solver: SolverOptions
) -> None:
    """Surfaces of varied orientation pull a nearby block back onto the truth."""
    cloud = reference_cloud(truth)
    block = pycolmap.Reconstruction(truth)
    report = perturb_reconstruction(block, NEAR)
    assert report.reach > 0.04

    result = run_cloud_constrained_bundle_adjustment(
        block,
        cloud,
        solver=solver,
        reference_options=ReferenceCloudConstraints(max_dist=0.05),
        max_dist_schedule=[0.1, 0.05],
        on_mismatch="raise",
        compatibility=COMPATIBILITY,
        verbose=False,
    )

    assert result.compatibility.passed
    assert result.prealign is None  # the guard did not need to step in
    assert result.correspondences.n_used == truth.num_points3D()
    assert result.passes[-1]["changed_fraction"] < 0.01
    assert camera_center_rmse(block, camera_centres(truth)) < TOLERANCE
    assert rmse(point_errors(block, truth)) < TOLERANCE


def test_flat_cloud_leaves_horizontal_datum_free(
    truth: pycolmap.Reconstruction,
) -> None:
    """Horizontal patches fix height, roll, pitch, scale - not E, N or yaw."""
    xyz = _truth_xyz(truth)
    flat = reference_cloud(truth, normals=np.tile([0.0, 0.0, 1.0], (len(xyz), 1)))
    varied = reference_cloud(truth)

    def modes(cloud: ReferenceCloud) -> tuple[np.ndarray, list[str]]:
        idx, _ = cloud.query(xyz, 0.05)
        info = datum_information(xyz, cloud.covariances(idx))
        labels = [
            info.LABELS[int(np.argmax(np.abs(vector)))]
            for vector in info.eigenvectors.T
        ]
        return info.sigmas, labels

    flat_sigmas, flat_labels = modes(flat)
    varied_sigmas, _ = modes(varied)

    # Eigenvalues ascend, so the largest sigmas (weakest modes) come first.
    weak = flat_sigmas[:3]
    assert set(flat_labels[:3]) == {"tx", "ty", "rz"}
    assert weak.min() > 10 * flat_sigmas[3:].max()
    assert varied_sigmas.max() < weak.min() / 10


def test_rough_patches_are_gated_out(
    truth: pycolmap.Reconstruction, solver: SolverOptions
) -> None:
    """`max_roughness` drops tie points on volumetric clutter, keeps surfaces."""
    xyz = _truth_xyz(truth)
    rough = np.arange(len(xyz)) % 2 == 1
    rng = np.random.default_rng(0)
    # Half the tie points sit on 5 mm "vegetation" blobs, half on flat patches.
    blobs = np.vstack([c + rng.normal(0.0, 0.005, (30, 3)) for c in xyz[rough]])
    discs = planar_patches(xyz[~rough], random_normals(int((~rough).sum())))
    cloud = ReferenceCloud.from_points(np.vstack([discs, blobs]), sigma_ref=0.002)
    max_roughness = 0.001
    # The gate judges the patch of the sample each tie point matched, so count
    # from those rather than from `rough`: where two tie points nearly touch, a
    # disc sample's neighbourhood can take in part of the next blob.
    idx, _ = cloud.query(xyz, 0.05)
    expected = int((cloud.roughness[idx] > max_roughness).sum())
    assert abs(expected - rough.sum()) <= 3

    result = run_bundle_adjustment(
        pycolmap.Reconstruction(truth),
        solver=solver,
        reference_cloud=cloud,
        reference_options=ReferenceCloudConstraints(
            max_dist=0.05, max_roughness=max_roughness
        ),
        verbose=False,
    )

    stats = result.correspondences
    assert stats.rejected["no_match"] == 0
    assert stats.rejected["roughness"] == expected
    assert stats.n_used == len(xyz) - expected
