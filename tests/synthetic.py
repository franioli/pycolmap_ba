"""Synthetic blocks for the tests: exact geometry, known truth.

`pycolmap.synthesize_dataset` builds a consistent reconstruction - 8 cameras
about 5 units out, looking at 300 points spread over the cube [-1, 1]^3, every
point seen by every image, noise-free observations. Any error left after a
bundle is therefore the bundle's own.

The tests knock a copy out of place with a 7-DoF similarity
(`utils.perturb_reconstruction`). Applied to points and cameras together it
leaves every reprojection residual unchanged, so the tie points cannot undo
it: whatever comes back is due to the constraint under test alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pycolmap

from pycolmap_ba import ReferenceCloud

# Large enough that an unconstrained block stays visibly wrong (0.5 m shift,
# 5 deg, 5 % scale), small enough for a quick, well-conditioned solve.
SHIFT_M, ROT_DEG, SCALE_PPM = 0.5, 5.0, 50_000

# Recovered geometry must match the truth to this, in the dataset's units.
# Noise-free data converges to ~1e-6; the perturbations above are ~1e5 larger.
TOLERANCE = 1e-4


def synthesize(seed: int = 0) -> pycolmap.Reconstruction:
    """The reference block: 8 single-camera frames, 300 points."""
    pycolmap.set_random_seed(seed)
    options = pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_cameras_per_rig=1, num_frames_per_rig=8, num_points3D=300
    )
    return pycolmap.synthesize_dataset(options)


def camera_centres(reconstruction: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    """Camera centres per image name."""
    return {
        image.name: image.projection_center()
        for image in reconstruction.images.values()
    }


def point_errors(
    reconstruction: pycolmap.Reconstruction, truth: pycolmap.Reconstruction
) -> np.ndarray:
    """`(N, 3)` point-coordinate errors against the truth, on common ids."""
    return np.array([
        point.xyz - truth.points3D[point_id].xyz
        for point_id, point in reconstruction.points3D.items()
    ])


def rmse(errors: np.ndarray) -> float:
    """3D RMSE of `(N, 3)` errors."""
    return float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))))


def marker_positions() -> dict[str, np.ndarray]:
    """Six well-spread markers: the cube's face centres, pulled in to 0.8."""
    axes = np.vstack([np.eye(3), -np.eye(3)]) * 0.8
    return {f"M{i}": xyz for i, xyz in enumerate(axes)}


def project_markers(
    truth: pycolmap.Reconstruction, xyz: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Exact image observations of the markers, as `MarkerConstraints` takes them.

    Args:
        truth: The block whose cameras observe the markers.
        xyz: True marker coordinates per name.

    Returns:
        Columns `image_name`, `marker_name`, `u`, `v`, one row per marker that
        projects inside an image.
    """
    rows = []
    for image in truth.images.values():
        camera = truth.cameras[image.camera_id]
        for name, point in xyz.items():
            uv = image.project_point(point)
            if uv is None or not (
                0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height
            ):
                continue
            rows.append((image.name, name, uv[0], uv[1]))
    return pd.DataFrame(rows, columns=["image_name", "marker_name", "u", "v"])


def random_normals(n: int, seed: int = 0) -> np.ndarray:
    """`(n, 3)` uniformly distributed unit vectors."""
    normals = np.random.default_rng(seed).normal(size=(n, 3))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def planar_patches(
    centres: np.ndarray,
    normals: np.ndarray,
    radius: float = 0.01,
    n: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Sample a small flat disc through each centre, orthogonal to its normal.

    A stand-in for a LiDAR scan of the scene: every tie point lies exactly on a
    surface element, and the element's orientation is chosen per point.

    Args:
        centres: `(M, 3)` disc centres.
        normals: `(M, 3)` unit disc normals.
        radius: Disc radius. Keep it below half the closest tie-point spacing,
            or neighbouring discs merge into one tilted patch.
        n: Samples per disc. Above the 20 neighbours `ReferenceCloud` fits.
        seed: Seed for the sample positions.

    Returns:
        `(M * n, 3)` reference points.
    """
    rng = np.random.default_rng(seed)
    discs = []
    for centre, normal in zip(centres, normals, strict=True):
        # Any two unit vectors spanning the disc's plane.
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        r = radius * np.sqrt(rng.random(n))
        theta = 2 * np.pi * rng.random(n)
        discs.append(
            centre + np.outer(r * np.cos(theta), u) + np.outer(r * np.sin(theta), v)
        )
    return np.vstack(discs)


def reference_cloud(
    truth: pycolmap.Reconstruction, normals: np.ndarray | None = None
) -> ReferenceCloud:
    """A reference cloud with one planar patch on every true tie point.

    Args:
        truth: Source of the tie points.
        normals: `(N, 3)` patch normals in `truth.points3D` order. None draws
            random ones, which make every datum mode observable.

    Returns:
        The `ReferenceCloud`, 2 mm accurate.
    """
    xyz = np.array([point.xyz for point in truth.points3D.values()])
    if normals is None:
        normals = random_normals(len(xyz))
    return ReferenceCloud.from_points(planar_patches(xyz, normals), sigma_ref=0.002)
