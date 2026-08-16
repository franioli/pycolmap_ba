"""Helpers: writing a reconstruction out.

COLMAP's own `Reconstruction.write` / `write_text` / `export_PLY` cover the
standard formats; this module just bundles them into one call and adds a point
cloud carrying the per-point quality attributes, which the plain PLY export
drops.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

__all__ = [
    "export_reconstruction",
    "save_point_cloud_as_ply",
    "save_points_as_ply",
]


def export_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    output_path: Path,
    text: bool = True,
    ply: bool = True,
    verbose: bool = True,
) -> Path:
    """Write a reconstruction in binary, text and PLY form.

    Args:
        reconstruction: The block to write.
        output_path: Folder to write into; created if missing.
        text: Also write the human-readable `.txt` model.
        ply: Also write `<folder name>.ply`.
        verbose: Print what was written.

    Returns:
        The output folder.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)
    if text:
        reconstruction.write_text(output_path)
    if ply:
        reconstruction.export_PLY(output_path / f"{output_path.name}.ply")
    if verbose:
        formats = "bin" + (" + txt" if text else "") + (" + ply" if ply else "")
        print(f"  wrote {formats} to {output_path}")
    return output_path


def save_point_cloud_as_ply(
    reconstruction: pycolmap.Reconstruction,
    output_path: Path,
    point_covariances: dict[int, np.ndarray | None] | None = None,
    **scalars: np.ndarray,
) -> bool:
    """Save the 3D points as a PLY carrying track length and reprojection error.

    Those two attributes are what makes a sparse cloud inspectable - filtering on
    them in a viewer separates the well-determined points from the weak ones -
    and COLMAP's own `export_PLY` does not include them. Needs `open3d`, which is
    imported lazily so the rest of the package works without it.

    Args:
        reconstruction: Source of the points.
        output_path: Destination `.ply` file.
        point_covariances: Per-point 3x3 posterior covariance (world X/Y/Z),
            keyed by `point3D_id` - e.g. `{id: cov.get_point_cov(id) for id in
            reconstruction.points3D}` from a `pycolmap.BACovariance` (see
            `estimate_covariance`). Adds the marginal standard deviations
            `std_x`/`std_y`/`std_z` plus `std_norm`, their Euclidean norm - the
            single-number tie-point "Error" Metashape reports in its reference
            panel. A missing id or a non-3x3 entry (point held constant in the
            adjustment) gets NaN.
        **scalars: Extra per-point fields, each `(N,)` in the iteration order of
            `reconstruction.points3D` - e.g. `surface_distance=...` to colour the
            cloud by its residual against a reference surface in CloudCompare.

    Returns:
        True if open3d reported a successful write.

    Raises:
        ValueError: If a scalar array does not have one entry per point.
    """
    import open3d as o3d

    ids = list(reconstruction.points3D.keys())
    points = list(reconstruction.points3D.values())
    cloud = o3d.t.geometry.PointCloud()
    cloud.point.positions = o3d.core.Tensor(
        np.array([p.xyz for p in points], np.float32)
    )
    cloud.point.colors = o3d.core.Tensor(np.array([p.color for p in points], np.uint8))
    cloud.point.track_length = o3d.core.Tensor(
        np.array([[p.track.length()] for p in points], np.int32)
    )
    cloud.point.reproj_error = o3d.core.Tensor(
        np.array([[p.error] for p in points], np.float32)
    )
    if point_covariances is not None:
        std = np.full((len(points), 3), np.nan, np.float32)
        for i, point_id in enumerate(ids):
            cov_pt = point_covariances.get(point_id)
            if cov_pt is not None and np.shape(cov_pt) == (3, 3):
                std[i] = np.sqrt(np.clip(np.diag(cov_pt), 0, None))
        cloud.point.std_x = o3d.core.Tensor(std[:, 0:1])
        cloud.point.std_y = o3d.core.Tensor(std[:, 1:2])
        cloud.point.std_z = o3d.core.Tensor(std[:, 2:3])
        cloud.point.std_norm = o3d.core.Tensor(
            np.linalg.norm(std, axis=1, keepdims=True).astype(np.float32)
        )
    for name, values in scalars.items():
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        if len(values) != len(points):
            raise ValueError(
                f"{name!r} has {len(values)} values for {len(points)} points"
            )
        setattr(cloud.point, name, o3d.core.Tensor(values))
    return o3d.t.io.write_point_cloud(str(output_path), cloud)


def save_points_as_ply(
    points: np.ndarray, output_path: Path, **scalars: np.ndarray
) -> bool:
    """Save a plain `(N, 3)` array as a PLY, with optional per-point fields.

    The counterpart of `save_point_cloud_as_ply` for clouds that are not a
    reconstruction - a reference LiDAR / TLS cloud, say, written out so it can be
    overlaid on the sparse cloud in CloudCompare.

    Args:
        points: `(N, 3)` coordinates.
        output_path: Destination `.ply` file.
        **scalars: Extra per-point fields, each `(N,)`.

    Returns:
        True if open3d reported a successful write.

    Raises:
        ValueError: If `points` is not `(N, 3)` or a scalar length disagrees.
    """
    import open3d as o3d

    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {xyz.shape}")

    cloud = o3d.t.geometry.PointCloud()
    cloud.point.positions = o3d.core.Tensor(xyz)
    for name, values in scalars.items():
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        if len(values) != len(xyz):
            raise ValueError(f"{name!r} has {len(values)} values for {len(xyz)} points")
        setattr(cloud.point, name, o3d.core.Tensor(values))
    return o3d.t.io.write_point_cloud(str(output_path), cloud)
