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

__all__ = ["export_reconstruction", "save_point_cloud_as_ply"]


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
    reconstruction: pycolmap.Reconstruction, output_path: Path
) -> bool:
    """Save the 3D points as a PLY carrying track length and reprojection error.

    Those two attributes are what makes a sparse cloud inspectable - filtering on
    them in a viewer separates the well-determined points from the weak ones -
    and COLMAP's own `export_PLY` does not include them. Needs `open3d`, which is
    imported lazily so the rest of the package works without it.

    Args:
        reconstruction: Source of the points.
        output_path: Destination `.ply` file.

    Returns:
        True if open3d reported a successful write.
    """
    import open3d as o3d

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
    return o3d.t.io.write_point_cloud(str(output_path), cloud)
