"""Helpers: bring a block within reach of the reference cloud before constraining.

`run_cloud_constrained_bundle_adjustment` is a *local* refiner. Its
correspondence search takes the nearest reference patch, so once the block is
further out than the spacing between distinct surfaces, "nearest" means the
wrong surface and no amount of iterating recovers it - widening the gate only
buys more confidently wrong matches. There is no schedule that fixes a block
sitting tens of metres away.

That is a **global registration** problem, and it belongs here rather than in
the bundle. The stages mirror what one does by hand in CloudCompare:

1. **Coarse** - put the two clouds on top of each other by centroid, the
   robust cousin of CloudCompare's bounding-box matching (a sparse SfM cloud
   always has stray points far outside the block, which move a bounding box a
   long way but barely touch a centroid). Scale starts at 1 here - if the
   block's scale is genuinely unknown, take it from camera priors
   (`utils.align_to_positions`) before calling this, not from cloud extents:
   the reference survey rarely covers exactly the same ground as the block,
   so an extent ratio is a footprint difference, not a scale error.
2. **Fine** - ICP, coarse to fine over decreasing voxel sizes. Point-to-point
   with scaling first, because that is the only open3d estimator that recovers a
   similarity and the block's scale is usually wrong; then point-to-plane, which
   ignores scale but converges far more precisely on planar scenes.

The result is a `Sim3d` applied to the reconstruction, after which the bundle
takes over. Prealignment moves points *and* cameras rigidly, so it changes no
reprojection residual and costs the block nothing - it only chooses which datum
the bundle starts from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pycolmap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.cloud import ReferenceCloud

__all__ = ["PrealignResult", "prealign_to_reference", "residual_rms"]


@dataclass
class PrealignResult:
    """What `prealign_to_reference` did, and how well it worked.

    Attributes:
        sim3d: The applied similarity, or None if the block was already close
            enough and nothing was done.
        rms_before: Robust RMS distance from the sparse points to the reference
            cloud before, metres.
        rms_after: The same after. If this is not much smaller than
            `rms_before`, the registration failed and the block is probably in a
            different place entirely - do not proceed to the bundle.
        fitness: Fraction of sparse points with a correspondence at the final
            ICP threshold. Below ~0.3 the result is not trustworthy.
        inlier_rmse: RMS over those correspondences, metres.
        stages: One record per stage - `name`, `voxel`, `fitness`,
            `inlier_rmse`, `scale`.
    """

    sim3d: pycolmap.Sim3d | None = None
    rms_before: float = 0.0
    rms_after: float = 0.0
    fitness: float = 0.0
    inlier_rmse: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Multi-line recap, one line per stage."""
        if self.sim3d is None:
            return (
                f"prealign: skipped, RMS to reference already {self.rms_before:.3f} m"
            )
        lines = [
            f"prealign: RMS to reference {self.rms_before:.3f} -> "
            f"{self.rms_after:.3f} m | fitness {self.fitness:.2f}, "
            f"inlier RMSE {self.inlier_rmse:.3f} m"
        ]
        for stage in self.stages:
            lines.append(
                f"  {stage['name']:<22} voxel {stage['voxel']:>6.2f} m  "
                f"fitness {stage['fitness']:.2f}  rmse {stage['inlier_rmse']:.3f} m  "
                f"scale {stage['scale']:.6f}"
            )
        return "\n".join(lines)


def _sparse_xyz(
    reconstruction: pycolmap.Reconstruction, min_track_length: int
) -> np.ndarray:
    """Well-determined tie points as an `(N, 3)` array."""
    return np.array(
        [
            p.xyz
            for p in reconstruction.points3D.values()
            if p.track.length() >= min_track_length
        ],
        dtype=np.float64,
    )


def _to_sim3d(matrix: np.ndarray) -> pycolmap.Sim3d:
    """Convert a 4x4 similarity matrix to a `pycolmap.Sim3d`.

    open3d returns `s R` fused in the upper-left block when the estimator was
    built `with_scaling=True`, so the scale has to be pulled back out. The
    rotation is re-orthonormalised via SVD: `Rotation3d` wants a proper rotation
    and the division leaves it only orthonormal to numerical precision.
    """
    linear = np.asarray(matrix, dtype=float)[:3, :3]
    scale = float(np.cbrt(max(np.linalg.det(linear), np.finfo(float).tiny)))
    u, _, vt = np.linalg.svd(linear / scale)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # reflection, not a rotation
        u[:, -1] *= -1
        rotation = u @ vt
    return pycolmap.Sim3d(
        scale, pycolmap.Rotation3d(rotation), np.asarray(matrix, float)[:3, 3]
    )


def residual_rms(
    reconstruction: pycolmap.Reconstruction,
    cloud: ReferenceCloud,
    *,
    min_track_length: int = 3,
    percentile: float = 90.0,
) -> float:
    """Robust RMS distance from the sparse points to the reference cloud, metres.

    The trigger for prealignment, and the before/after measure of whether it
    worked. Trimmed at `percentile` because a sparse cloud always contains
    points with no counterpart at all - sky, vegetation, reconstruction blunders
    - and those would otherwise dominate the statistic and hide a good fit.

    Args:
        reconstruction: Source of the tie points.
        cloud: The reference cloud.
        min_track_length: Ignore points seen by fewer images.
        percentile: Trim above this percentile of the distances.

    Returns:
        The trimmed RMS, or `inf` if there are no points to measure.
    """
    import open3d as o3d

    xyz = _sparse_xyz(reconstruction, min_track_length)
    if len(xyz) == 0:
        return float("inf")
    index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(cloud.points))
    index.knn_index()
    _, squared = index.knn_search(o3d.core.Tensor(np.ascontiguousarray(xyz)), 1)
    distances = np.sqrt(np.abs(squared.numpy().reshape(-1)))
    kept = distances[distances <= np.percentile(distances, percentile)]
    return float(np.sqrt(np.mean(kept**2)))


def prealign_to_reference(
    reconstruction: pycolmap.Reconstruction,
    cloud: ReferenceCloud,
    *,
    trigger_rms: float = 0.5,
    voxel_sizes: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
    with_scale: bool = True,
    min_track_length: int = 3,
    verbose: bool = True,
) -> PrealignResult:
    """Register the block onto the reference cloud before the bundle sees it.

    Applies the resulting similarity to `reconstruction` in place. Does nothing
    if the block is already within `trigger_rms`, so this is safe to call
    unconditionally.

    Args:
        reconstruction: Modified in place - points and cameras together.
        cloud: The reference cloud to register onto.
        trigger_rms: Skip entirely when `residual_rms` is already below this,
            metres. Prealigning a block that is already close can only make it
            worse: ICP optimises geometric distance, which is a cruder objective
            than the bundle's.
        voxel_sizes: Coarse-to-fine ICP resolutions, metres, descending. The
            first should be comparable to the expected misalignment; the last to
            the accuracy wanted before handing over to the bundle.
        with_scale: Let ICP recover scale as well as pose. Needed whenever the
            block came from an uncontrolled SfM run, where scale is arbitrary.
        min_track_length: Ignore tie points seen by fewer images.
        verbose: Print the stage table.

    Returns:
        A `PrealignResult`. Check `rms_after` and `fitness` before trusting it.

    Raises:
        ValueError: If `voxel_sizes` is empty or the reconstruction has no
            usable points.
    """
    import open3d as o3d

    if not voxel_sizes:
        raise ValueError("voxel_sizes must not be empty")

    result = PrealignResult(
        rms_before=residual_rms(
            reconstruction, cloud, min_track_length=min_track_length
        )
    )
    if result.rms_before <= trigger_rms:
        result.rms_after = result.rms_before
        if verbose:
            print(result.summary())
        return result

    xyz = _sparse_xyz(reconstruction, min_track_length)
    if len(xyz) == 0:
        raise ValueError("no tie points with the requested track length")

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
    # The reference normals are already computed - reuse them rather than paying
    # for a second neighbourhood pass, and keep point-to-plane consistent with
    # the normals the constraint itself uses.
    target.normals = o3d.utility.Vector3dVector(cloud.normals)

    # Stage 1: centroid. Coarse enough to make the ICP basin reachable without
    # pretending to more accuracy than it has - see the module docstring for
    # why this does not also try to match cloud extent.
    source_centre, target_centre = xyz.mean(axis=0), cloud.points.mean(axis=0)
    current = np.eye(4)
    current[:3, 3] = target_centre - source_centre
    result.stages.append({
        "name": "centroid",
        "voxel": float("nan"),
        "fitness": float("nan"),
        "inlier_rmse": float("nan"),
        "scale": 1.0,
    })

    registration = o3d.pipelines.registration

    # Stage 2: ICP, coarse to fine. Point-to-point recovers scale, point-to-plane
    # then polishes - it cannot change scale, but by that point scale is right
    # and its normal-direction objective converges much closer on planar scenes.
    for voxel in voxel_sizes:
        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
        )
        for name, estimator in (
            (
                "ICP point-to-point",
                registration.TransformationEstimationPointToPoint(with_scale),
            ),
            ("ICP point-to-plane", registration.TransformationEstimationPointToPlane()),
        ):
            icp = registration.registration_icp(
                source_down,
                target_down,
                voxel * 2.0,
                current,
                estimator,
                registration.ICPConvergenceCriteria(max_iteration=60),
            )
            # A stage that found almost nothing is noise; keep the old estimate.
            if icp.fitness > 0.0:
                current = icp.transformation
            result.stages.append({
                "name": name,
                "voxel": voxel,
                "fitness": icp.fitness,
                "inlier_rmse": icp.inlier_rmse,
                "scale": float(np.cbrt(np.linalg.det(current[:3, :3]))),
            })
            result.fitness, result.inlier_rmse = icp.fitness, icp.inlier_rmse

    transform = _to_sim3d(current)
    reconstruction.transform(transform)
    result.sim3d = transform
    result.rms_after = residual_rms(
        reconstruction, cloud, min_track_length=min_track_length
    )
    if verbose:
        print(result.summary())
    return result
