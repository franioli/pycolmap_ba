"""Score a sparse reconstruction against the ENRICH ground-truth depth maps.

Why not CloudCompare against the Metashape dense cloud: that dense cloud was
built at *Lowest* quality (0.40 m point spacing, ~46 cm effective GSD), so a
cloud-to-cloud nearest-neighbour distance carries a 15-20 cm floor from
discretisation alone - before any real error. It measures the reference, not
the reconstruction.

The dataset ships per-pixel ground-truth depth instead. For each sparse point,
for each image that observes it, we project with the *reconstructed* pose,
sample the GT depth at that pixel and compare with the reconstructed depth.
That is a signed per-point error against exact truth: no nearest-neighbour
bias, no resolution floor, no dependence on another package.

Depth is z-depth along the optical axis (see enrich_data.load_depth).
"""

from pathlib import Path

import numpy as np
import pycolmap

import enrich_data

ROOT = Path(__file__).parent
SPARSE_PATH = ROOT / "nadir-2/colmap/sparse_gcp"

# Filtering: drop weak/poorly conditioned points before scoring.
MIN_TRACK_LENGTH = 3
MAX_REPROJ_ERROR = 2.0  # px

# A GT depth discontinuity (roof edge, tree) makes bilinear sampling
# meaningless, so skip pixels whose local neighbourhood is not smooth.
MAX_LOCAL_DEPTH_RANGE = 1.0  # m, over a 3x3 window


def nmad(x):
    """Normalised median absolute deviation - a robust sigma."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def sample_depth(depth, u, v, max_range=MAX_LOCAL_DEPTH_RANGE):
    """GT depth at (u, v), or None if the neighbourhood is not smooth."""
    col, row = int(u), int(v)
    if not (1 <= col < depth.shape[1] - 1 and 1 <= row < depth.shape[0] - 1):
        return None
    window = depth[row - 1 : row + 2, col - 1 : col + 2]
    if not np.isfinite(window).all() or window.max() - window.min() > max_range:
        return None  # depth discontinuity: sampling would be ambiguous
    return float(window[1, 1])


def depth_map_offset(truth):
    """Systematic offset of the GT depth maps themselves.

    Control experiment: take the exact GT GCP coordinates, the exact GT poses
    and the exact GT intrinsics, and compare against the depth maps. Everything
    on the input side is truth, so whatever offset remains belongs to the depth
    maps (a rendering convention), not to any reconstruction. Measured here at
    about +6 cm, and the sparse cloud shows the same, so it must be subtracted
    before quoting a systematic error.
    """
    focal, (cx, cy) = enrich_data.FOCAL_PX, enrich_data.PRINCIPAL_POINT
    residuals = []
    for image_name in sorted(truth.marker_data.image_name.unique()):
        depth = enrich_data.load_depth(image_name)
        pose = enrich_data.cam_from_world(truth, image_name)
        marks = truth.marker_data[truth.marker_data.image_name == image_name]
        for mark in marks.itertuples():
            point_in_cam = pose * truth.marker_xyz[mark.marker_name]
            if point_in_cam[2] <= 0:
                continue
            u = focal * point_in_cam[0] / point_in_cam[2] + cx
            v = focal * point_in_cam[1] / point_in_cam[2] + cy
            sampled = sample_depth(depth, u, v)
            if sampled is not None:
                residuals.append(point_in_cam[2] - sampled)
    return np.array(residuals)


def main():
    truth = enrich_data.load()
    rec = pycolmap.Reconstruction(SPARSE_PATH)
    print(
        f"reconstruction: {rec.num_reg_images()} images, "
        f"{rec.num_points3D()} points, "
        f"{rec.compute_mean_reprojection_error():.3f} px reprojection"
    )

    images = {im.name: im for im in rec.images.values()}
    depth_cache = {}
    residuals, track_lengths, xy = [], [], []

    for point_id, point in rec.points3D.items():
        if point.track.length() < MIN_TRACK_LENGTH or point.error > MAX_REPROJ_ERROR:
            continue
        for element in point.track.elements:
            image = rec.images[element.image_id]
            camera = rec.cameras[image.camera_id]

            # Reconstructed depth along the optical axis.
            point_in_cam = image.cam_from_world() * point.xyz
            if point_in_cam[2] <= 0:
                continue
            uv = camera.img_from_cam(point_in_cam)
            if uv is None:
                continue
            u, v = uv
            if not (1 <= u < camera.width - 1 and 1 <= v < camera.height - 1):
                continue

            if image.name not in depth_cache:
                depth_cache[image.name] = enrich_data.load_depth(image.name)
            sampled = sample_depth(depth_cache[image.name], u, v)
            if sampled is None:
                continue

            residuals.append(float(point_in_cam[2]) - sampled)
            track_lengths.append(point.track.length())
            xy.append(point.xyz[:2])

    residuals = np.array(residuals)
    track_lengths = np.array(track_lengths)
    xy = np.array(xy)

    control = depth_map_offset(truth)
    offset = float(np.median(control))
    print(
        f"\ncontrol (exact GT geometry vs the same depth maps): "
        f"n={len(control)}, median {offset * 100:+.2f} cm"
    )
    print("  this is the depth maps' own offset and must be subtracted below")

    print(f"\n{len(residuals)} point-image samples from {len(depth_cache)} depth maps")
    print(f"  median signed : {np.median(residuals) * 100:+7.2f} cm")
    print(
        f"  ... minus control: {(np.median(residuals) - offset) * 100:+6.2f} cm   "
        "<- systematic error of the reconstruction"
    )
    print(f"  mean signed   : {residuals.mean() * 100:+7.2f} cm")
    print(f"  NMAD          : {nmad(residuals) * 100:7.2f} cm   (robust sigma)")
    print(f"  std           : {residuals.std() * 100:7.2f} cm")
    print(
        f"  p05 / p95     : {np.percentile(residuals, 5) * 100:+7.2f} / "
        f"{np.percentile(residuals, 95) * 100:+7.2f} cm"
    )

    print("\n  by track length:")
    for lo, hi in [(3, 3), (4, 5), (6, 9), (10, 999)]:
        m = (track_lengths >= lo) & (track_lengths <= hi)
        if m.sum():
            label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 999 else '+'}"
            print(
                f"    {label:>5s} rays: n={m.sum():6d}  "
                f"median {np.median(residuals[m]) * 100:+6.2f} cm  "
                f"NMAD {nmad(residuals[m]) * 100:5.2f} cm"
            )

    # Radial trend reveals residual doming: a wrong focal bends the block into
    # a bowl, so the error grows with distance from the block centre.
    centre = xy.mean(axis=0)
    radius = np.linalg.norm(xy - centre, axis=1)
    print("\n  by distance from block centre (doming check):")
    edges = np.percentile(radius, [0, 25, 50, 75, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (radius >= lo) & (radius <= hi)
        print(
            f"    {lo:6.0f}-{hi:6.0f} m: n={m.sum():6d}  "
            f"median {np.median(residuals[m]) * 100:+6.2f} cm"
        )
    slope = np.polyfit(radius, residuals, 1)[0]
    print(
        f"    linear trend: {slope * 100:+.3f} cm per m of radius "
        "(near zero => no doming)"
    )


if __name__ == "__main__":
    main()
