"""Reference-cloud constrained bundle adjustment on ENRICH-Aerial nadir-2.

Does the LiDAR / TLS constraint actually work? Two questions, both answerable
here because the dataset ships ground truth for everything:

1. **Datum** - can a reference cloud replace the GCPs entirely? Run the bundle
   with *no* control at all and score the markers, every one of which is then a
   check point.
2. **Intrinsics** - can it recover a deliberately wrong focal length? Perturb
   the focal, let the bundle refine it, and see whether it comes back. A wrong
   focal bends the block into a dome, and the deformation-free reference surface
   is what makes that observable on a flat nadir block where the focal would
   otherwise trade against depth.

The reference cloud is the dataset's ground-truth mesh, sampled and given
Gaussian noise to stand in for a real survey. That is a *synthetic* reference:
perfectly registered, no occlusion, no vegetation, no moving objects. It answers
"is the estimator right", not "will this survive real data".

Usage::

    python enrich/run_cloud_ba.py                     # datum test
    python enrich/run_cloud_ba.py --perturb-focal 2   # intrinsics test, +2 %
    python enrich/run_cloud_ba.py --include-gcp        # the GCP bundle to beat
    python enrich/run_cloud_ba.py --preset recover     # blind-recovery self-test:
                                                        # 10 m / 5 deg / 200 ppm / 2%
                                                        # perturbation, recovered via
                                                        # the guard + pre-alignment
                                                        # path alone
    python enrich/run_cloud_ba.py --check-only         # compatibility check only,
                                                        # no bundle - the diagnostic
                                                        # to run on real, non-synthetic
                                                        # data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pycolmap

from pycolmap_ba import core, utils

import enrich_data

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/ENRICH-Aerial"
FREE_PATH = DATA_ROOT / "nadir-2/colmap/ref_cloud_ba/sparse"
MESH_PATH = DATA_ROOT / "nadir-2/colmap/ref_cloud_ba/ENRICH-Aerial_cut.obj"
CACHE_PATH = DATA_ROOT / "nadir-2/colmap/ref_cloud_ba/reference_cloud.npy"

# Reference cloud. The block covers about 820 x 680 m, so 5 M samples give
# roughly 9 points per square metre - a dense airborne survey. 2 cm noise is a
# realistic ALS range accuracy. The achieved spacing is measured and printed
# rather than assumed, since the mesh area exceeds the ground footprint.
N_SAMPLES = 5_000_000
CLOUD_NOISE_M = 0.02
NEIGHBORS = 20

# Correspondence gates. max_dist is deliberately loose: the free network starts
# tens of centimetres out, and a gate tighter than the initial misalignment
# leaves the block unable to pull itself in.
MAX_DIST_M = 1.0
MIN_TRACK_LENGTH = 3


def build_reference_cloud(
    *, rebuild: bool = False, verbose: bool = True
) -> core.ReferenceCloud:
    """Sample the ground-truth mesh into a noisy pseudo-LiDAR reference cloud.

    The mesh is 1.3 GB, so the samples are cached: only the first call pays for
    reading it.

    Args:
        rebuild: Re-sample even if the cache exists.
        verbose: Print timings.

    Returns:
        The reference cloud, patches already computed.
    """
    if rebuild or not CACHE_PATH.exists():
        import open3d as o3d

        started = time.perf_counter()
        mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
        sampled = mesh.sample_points_uniformly(N_SAMPLES)
        points = np.asarray(sampled.points, dtype=np.float64)
        np.save(CACHE_PATH, points)
        if verbose:
            print(
                f"sampled {len(points):,} points from the mesh in "
                f"{time.perf_counter() - started:.0f} s -> {CACHE_PATH.name}"
            )
    points = np.load(CACHE_PATH)

    # Noise on the *cached* samples, so the cache stays reusable at any sigma.
    points = points + np.random.default_rng(0).normal(0.0, CLOUD_NOISE_M, points.shape)
    started = time.perf_counter()
    cloud = core.ReferenceCloud.from_points(
        points, k=NEIGHBORS, sigma_ref=CLOUD_NOISE_M, sigma_slack=20.0
    )
    if verbose:
        # sqrt(lambda_3) is the patch half-extent, so the point spacing it
        # implies is a fair description of the cloud's actual resolution.
        spacing = 2 * np.median(np.sqrt(cloud.eigenvalues[:, 2])) / np.sqrt(NEIGHBORS)
        print(
            f"reference cloud: {len(cloud):,} patches in "
            f"{time.perf_counter() - started:.1f} s | "
            f"spacing ~{spacing:.2f} m | "
            f"roughness median {np.median(cloud.roughness):.3f} m | "
            f"planarity median {np.median(cloud.planarity):.2f}"
        )
    return cloud


def export(
    reconstruction: pycolmap.Reconstruction,
    cloud: core.ReferenceCloud,
    options: core.ReferenceCloudConstraints,
    output_path: Path,
    label: str,
) -> None:
    """Write the sparse cloud as PLY, coloured by its residual to the reference.

    The scalar field is what makes this worth opening in CloudCompare: a plain
    sparse cloud tells you nothing, whereas the signed distance to the reference
    surface shows *where* the block still does not fit - the radial pattern of a
    focal error, the tilt of a bad datum, the edges where the correspondence is
    simply ambiguous.

    Distances are signed along the patch normal, so a systematic offset reads as
    a one-sided colour ramp rather than a symmetric one.

    Args:
        reconstruction: Source of the points.
        cloud: Reference cloud to measure against.
        options: Supplies the `max_dist` used for the query.
        output_path: Folder to write into; created if missing.
        label: File stem, e.g. "before" or "after".
    """
    output_path.mkdir(parents=True, exist_ok=True)
    points = list(reconstruction.points3D.values())
    xyz = np.array([p.xyz for p in points])

    idx, sample_distance = cloud.query(xyz, options.max_dist)
    matched = idx >= 0
    safe = np.where(matched, idx, 0)
    signed = np.einsum("mi,mi->m", xyz - cloud.mu[safe], cloud.normals[safe])
    # The normal's sign is arbitrary per patch; orient it upward so the sign of
    # the residual is consistent across the block and readable as a colour ramp.
    signed *= np.sign(cloud.normals[safe][:, 2] + 1e-12)
    signed = np.where(matched, signed, np.nan)

    utils.save_point_cloud_as_ply(
        reconstruction,
        output_path / f"sparse_{label}.ply",
        surface_distance=signed,
        sample_distance=np.where(matched, sample_distance, np.nan),
    )
    inliers = np.isfinite(signed)
    print(
        f"  wrote sparse_{label}.ply | signed distance to reference: "
        f"median {np.median(signed[inliers]) * 100:+.2f} cm, "
        f"NMAD {utils.nmad(signed[inliers]) * 100:.2f} cm "
        f"({inliers.sum():,} / {len(points):,} matched)"
    )


def main() -> None:
    """Run the experiment selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perturb-focal",
        type=float,
        default=0.0,
        help="scale the focal length by this %% before adjusting",
    )
    parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="let the bundle refine focal length and distortion",
    )
    parser.add_argument(
        "--include-gcp",
        action="store_true",
        help="run the GCP-controlled bundle instead, as the reference result",
    )
    parser.add_argument(
        "--perturb-shift",
        type=float,
        default=0.0,
        help="translate the whole block by this many metres before adjusting",
    )
    parser.add_argument(
        "--perturb-rot",
        type=float,
        default=0.0,
        help="rotate the whole block by this many degrees about its centroid",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.0,
        help="scale the whole block by this many ppm about its centroid",
    )
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--rebuild-cloud", action="store_true")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="write PLYs here for CloudCompare (reference cloud + sparse "
        "before/after, carrying the signed distance to the reference surface)",
    )
    parser.add_argument(
        "--preset",
        choices=["recover"],
        default=None,
        help="apply a fixed perturbation preset instead of the individual "
        "--perturb-* flags: 'recover' is the blind-recovery self-test (10 m "
        "shift, 5 deg rotation, 200 ppm scale, 2%% focal) - overrides any "
        "--perturb-* flags given alongside it",
    )
    parser.add_argument(
        "--no-prealign",
        action="store_true",
        help="disable pre-alignment on a failed compatibility check (on_mismatch="
        "'warn' instead of 'prealign') - reproduces the old oracle-gated "
        "behaviour, where only a hand-sized gate schedule could recover a "
        "misaligned block",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the compatibility check against the reference cloud and exit - "
        "the diagnostic to run on real, non-synthetic data, where there is no "
        "known perturbation to size a gate schedule from",
    )
    args = parser.parse_args()

    if args.preset == "recover":
        args.perturb_shift = 10.0
        args.perturb_rot = 5.0
        args.perturb_scale = 200.0
        args.perturb_focal = 2.0
        if args.export is None:
            args.export = FREE_PATH.parent

    truth_data = enrich_data.load(subblock="nadir-2", split_control_check=True)
    control = truth_data.control_names if args.include_gcp else []
    check = truth_data.check_names if args.include_gcp else truth_data.marker_names
    truth = utils.GroundTruth(
        camera_xyz=truth_data.camera_xyz,
        marker_xyz=truth_data.marker_xyz,
        control_names=control,
        check_names=check,
        focal_px=enrich_data.FOCAL_PX,
    )
    # The markers are always in the bundle, but with `control_names=[]` in the
    # reference-cloud runs none of them gets a world prior. Every one is then a
    # check point, and their residual is the unbiased accuracy of a datum that
    # came from the cloud alone. Without their image observations they would not
    # be triangulated at all and there would be nothing to score.
    markers = core.MarkerConstraints(
        observations=truth_data.marker_data,
        xyz=truth_data.marker_xyz,
        control_names=control,
        sigma_px=enrich_data.gcp_sigma_px,
        sigma_m=enrich_data.gcp_sigma_m,
    )

    # Load the intial reconstruction (coarse aligment to the camera priors)
    rec = pycolmap.Reconstruction(FREE_PATH)

    # Push the block off the cloud with a known 7-DoF similarity. The tie-point
    # network is blind to this (a similarity leaves every reprojection residual
    # unchanged), so anything recovered is the reference cloud's doing. A no-op
    # `Perturbation()` (the default, no --perturb-* / --preset given) leaves the
    # block exactly where it started - the mode to use on real data.
    perturb = utils.Perturbation(
        shift_m=args.perturb_shift,
        rot_deg=args.perturb_rot,
        scale_ppm=args.perturb_scale,
        focal_pct=args.perturb_focal,
        seed=0,
    )
    # `report.reach`/`report.sim3d` are kept only as the final oracle-assertion
    # self-test below - checking whether the *blind* path (guard + prealign,
    # no knowledge of this perturbation) actually recovered it. Nothing in
    # between here and there is allowed to read them.
    perturb_report = utils.perturb_reconstruction(rec, perturb)
    if perturb:
        print(perturb_report.summary())

    utils.block_report(rec)
    utils.accuracy_report(rec, {}, truth, "free network")

    cloud = build_reference_cloud(rebuild=args.rebuild_cloud)

    if args.check_only:
        utils.banner("COMPATIBILITY CHECK ONLY")
        check = utils.check_cloud_compatibility(
            rec, cloud, utils.CompatibilityOptions(max_dist=MAX_DIST_M)
        )
        print(check.summary())
        return

    solver = core.SolverOptions(
        refine_focal_length=args.refine_intrinsics,
        refine_extra_params=args.refine_intrinsics,
        min_track_length=0,
        function_tolerance=1e-8,
    )
    options = core.ReferenceCloudConstraints(
        max_dist=MAX_DIST_M,
        min_track_length=MIN_TRACK_LENGTH,
        max_roughness=4 * CLOUD_NOISE_M,
        robust_loss_scale=2.0,
    )

    # Blind, fixed coarse-to-fine schedule - no longer sized from the known
    # perturbation. Large misalignments are now the guard + pre-alignment's
    # job (`on_mismatch="prealign"` below), not this schedule's; it only has
    # to cover ordinary refinement, hence a fixed multiple of the final gate.
    schedule = list(np.geomspace(4.0 * MAX_DIST_M, MAX_DIST_M, num=3))
    print(f"gate schedule: {[round(g, 2) for g in schedule]} m")

    if args.export:
        utils.save_points_as_ply(cloud.points, args.export / "reference_cloud.ply")
        print(f"  wrote reference_cloud.ply ({len(cloud):,} points)")
        export(rec, cloud, options, args.export, "before")

    result = core.run_cloud_constrained_bundle_adjustment(
        rec,
        cloud,
        reference_options=options,
        max_dist_schedule=schedule,
        solver=solver,
        markers=markers,
        filters=utils.FilterOptions(),
        tie_points_sigma_px=enrich_data.tie_points_sigma_px,
        max_passes=args.max_passes,
        on_mismatch="warn" if args.no_prealign else "prealign",
    )
    if result.compatibility is not None:
        print(f"\n{result.compatibility.summary()}")
    if result.prealign is not None:
        print(result.prealign.summary())

    utils.block_report(
        rec, result, cloud=cloud, solver=solver, label="reference-cloud bundle"
    )
    utils.accuracy_report(rec, result.marker_xyz, truth, "reference-cloud bundle")

    if args.export:
        export(rec, cloud, options, args.export, "after")

    if perturb:
        # Self-test: did the blind path (guard + prealign inside the BA, no
        # oracle) actually undo a perturbation whose reach is known here only
        # because this is a synthetic test - never used to size anything above.
        rmse = utils.camera_center_rmse(rec, truth.camera_xyz)
        recovered = rmse < 0.10  # ~ back to the few-cm baseline a GCP bundle gets
        prealigned = result.prealign is not None and result.prealign.sim3d is not None
        print(
            f"\nself-test: known perturbation reach {perturb_report.reach:.2f} m, "
            f"prealign triggered: {prealigned} -> final camera centre RMSE "
            f"{rmse:.3f} m ({'RECOVERED' if recovered else 'NOT RECOVERED'})"
        )
    else:
        print(
            f"\nfinal camera centre RMSE to truth: "
            f"{utils.camera_center_rmse(rec, truth.camera_xyz):.3f} m"
        )


if __name__ == "__main__":
    main()
