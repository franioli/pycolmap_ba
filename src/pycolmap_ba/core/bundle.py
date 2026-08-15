"""Backend: assemble and solve one customised bundle adjustment.

One entry point, `run_bundle_adjustment`, covers a plain bundle, marker (GCP)
controlled georeferencing, GNSS-constrained adjustment, reference-cloud
(LiDAR / TLS) constrained adjustment, or any combination, by adding the blocks
from `gcp`, `camera_pose` and `reference_cloud` to COLMAP's own Ceres problem
before solving it.

Deliberately just the one solve. The things that are not part of assembling
and solving it live elsewhere:

* `cloud_bundle`     - alternating solve / re-correspondence around this
  function, in this package
* `utils.filtering`  - post-solve removal of unsupportable points
* `utils.covariance` - posterior covariances of the result
* `utils.prealign`   - getting the block into range before any of this

Examples:
    Markers as control, plus camera position priors::

        result = run_bundle_adjustment(
            rec,
            solver=SolverOptions(min_track_length=3),
            markers=MarkerConstraints(
                observations=marker_df,          # image_name, marker_name, u, v
                xyz=surveyed_xyz,
                control_names=control_names,     # the rest become check points
                sigma_px=1.0,
                sigma_m=(0.02, 0.02, 0.03),
            ),
            camera_poses=CameraPoseConstraints(positions=centres, sigma_m=0.01),
            filters=FilterOptions(),
            tie_points_sigma_px=0.5,
        )
        adjusted = result.marker_xyz

    Markers only, no camera constraints - the honest georeferencing setup::

        result = run_bundle_adjustment(rec, markers=MarkerConstraints(...))

    A plain bundle, nothing added - equivalent to COLMAP's global BA::

        result = run_bundle_adjustment(rec)

    A LiDAR / TLS cloud as the whole datum, no GCPs, with the intrinsics free::

        cloud = ReferenceCloud.from_points(lidar_xyz, sigma_ref=0.02)
        result = run_cloud_constrained_bundle_adjustment(
            rec, cloud, solver=SolverOptions(refine_extra_params=True)
        )
        print(datum_information(...).summary())   # did it really fix all 7 DoF?

For full control over Ceres itself, build the adjuster as done here and then
call `options.ceres.create_solver_options(config, adjuster.problem)` followed by
`pyceres.solve(...)`, as in `pycolmap_examples/custom_bundle_adjustment.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pycolmap

from ..utils.covariance import estimate_covariance
from ..utils.filtering import FilterOptions, filter_reconstruction
from .camera_pose import (
    CameraPoseConstraints,
    add_camera_pose_constraints,
)
from .ref_cloud import (
    CorrespondenceStats,
    ReferenceCloud,
    ReferenceCloudConstraints,
    add_reference_cloud_constraints,
)
from .gcp import MarkerConstraints, add_marker_constraints

__all__ = [
    "BundleResult",
    "SolverOptions",
    "build_config",
    "build_options",
    "run_bundle_adjustment",
    "solve_to_convergence",
]


@dataclass
class SolverOptions:
    """Everything that goes into `pycolmap.BundleAdjustmentOptions`.

    Attributes:
        refine_focal_length: Optimise the focal length. On a flat nadir block
            this is degenerate without control - focal trades against scene
            depth - so keep it False unless markers or priors constrain it.
        refine_principal_point: Optimise cx, cy. Usually False; the principal
            point is weakly determined and readily absorbs other errors.
        refine_extra_params: Optimise distortion coefficients.
        refine_points3D: Optimise the 3D tie points themselves.
        min_track_length: Exclude points with fewer observations from the
            problem. 0 disables the test. NOTE: such points are *ignored*, not
            deleted - they keep their pre-bundle coordinates and still reach any
            export. Pair this with `FilterOptions.min_track_length`.
        gauge: How to fix the 7-DoF datum of a free network. None (the default)
            decides automatically: no gauge fixing when a world-anchoring
            constraint is present - marker world priors or camera pose priors
            already define the datum - and `TWO_CAMS_FROM_WORLD` otherwise,
            matching what COLMAP's own mapper does.
        loss_function_type: Robustifier for the tie-point residuals. COLMAP
            itself uses SOFT_L1 at scale 1.0 for local BA and TRIVIAL for global
            BA, so TRIVIAL reproduces its global default.
        loss_function_scale: Scale of the above, in pixels.
        max_num_iterations: Ceres iteration cap.
        function_tolerance: Relative cost-decrease stopping test. COLMAP ships
            0.0, which *disables* it, leaving only the absolute
            gradient_tolerance; large blocks then run to the iteration cap and
            report NO_CONVERGENCE even though the cost stopped improving. 1e-8
            restores the usual behaviour.
        gradient_tolerance: Absolute gradient max-norm stopping test.
        parameter_tolerance: Relative step-size stopping test (0.0 = disabled,
            matching COLMAP).
        max_solve_restarts: Re-invoke the solver up to this many times while it
            reports non-convergence; each call resumes from the current state.
        print_summary: Let COLMAP print its own bundle-adjustment report.
    """

    refine_focal_length: bool = True
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    refine_points3D: bool = True
    min_track_length: int = 0
    gauge: Any = None  # None -> decided from the constraints
    loss_function_type: Any = None  # None -> pycolmap.LossFunctionType.TRIVIAL
    loss_function_scale: float = 1.0
    max_num_iterations: int = 200
    function_tolerance: float = 1e-8
    gradient_tolerance: float = 1e-4
    parameter_tolerance: float = 0.0
    max_solve_restarts: int = 3
    print_summary: bool = False


@dataclass
class BundleResult:
    """Outcome of `run_bundle_adjustment`.

    Attributes:
        reconstruction: The adjusted reconstruction (mutated in place).
        marker_xyz: Adjusted marker coordinates, empty if no markers were used.
            These are parameter blocks owned by the backend, *not* Point3D
            entries in the reconstruction.
        summary: The final `BundleAdjustmentSummary`, None if nothing was solved.
        converged: Whether the solver reported CONVERGENCE.
        usable: Whether Ceres considers the solution usable. A solve can stop on
            the iteration cap - not converged - and still be perfectly usable.
        n_blocks: Residual-block counts by kind - `tie_points`, `marker_image`,
            `marker_world`, `camera_pose`, `reference_cloud`.
        n_filtered: Observations removed by post-solve filtering.
        covariance: `pycolmap.BACovariance` when `covariance=True` was requested
            and the estimation succeeded, else None.
        correspondences: Reference-cloud correspondence stats of this solve,
            None if no reference cloud was used.
        passes: Per-pass records from
            `run_cloud_constrained_bundle_adjustment`, empty for a single solve.
    """

    reconstruction: pycolmap.Reconstruction
    marker_xyz: dict[str, np.ndarray] = field(default_factory=dict)
    summary: Any = None
    converged: bool = False
    usable: bool = False
    n_blocks: dict[str, int] = field(default_factory=dict)
    n_filtered: int = 0
    covariance: Any = None
    correspondences: CorrespondenceStats | None = None
    passes: list[dict[str, Any]] = field(default_factory=list)


def build_options(solver: SolverOptions) -> pycolmap.BundleAdjustmentOptions:
    """Translate `SolverOptions` into a `pycolmap.BundleAdjustmentOptions`.

    Args:
        solver: The settings to apply.

    Returns:
        A configured options object ready for
        `create_default_ceres_bundle_adjuster`.
    """
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = solver.refine_focal_length
    options.refine_principal_point = solver.refine_principal_point
    options.refine_extra_params = solver.refine_extra_params
    options.refine_points3D = solver.refine_points3D
    options.print_summary = solver.print_summary
    # Lives on the options object itself, unlike the loss/solver settings.
    options.min_track_length = solver.min_track_length

    # The loss knobs live on `ceres`, mirroring the C++ options.ceres->...
    options.ceres.loss_function_type = (
        solver.loss_function_type or pycolmap.LossFunctionType.TRIVIAL
    )
    options.ceres.loss_function_scale = solver.loss_function_scale

    solver_options = options.ceres.solver_options
    solver_options.max_num_iterations = solver.max_num_iterations
    solver_options.function_tolerance = solver.function_tolerance
    solver_options.gradient_tolerance = solver.gradient_tolerance
    solver_options.parameter_tolerance = solver.parameter_tolerance

    return options


def build_config(
    reconstruction: pycolmap.Reconstruction,
    constant_intrinsics: bool = False,
    image_ids: Sequence[int] | None = None,
    gauge: Any = None,
) -> pycolmap.BundleAdjustmentConfig:
    """Build the config selecting which images, intrinsics and gauge take part.

    Args:
        reconstruction: Source of the registered images.
        constant_intrinsics: Hold every camera's intrinsics fixed.
        image_ids: Restrict to these image ids. None uses all registered ones.
        gauge: A `pycolmap.BundleAdjustmentGauge` to fix the datum, or None to
            leave the network free. A free network is determined only up to a
            7-DoF similarity, so without either a gauge or world constraints the
            normal equations are rank-deficient by 7.

    Returns:
        The populated config.
    """
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in image_ids or reconstruction.reg_image_ids():
        config.add_image(image_id)
    if constant_intrinsics:
        for camera_id in reconstruction.cameras:
            config.set_constant_cam_intrinsics(camera_id)
    if gauge is not None:
        config.fix_gauge(gauge)
    return config


def solve_to_convergence(
    bundle_adjuster: Any, max_restarts: int = 3, verbose: bool = True
) -> tuple[Any, bool]:
    """Solve, re-invoking while the solver reports non-convergence.

    Each call resumes from the current parameter values, so a second attempt
    continues rather than restarting. With `function_tolerance` re-enabled this
    normally converges on the first attempt.

    Args:
        bundle_adjuster: The adjuster whose problem carries all the blocks.
        max_restarts: Maximum number of `solve()` calls.
        verbose: Print the termination type of each attempt.

    Returns:
        `(summary, converged)` for the final attempt.
    """
    summary = None
    for attempt in range(1, max_restarts + 1):
        summary = bundle_adjuster.solve()
        converged = (
            summary.termination_type
            == pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
        )
        if verbose:
            print(f"  solve {attempt}: {summary.termination_type}")
        if converged:
            return summary, True
    return summary, False


def run_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    *,
    solver: SolverOptions | None = None,
    markers: MarkerConstraints | None = None,
    camera_poses: CameraPoseConstraints | None = None,
    reference_cloud: ReferenceCloud | None = None,
    reference_options: ReferenceCloudConstraints | None = None,
    filters: FilterOptions | None = None,
    tie_points_sigma_px: float = 0.5,
    image_ids: Sequence[int] | None = None,
    covariance: bool = False,
    verbose: bool = True,
) -> BundleResult:
    """Assemble and solve a bundle adjustment with optional extra constraints.

    Every constraint is optional, so the same call covers a plain bundle, marker
    controlled georeferencing, GNSS-constrained adjustment, reference-cloud
    (LiDAR / TLS) constrained adjustment, or any combination.

    Args:
        reconstruction: Adjusted in place.
        solver: Solver and refinement settings. None uses `SolverOptions()`.
        markers: Marker observations and world coordinates. None adds neither.
        camera_poses: Prior camera poses. None leaves the cameras free.
        reference_cloud: Georeferenced LiDAR / TLS cloud to pull the tie points
            onto. Correspondences are resolved once, here; refreshing them needs
            an outer loop, so prefer `run_cloud_constrained_bundle_adjustment`
            unless the block is already well aligned.
        reference_options: Gates for the above. None uses the defaults, which
            only matters when `reference_cloud` is given.
        filters: Post-solve filtering. None skips it.
        tie_points_sigma_px: Assumed 1-sigma of the tie-point observations,
            pixels. All other sigmas are expressed relative to this.
        image_ids: Restrict the adjustment to these images.
        covariance: Also estimate posterior covariances (see
            `estimate_covariance`; may return None).
        verbose: Print the weighting summary, block counts and progress.

    Returns:
        A `BundleResult` with the adjusted marker coordinates and diagnostics.

    Raises:
        AssertionError: If the assembled problem does not contain exactly the
            expected number of residual blocks. This catches the classic failure
            of building a second adjuster, which silently orphans every block
            added here and leaves the solve running on tie points alone.
    """
    solver = solver or SolverOptions()
    options = build_options(solver)

    # Anything that ties the block to world coordinates also fixes the datum.
    # Only then is gauge fixing wrong: it would over-constrain the network. A
    # reference cloud counts, but only conditionally - on a plane-only scene it
    # leaves the horizontal datum free, so check `datum_information` rather than
    # assuming. `solver.gauge` stays available to force the issue either way.
    anchored = (
        markers is not None or camera_poses is not None or reference_cloud is not None
    )
    gauge = solver.gauge
    if gauge is None and not anchored:
        gauge = pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD

    config = build_config(
        reconstruction,
        constant_intrinsics=not (
            solver.refine_focal_length
            or solver.refine_principal_point
            or solver.refine_extra_params
        ),
        image_ids=image_ids,
        gauge=gauge,
    )

    if verbose:
        print(f"tie points      : sigma {tie_points_sigma_px} px  (reference unit)")
        if markers is not None:
            print(
                f"marker image obs: sigma {markers.sigma_px} px -> "
                f"{markers.sigma_px / tie_points_sigma_px:.2f} in tie units"
            )
            print(
                f"marker world    : sigma {np.asarray(markers.sigma_m)} m -> "
                f"{np.asarray(markers.sigma_m) / tie_points_sigma_px} in tie units"
            )
        if camera_poses is not None:
            print(
                f"camera positions: sigma {camera_poses.sigma_m} m -> "
                f"{np.asarray(camera_poses.sigma_m) / tie_points_sigma_px} "
                "in tie units"
            )
            if camera_poses.sigma_rot_deg is not None:
                print(f"camera rotations: sigma {camera_poses.sigma_rot_deg} deg")
        print(f"gauge           : {gauge}")

    # Negative-depth points make the problem degenerate; COLMAP's own mapper
    # drops them before every global bundle.
    pycolmap.ObservationManager(
        reconstruction
    ).filter_observations_with_negative_depth()

    # Create the adjuster ONCE and add every block to *its* problem. Building a
    # second adjuster over the same config/reconstruction yields a different
    # ceres::Problem; the blocks added below would then be orphaned and the
    # solve would silently run on tie points alone.
    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        options, config, reconstruction
    )
    problem = bundle_adjuster.problem
    n_blocks = {
        "tie_points": problem.num_residual_blocks(),
        "marker_image": 0,
        "marker_world": 0,
        "camera_pose": 0,
        "reference_cloud": 0,
    }
    correspondences: CorrespondenceStats | None = None

    # pyceres does not take ownership of cost functions: without this list they
    # are garbage collected while Ceres still holds the pointers.
    retained: list[Any] = []
    marker_xyz: dict[str, np.ndarray] = {}

    if markers is not None:
        marker_xyz, n_image, n_world = add_marker_constraints(
            problem, reconstruction, markers, tie_points_sigma_px, retained
        )
        n_blocks["marker_image"] = n_image
        n_blocks["marker_world"] = n_world

    if camera_poses is not None:
        n_blocks["camera_pose"] = add_camera_pose_constraints(
            problem, reconstruction, camera_poses, tie_points_sigma_px, retained
        )

    if reference_cloud is not None:
        n_blocks["reference_cloud"], correspondences = add_reference_cloud_constraints(
            problem,
            reconstruction,
            reference_cloud,
            reference_options or ReferenceCloudConstraints(),
            tie_points_sigma_px,
            retained,
        )
        if verbose:
            print(f"  {correspondences.summary()}")

    expected = sum(n_blocks.values())
    assert problem.num_residual_blocks() == expected, (
        f"{problem.num_residual_blocks()} residual blocks, expected {expected} - "
        "the added constraints are not in the problem being solved"
    )
    if verbose:
        print(
            f"  problem: {expected:,} blocks = {n_blocks['tie_points']:,} tie points"
            f" + {expected - n_blocks['tie_points']} added "
            f"({n_blocks['marker_image']} marker image, "
            f"{n_blocks['marker_world']} marker world, "
            f"{n_blocks['camera_pose']} camera pose, "
            f"{n_blocks['reference_cloud']} reference cloud)"
        )

    summary, converged = solve_to_convergence(
        bundle_adjuster, solver.max_solve_restarts, verbose
    )
    usable = bool(summary is not None and summary.is_solution_usable())
    if verbose and not converged:
        print(f"  solution usable: {usable}")
        print(summary.brief_report() if summary is not None else "  no summary")

    ba_covariance = None
    if covariance:
        ba_covariance = estimate_covariance(reconstruction, bundle_adjuster)
        if verbose and ba_covariance is None:
            print("  covariance estimation failed (problem structure)")

    n_filtered = 0
    if filters is not None:
        n_filtered = filter_reconstruction(reconstruction, filters, verbose)

    return BundleResult(
        reconstruction=reconstruction,
        marker_xyz=marker_xyz,
        summary=summary,
        converged=converged,
        usable=usable,
        n_blocks=n_blocks,
        n_filtered=n_filtered,
        covariance=ba_covariance,
        correspondences=correspondences,
    )
