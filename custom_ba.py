"""Composable custom bundle adjustment on top of pycolmap / pyceres.

COLMAP's own bundle adjuster handles tie points. This module lets you add
extra constraints to *that same* Ceres problem - marker (GCP/check-point)
observations, marker world coordinates, camera position priors - and control
the solver and post-solve filtering, without rewriting the boilerplate each
time.

Dataset-agnostic on purpose: inputs are plain dicts / DataFrames / arrays.

Examples
--------
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
        camera_priors=CameraPriorConstraints(xyz=camera_centers, sigma_m=0.01),
        filters=FilterOptions(),
        tie_points_sigma_px=0.5,
    )
    adjusted = result.marker_xyz

Markers only, no camera constraints (the honest georeferencing setup)::

    result = run_bundle_adjustment(rec, markers=MarkerConstraints(...))

A plain bundle, nothing added - equivalent to COLMAP's global BA::

    result = run_bundle_adjustment(rec)

Weighting
---------
All sigmas are given in **native units** (pixels, metres). COLMAP adds its
tie-point reprojection residuals *unweighted*, i.e. with an implicit sigma of
1 px, so every other sigma is internally divided by `tie_points_sigma_px` to
put the whole problem in one consistent whitened least-squares system. Callers
never do that arithmetic themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyceres
import pycolmap
from pycolmap import cost_functions

__all__ = [
    "BundleResult",
    "CameraPriorConstraints",
    "FilterOptions",
    "MarkerConstraints",
    "SolverOptions",
    "add_camera_priors",
    "add_marker_constraints",
    "build_config",
    "build_options",
    "filter_reconstruction",
    "run_bundle_adjustment",
    "solve_to_convergence",
]


# ===========================================================================
# Options
# ===========================================================================
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
            deleted - they keep their pre-bundle coordinates and still reach
            any export. Pair this with `FilterOptions.min_track_length`.
        loss_function_type: Robustifier for the tie-point residuals. COLMAP
            itself uses SOFT_L1 at scale 1.0 for local BA and TRIVIAL for
            global BA, so TRIVIAL reproduces its global default.
        loss_function_scale: Scale of the above, in pixels.
        max_num_iterations: Ceres iteration cap.
        function_tolerance: Relative cost-decrease stopping test. COLMAP ships
            0.0, which *disables* it, leaving only the absolute
            gradient_tolerance; large blocks then run to the iteration cap and
            report NO_CONVERGENCE even though the cost stopped improving.
            1e-8 restores the usual behaviour.
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
    loss_function_type: Any = None  # None -> pycolmap.LossFunctionType.TRIVIAL
    loss_function_scale: float = 1.0
    max_num_iterations: int = 200
    function_tolerance: float = 1e-8
    gradient_tolerance: float = 1e-4
    parameter_tolerance: float = 0.0
    max_solve_restarts: int = 3
    print_summary: bool = False


@dataclass
class MarkerConstraints:
    """Marker (GCP / check point) observations and world coordinates.

    A marker contributes two independent kinds of information: its image
    measurements (always used, for every marker) and its surveyed world
    coordinate (used only for markers listed in `control_names`). Markers left
    out of `control_names` become **check points**: they are triangulated by
    the bundle but never constrained, so their error is an unbiased accuracy
    estimate.

    Attributes:
        observations: Columns `image_name`, `marker_name`, `u`, `v`. Image
            coordinates in pixels, in the same convention as the keypoints
            (COLMAP stores them at full resolution).
        xyz: Surveyed / initial world coordinates per marker name, metres.
            Used to initialise every marker and to constrain the control ones.
        control_names: Markers to constrain with their world coordinate.
            None means *all* markers are control (no check points).
        sigma_px: 1-sigma accuracy of the image measurements, pixels. Marking
            a target by hand is typically far less precise than an automatic
            tie point, so this is usually larger than `tie_points_sigma_px`.
        sigma_m: 1-sigma accuracy of the surveyed coordinates, metres. Scalar
            or (x, y, z); height is commonly looser than planimetry. Set it to
            what the survey actually achieved - tighter is not safer, it just
            forces the block onto survey noise.
        robust_loss_scale: Cauchy scale for the marker *image* residuals, to
            absorb mis-identified targets. None disables robustification. The
            world priors are always plain quadratic.
    """

    observations: pd.DataFrame
    xyz: dict[str, np.ndarray]
    control_names: Sequence[str] | None = None
    sigma_px: float = 1.0
    sigma_m: float | Sequence[float] = (0.02, 0.02, 0.03)
    robust_loss_scale: float | None = 2.8


@dataclass
class CameraPriorConstraints:
    """Prior positions for the camera projection centres.

    Typically GNSS. Be aware that once the cameras are constrained, the
    resulting camera-position residual is a *fit* to data the bundle was
    given, not an independent accuracy figure.

    Attributes:
        xyz: Prior camera **centres** (not translation vectors) per image name,
            in world coordinates, metres.
        sigma_m: 1-sigma accuracy, metres. Scalar or (x, y, z).
        robust_loss_scale: Cauchy scale for the priors. None = plain quadratic.
    """

    xyz: dict[str, np.ndarray]
    sigma_m: float | Sequence[float] = 0.10
    robust_loss_scale: float | None = None


@dataclass
class FilterOptions:
    """Post-solve filtering, applied through COLMAP's `ObservationManager`.

    Unlike `SolverOptions.min_track_length`, these actually delete points, and
    they go through the observation manager so COLMAP's internal correspondence
    bookkeeping stays consistent (raw `Reconstruction.delete_point3D` does not
    maintain it).

    Attributes:
        max_reproj_error: Drop points above this reprojection error, pixels.
        min_tri_angle_deg: Drop points whose best pairwise triangulation angle
            is below this, **degrees** (COLMAP converts internally). On a block
            with good overlap this is typically a no-op.
        min_track_length: Drop points with fewer observations. 0 disables.
    """

    max_reproj_error: float = 4.0
    min_tri_angle_deg: float = 1.5
    min_track_length: int = 0


@dataclass
class BundleResult:
    """Outcome of `run_bundle_adjustment`.

    Attributes:
        reconstruction: The adjusted reconstruction (mutated in place).
        marker_xyz: Adjusted marker coordinates, empty if no markers were used.
            These are parameter blocks owned by this module, *not* Point3D
            entries in the reconstruction.
        summary: The final `BundleAdjustmentSummary`, None if nothing was
            solved.
        converged: Whether the solver reported CONVERGENCE.
        n_blocks: Residual-block counts by kind - `tie_points`,
            `marker_image`, `marker_world`, `camera`.
        n_filtered: Observations removed by post-solve filtering.
    """

    reconstruction: pycolmap.Reconstruction
    marker_xyz: dict[str, np.ndarray] = field(default_factory=dict)
    summary: Any = None
    converged: bool = False
    n_blocks: dict[str, int] = field(default_factory=dict)
    n_filtered: int = 0


# ===========================================================================
# Builders
# ===========================================================================
def _cov_in_tie_units(
    sigma: float | Sequence[float], tie_points_sigma_px: float, size: int
) -> np.ndarray:
    """Diagonal covariance for `sigma`, expressed in tie-point sigma units.

    COLMAP adds tie-point reprojection residuals unweighted, i.e. with an
    implicit sigma of 1 px. Only the *ratios* between sigmas matter, so
    multiplying the whole objective by sigma_tie^2 leaves the tie-point term
    matching COLMAP's blocks exactly and rescales every other sigma by
    1 / sigma_tie. That is what this does.

    Args:
        sigma: Scalar or per-axis 1-sigma, in native units (px or m).
        tie_points_sigma_px: Reference tie-point sigma, pixels.
        size: Dimension of the residual (2 for image, 3 for world).

    Returns:
        A `size` x `size` diagonal covariance matrix.
    """
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (size,))
    return np.diag((s / tie_points_sigma_px) ** 2)


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
    loss_type = solver.loss_function_type
    if loss_type is None:
        loss_type = pycolmap.LossFunctionType.TRIVIAL
    options.ceres.loss_function_type = loss_type
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
) -> pycolmap.BundleAdjustmentConfig:
    """Build the config selecting which images and intrinsics take part.

    Args:
        reconstruction: Source of the registered images.
        constant_intrinsics: Hold every camera's intrinsics fixed.
        image_ids: Restrict to these image ids. None uses all registered ones.

    Returns:
        The populated config.
    """
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in image_ids or reconstruction.reg_image_ids():
        config.add_image(image_id)
    if constant_intrinsics:
        for camera_id in reconstruction.cameras:
            config.set_constant_cam_intrinsics(camera_id)
    return config


# ===========================================================================
# Constraints
# ===========================================================================
def add_marker_constraints(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    markers: MarkerConstraints,
    tie_points_sigma_px: float,
    retained: list[Any],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Add marker image residuals and world priors to an existing problem.

    Each marker becomes a 3-vector parameter block owned by this module, so
    the reconstruction's Point3D / Track structures are never touched. Image
    residuals tie that block to the poses and intrinsics; world priors tie it
    to the surveyed coordinate.

    Args:
        problem: The Ceres problem from the *single* bundle adjuster.
        reconstruction: Provides images, cameras and pose parameter blocks.
        markers: Observations, coordinates, control list and sigmas.
        tie_points_sigma_px: Reference sigma for the weighting (see module doc).
        retained: List that keeps cost functions alive - pyceres does not take
            ownership, and a garbage-collected cost function leaves the solver
            reading freed memory.

    Returns:
        `(marker_xyz, n_image_blocks, n_world_blocks)` where `marker_xyz` maps
        marker name to the parameter block the solver will optimise.
    """
    image_cov = _cov_in_tie_units(markers.sigma_px, tie_points_sigma_px, 2)
    world_cov = _cov_in_tie_units(markers.sigma_m, tie_points_sigma_px, 3)

    # Own, mutable copies: these are the parameter blocks Ceres writes into.
    marker_xyz = {name: np.array(xyz, dtype=float) for name, xyz in markers.xyz.items()}
    by_name = {im.name: im for im in reconstruction.images.values()}
    robust = (
        pyceres.CauchyLoss(markers.robust_loss_scale)
        if markers.robust_loss_scale is not None
        else None
    )

    n_image = 0
    for obs in markers.observations.itertuples():
        image = by_name.get(obs.image_name)
        if image is None or obs.marker_name not in marker_xyz:
            continue  # marker not visible in this reconstruction
        camera = reconstruction.cameras[image.camera_id]
        cost = cost_functions.ReprojErrorCost(
            camera.model, image_cov, np.array([obs.u, obs.v])
        )
        retained.append(cost)
        # Block order is positional: [point3D(3), cam_from_world(7), camera(N)].
        # For a single-camera rig cam_from_world == rig_from_world, and these
        # are live views into COLMAP's memory, so they are the same parameter
        # blocks COLMAP optimises - with the quaternion manifold already set.
        problem.add_residual_block(
            cost,
            robust,
            [
                marker_xyz[obs.marker_name],
                reconstruction.frames[image.frame_id].rig_from_world.params,
                camera.params,
            ],
        )
        n_image += 1

    # Only control markers get a world constraint; the rest stay free and act
    # as check points.
    control = (
        list(markers.control_names)
        if markers.control_names is not None
        else list(marker_xyz)
    )
    n_world = 0
    for name in control:
        if name not in marker_xyz:
            continue
        # NormalPrior(mean, covariance) whitens internally: the residual is
        # chol(cov^-1)^T (x - mean), i.e. the Mahalanobis distance.
        cost = pyceres.factors.NormalPrior(markers.xyz[name], world_cov)
        retained.append(cost)
        problem.add_residual_block(cost, None, [marker_xyz[name]])
        n_world += 1

    return marker_xyz, n_image, n_world


def add_camera_priors(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    priors: CameraPriorConstraints,
    tie_points_sigma_px: float,
    retained: list[Any],
) -> int:
    """Constrain camera projection centres to prior positions.

    Args:
        problem: The Ceres problem from the single bundle adjuster.
        reconstruction: Provides the pose parameter blocks.
        priors: Prior centres and their sigma.
        tie_points_sigma_px: Reference sigma for the weighting.
        retained: Keep-alive list for the cost functions.

    Returns:
        Number of residual blocks added (one per registered image with a prior).
    """
    cov = _cov_in_tie_units(priors.sigma_m, tie_points_sigma_px, 3)
    robust = (
        pyceres.CauchyLoss(priors.robust_loss_scale)
        if priors.robust_loss_scale is not None
        else None
    )

    n_added = 0
    for image in reconstruction.images.values():
        if image.name not in priors.xyz:
            continue
        # AbsolutePosePositionPriorCost, NOT NormalPrior. COLMAP's pose
        # parameter block is rig_from_world = (quaternion, t), and the camera
        # centre is C = -R^T t. This cost's residual evaluates to
        # R * (C_prior - C_est), so the sigma really is metres on the CENTRE.
        # A NormalPrior on the same block would constrain t instead, which is
        # a different quantity and silently wrong.
        cost = cost_functions.AbsolutePosePositionPriorCost(
            cov, np.asarray(priors.xyz[image.name], dtype=float)
        )
        retained.append(cost)
        problem.add_residual_block(
            cost, robust, [reconstruction.frames[image.frame_id].rig_from_world.params]
        )
        n_added += 1
    return n_added


# ===========================================================================
# Solve and filter
# ===========================================================================
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


def filter_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    filters: FilterOptions,
    verbose: bool = True,
) -> int:
    """Delete weak points through COLMAP's `ObservationManager`.

    Uses the observation manager rather than `Reconstruction.delete_point3D`
    so COLMAP's correspondence bookkeeping stays consistent.

    Args:
        reconstruction: Modified in place.
        filters: Thresholds to apply.
        verbose: Print how many observations were removed.

    Returns:
        Number of filtered observations.
    """
    manager = pycolmap.ObservationManager(reconstruction)
    # Covers reprojection error, negative depth and triangulation angle.
    n_filtered = manager.filter_all_points3D(
        filters.max_reproj_error, filters.min_tri_angle_deg
    )
    if filters.min_track_length > 0:
        n_filtered += manager.filter_points3D_with_short_tracks(
            filters.min_track_length
        )
    if verbose:
        print(
            f"  filtered {n_filtered} observations -> "
            f"{reconstruction.num_points3D():,} points remain"
        )
    return n_filtered


# ===========================================================================
# Entry point
# ===========================================================================
def run_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    *,
    solver: SolverOptions | None = None,
    markers: MarkerConstraints | None = None,
    camera_priors: CameraPriorConstraints | None = None,
    filters: FilterOptions | None = None,
    tie_points_sigma_px: float = 0.5,
    image_ids: Sequence[int] | None = None,
    verbose: bool = True,
) -> BundleResult:
    """Assemble and solve a bundle adjustment with optional extra constraints.

    Every constraint is optional, so the same call covers a plain bundle,
    marker-controlled georeferencing, GNSS-constrained adjustment, or any
    combination.

    Args:
        reconstruction: Adjusted in place.
        solver: Solver and refinement settings. None uses `SolverOptions()`.
        markers: Marker observations and world coordinates. None adds neither.
        camera_priors: Prior camera centres. None leaves the cameras free.
        filters: Post-solve filtering. None skips it.
        tie_points_sigma_px: Assumed 1-sigma of the tie-point observations,
            pixels. All other sigmas are expressed relative to this.
        image_ids: Restrict the adjustment to these images.
        verbose: Print the weighting summary, block counts and progress.

    Returns:
        A `BundleResult` with the adjusted marker coordinates and diagnostics.

    Raises:
        AssertionError: If the assembled problem does not contain exactly the
            expected number of residual blocks. This catches the classic
            failure of building a second adjuster, which silently orphans every
            block added here and leaves the solve running on tie points alone.
    """
    solver = solver or SolverOptions()
    options = build_options(solver)
    config = build_config(
        reconstruction,
        constant_intrinsics=not solver.refine_focal_length
        and not solver.refine_principal_point
        and not solver.refine_extra_params,
        image_ids=image_ids,
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
        if camera_priors is not None:
            print(
                f"camera priors   : sigma {camera_priors.sigma_m} m -> "
                f"{np.asarray(camera_priors.sigma_m) / tie_points_sigma_px} "
                "in tie units"
            )

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
        "camera": 0,
    }

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

    if camera_priors is not None:
        n_blocks["camera"] = add_camera_priors(
            problem, reconstruction, camera_priors, tie_points_sigma_px, retained
        )

    expected = sum(n_blocks.values())
    assert problem.num_residual_blocks() == expected, (
        f"{problem.num_residual_blocks()} residual blocks, expected {expected} - "
        "the added constraints are not in the problem being solved"
    )
    if verbose:
        extra = expected - n_blocks["tie_points"]
        print(
            f"  problem: {expected:,} blocks = {n_blocks['tie_points']:,} tie points"
            f" + {extra} added "
            f"({n_blocks['marker_image']} marker image, "
            f"{n_blocks['marker_world']} marker world, "
            f"{n_blocks['camera']} camera)"
        )

    summary, converged = solve_to_convergence(
        bundle_adjuster, solver.max_solve_restarts, verbose
    )

    n_filtered = 0
    if filters is not None:
        n_filtered = filter_reconstruction(reconstruction, filters, verbose)

    return BundleResult(
        reconstruction=reconstruction,
        marker_xyz=marker_xyz,
        summary=summary,
        converged=converged,
        n_blocks=n_blocks,
        n_filtered=n_filtered,
    )
