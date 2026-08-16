"""Covariance, both directions of the same tie-point-sigma-unit convention.

**Prior** (S1): every constraint this package adds - marker marks, marker
world priors, camera pose priors, reference-cloud pulls - is whitened into
`tie_points_sigma_px` units before it can share a Ceres problem with COLMAP's
own unweighted tie-point residuals (`docs/gcp_bundle_adjustment.md` S4).
`cov_in_tie_units`/`covs_in_tie_units` do that whitening; every `add_*`
function in `gcp`, `camera_pose`, `ref_cloud` calls one of them on the way in.

**Posterior** (S2-3): `pycolmap.estimate_ba_covariance_from_problem` returns a
**cofactor** matrix `Q = (J^T J)^-1` - correct only if those assumed sigmas
were exactly right. The actual posterior covariance is
`Sigma_hat = sigma0_hat^2 * Q`, where `sigma0_hat^2` is the empirical variance
factor - the fit-quality check that tells you whether the assumed sigmas held
up (S2, `variance_factor`), used to scale the per-parameter extraction
functions in S3. See `docs/posterior_covariance.md` for the full derivation,
in particular why the same `sigma_tie` from S1 cancels exactly out of the S2
scale factor and no further unit conversion is needed.

The two directions are kept in one module, rather than splitting prior
weighting into its own file, because they are the same convention read
forwards and backwards - a change to one has to be checked against the other.
Lives in `core`, not `utils`, because every S2-3 function takes or is a method
of `BundleResult` - as tightly coupled to the solve as `filtering` is, just
running after it rather than during it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pycolmap

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from .bundle import BundleResult, SolverOptions

__all__ = [
    "VarianceFactor",
    "camera_center_covariances",
    "camera_pose_covariances",
    "cov_in_tie_units",
    "covs_in_tie_units",
    "estimate_covariance",
    "marker_covariances",
    "point_covariances",
    "std_norm",
    "variance_factor",
]


# =============================================================================
# 1. Prior: weighting a constraint into tie-point-sigma units
# =============================================================================


def cov_in_tie_units(
    sigma: float | Sequence[float] | np.ndarray, tie_points_sigma_px: float, size: int
) -> np.ndarray:
    """Build a diagonal covariance for `sigma`, in tie-point sigma units.

    For constraints specified as independent per-axis sigmas - marker world
    coordinates, camera positions, image measurements.

    Args:
        sigma: Scalar or per-axis 1-sigma, in native units (px or m).
        tie_points_sigma_px: Reference tie-point sigma, pixels.
        size: Dimension of the residual (2 for image, 3 for world).

    Returns:
        A `size` x `size` diagonal covariance matrix.
    """
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (size,))
    return np.diag((s / tie_points_sigma_px) ** 2)


def covs_in_tie_units(cov: np.ndarray, tie_points_sigma_px: float) -> np.ndarray:
    """Rescale full covariance matrices into tie-point sigma units.

    The matrix counterpart of `cov_in_tie_units`, for constraints whose
    covariance is not diagonal - a reference-cloud GICP covariance, say. Scaling
    a covariance by `1 / sigma_tie^2` is the same operation as dividing its
    sigmas by `sigma_tie`.

    Args:
        cov: `(..., n, n)` covariance matrices in native units.
        tie_points_sigma_px: Reference tie-point sigma, pixels.

    Returns:
        The same array scaled by `1 / tie_points_sigma_px^2`.
    """
    return np.asarray(cov, dtype=float) / tie_points_sigma_px**2


# =============================================================================
# 2. Posterior: the a-posteriori variance factor sigma-hat-0
# =============================================================================


def estimate_covariance(
    reconstruction: pycolmap.Reconstruction,
    bundle_adjuster: Any,
    options: Any = None,
) -> Any:
    """Estimate posterior covariances for the adjusted parameters.

    Thin wrapper on `pycolmap.estimate_ba_covariance_from_problem`, which needs
    a problem whose structure suits the Schur complement trick. The marker
    parameter blocks added by `add_marker_constraints` are extra 3-vectors that
    are *not* registered as COLMAP points, so the estimation can legitimately
    fail here - hence the None return rather than an exception.

    The matrix this returns is an unscaled cofactor matrix, not a covariance -
    see the module docstring and `variance_factor`.

    Args:
        reconstruction: The adjusted reconstruction.
        bundle_adjuster: The adjuster that was solved.
        options: A `pycolmap.BACovarianceOptions`. None uses the defaults.

    Returns:
        A `pycolmap.BACovariance`, or None if the estimation failed.
    """
    options = options or pycolmap.BACovarianceOptions()
    return pycolmap.estimate_ba_covariance_from_problem(
        options, reconstruction, bundle_adjuster.problem
    )


def std_norm(cov: np.ndarray | None) -> float:
    """`sqrt(trace(cov))` = `||(sigma_x, sigma_y, sigma_z)||`, NaN-safe.

    The single-number summary used throughout this package (`utils.io`,
    `utils.report`, `filter_points_by_covariance`) - matches the marginal
    standard deviations Metashape reports as its per-point "Error".

    Args:
        cov: A square covariance matrix, or None.

    Returns:
        `sqrt(trace(cov))`, or NaN if `cov` is None or not square.
    """
    if cov is None:
        return float("nan")
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return float("nan")
    return float(np.sqrt(np.clip(np.trace(arr), 0, None)))


def _skew(v: np.ndarray) -> np.ndarray:
    """The 3x3 matrix `[v]_x` such that `[v]_x @ w == cross(v, w)`."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _free_camera_param_count(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult | None,
    solver: SolverOptions | None,
) -> int:
    """Number of intrinsics parameters actually left free.

    Prefers `result.options` - the `pycolmap.BundleAdjustmentOptions` that was
    actually solved with - over `solver`, since `SolverOptions.extra` can
    override a refine_* flag after `solver` was built. Falling back to neither
    (no result and no solver given) assumes "every parameter free", the same
    assumption the old, buggy redundancy formula made unconditionally - kept
    here only as the least-wrong default when the caller cannot say, and
    flagged by `VarianceFactor.assumed_intrinsics_free`.

    Args:
        reconstruction: Source of the cameras.
        result: The solved `BundleResult`, if available.
        solver: The settings the bundle was run with, if `result` isn't.

    Returns:
        Total free intrinsics parameters, summed over all cameras.
    """
    result_options = result.options if result is not None else None
    source = result_options if result_options is not None else solver
    if source is None:
        return sum(len(camera.params) for camera in reconstruction.cameras.values())
    if not (
        source.refine_focal_length
        or source.refine_principal_point
        or source.refine_extra_params
    ):
        return 0
    total = 0
    for camera in reconstruction.cameras.values():
        if source.refine_focal_length:
            total += len(list(camera.focal_length_idxs()))
        if source.refine_principal_point:
            total += len(list(camera.principal_point_idxs()))
        if source.refine_extra_params:
            total += len(list(camera.extra_params_idxs()))
    return total


def _tie_point_residual_sum_sq(
    reconstruction: pycolmap.Reconstruction, min_track_length: int = 0
) -> tuple[float, int, int]:
    """Exact sum of squared tie-point image residuals, by direct reprojection.

    COLMAP never hands back the tie-point residual-block ids (they are added
    and owned entirely inside `create_default_ceres_bundle_adjuster`), so the
    only exact route to their residuals is reprojecting every observation by
    hand. This replaces the previous `sum(track_length * Point3D.error^2)`
    approximation, which is biased low: `Point3D.error` is COLMAP's **mean**
    per-observation reprojection error, not the RMS, and for 2-D isotropic
    Gaussian noise `|r|` is Rayleigh-distributed, so
    `(E|r|)^2 / E|r|^2 = (pi/2) / 2 = pi/4 ~= 0.785` - i.e. the mean-based sum
    understates the true sum of squares by exactly that Jensen-gap factor.
    `sandbox/examples/check_covariance.py` checks the ratio against `4/pi`
    directly on a synthetic block.

    Args:
        reconstruction: The adjusted block.
        min_track_length: Skip observations of points below this track length -
            matches `SolverOptions.min_track_length`, which excludes them from
            the Ceres problem entirely (tested on the *full* track length, so
            reproducible exactly from Python).

    Returns:
        `(sum_sq, n_observations, n_behind_camera)`, pixels^2. Points behind
        the camera (`img_from_cam`'s cheirality check) are excluded from both
        the sum and the count, as they are not in the solved problem either.
    """
    total_sq = 0.0
    n_obs = 0
    n_behind = 0
    points3D = reconstruction.points3D
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        observations = image.get_observation_points2D()
        if len(observations) == 0:
            continue
        ids = [p.point3D_id for p in observations]
        keep = [
            pid in points3D and points3D[pid].track.length() >= min_track_length
            for pid in ids
        ]
        if not any(keep):
            continue
        xyz = np.array(
            [points3D[pid].xyz for pid, k in zip(ids, keep, strict=True) if k],
            dtype=float,
        )
        xy = np.array(
            [p.xy for p, k in zip(observations, keep, strict=True) if k], dtype=float
        )
        camera = reconstruction.cameras[image.camera_id]
        cam_pts = image.cam_from_world() * xyz
        uv = camera.img_from_cam(cam_pts, True)
        behind = np.isnan(uv).any(axis=1)
        n_behind += int(behind.sum())
        residual = uv[~behind] - xy[~behind]
        total_sq += float(np.sum(residual**2))
        n_obs += int((~behind).sum())
    return total_sq, n_obs, n_behind


@dataclass(frozen=True)
class VarianceFactor:
    """The a-posteriori variance factor sigma-hat-0 of a solved block.

    In a weighted least-squares (Gauss-Markov) adjustment with weight matrix
    `P = Sigma_y^-1` (assumed sigmas taken as exact),

        Q_xx = (A^T P A)^-1                    cofactor matrix
        sigma0_hat^2 = (v^T P v) / r,  r = n - u     empirical variance factor
        Sigma_hat_xx = sigma0_hat^2 * Q_xx           posterior covariance

    `sigma0_hat` near `tie_points_sigma_px` says the assumed sigmas were about
    right; well above says they were too tight (or the model does not fit);
    well below says too loose. `ratio = sigma0 / tie_points_sigma_px` is the
    dimensionless number to compare across runs and against e.g. another
    package's own a-posteriori sigma0. Förstner & Wrobel, *Photogrammetric
    Computer Vision* (Springer, 2016), §2.4/§4.2; Koch, *Parameter Estimation
    and Hypothesis Testing in Linear Models*, 2nd ed. (Springer, 1999), ch. 3.

    Attributes:
        sigma0: Overall variance factor, pixels (tie-point units). NaN if
            `redundancy <= 0`.
        ratio: `sigma0 / tie_points_sigma_px`. 1.0 means the assumed sigmas
            were right; this, not `sigma0` itself, is the number comparable to
            another tool's "a-posteriori sigma0" printed against a unit prior.
        redundancy: `n_equations - n_unknowns + n_datum`.
        n_equations: Observation equations (tie points + every added
            constraint group).
        n_unknowns: Free parameters (poses, points, free intrinsics, marker
            coordinates).
        n_datum: 7 if a gauge was fixed, else 0.
        residual_sum_sq: `sum(whitened residuals^2)` over every group, px^2
            (or the group's native squared units for the added constraints,
            already whitened into tie-point units per §1 above).
        tie_points_sigma_px: The reference sigma this was computed against.
        n_observations: Tie-point observations actually in the problem (after
            `min_track_length` and cheirality exclusions).
        n_points: 3D points actually contributing unknowns to the problem.
        n_behind: Observations excluded because the point reprojects behind
            the camera - diagnostic only, already excluded from the above.
        assumed_intrinsics_free: True when neither `result.options` nor
            `solver` was given, so every intrinsics parameter had to be
            assumed free.
    """

    sigma0: float
    ratio: float
    redundancy: int
    n_equations: int
    n_unknowns: int
    n_datum: int
    residual_sum_sq: float
    tie_points_sigma_px: float
    n_observations: int
    n_points: int
    n_behind: int
    assumed_intrinsics_free: bool

    def summary(self) -> str:
        """The two-line block `block_report` prints: redundancy and sigma0_hat."""
        datum = f" + {self.n_datum} datum" if self.n_datum else ""
        caveat = (
            "  (no solver/result given - assumed all intrinsics free)"
            if self.assumed_intrinsics_free
            else ""
        )
        return (
            f"\n  redundancy        {self.n_equations:,} equations - "
            f"{self.n_unknowns:,} unknowns{datum}  ->  r = {self.redundancy:,}"
            f"{caveat}"
            f"\n  sigma0_hat        {self.sigma0:.3f} px  "
            f"(tie_points_sigma_px={self.tie_points_sigma_px:.2f} px, "
            f"ratio {self.ratio:.3f} - 1.0 = assumed sigmas about right)"
        )


def variance_factor(
    reconstruction: pycolmap.Reconstruction,
    result: BundleResult | None = None,
    *,
    solver: SolverOptions | None = None,
    tie_points_sigma_px: float = 0.5,
) -> VarianceFactor:
    """Compute the a-posteriori variance factor sigma-hat-0 of a solved block.

    Args:
        reconstruction: The adjusted block.
        result: The solved `BundleResult`, if available - supplies the added
            constraint-equation counts, the marker unknowns, whether the gauge
            was fixed, and (via `result.options`) which intrinsics were free.
        solver: The solver settings used, if `result.options` isn't available -
            decides how many intrinsics parameters were actually free.
        tie_points_sigma_px: The sigma the bundle was assembled with (see §1
            above); every other constraint's residual is already whitened
            into these units, so no further conversion is needed.

    Returns:
        The `VarianceFactor`.
    """
    points = list(reconstruction.points3D.values())
    min_track_length = 0
    options = result.options if result is not None else None
    source = options if options is not None else solver
    if source is not None:
        min_track_length = source.min_track_length

    residual_sum_sq, n_obs, n_behind = _tie_point_residual_sum_sq(
        reconstruction, min_track_length
    )
    if result is not None:
        for group in ("marker_image", "marker_world", "camera_pose", "reference_cloud"):
            r = result.residuals(group)
            if r.size:
                residual_sum_sq += float((r**2).sum())

    n_points = sum(1 for p in points if p.track.length() >= min_track_length)
    n_img = reconstruction.num_reg_images()
    n_blocks = result.n_blocks if result is not None else {}
    n_marker_xyz = len(result.marker_xyz) if result is not None else 0
    n_eq = (
        2 * n_obs
        + 2 * n_blocks.get("marker_image", 0)
        + 3
        * (
            n_blocks.get("marker_world", 0)
            + n_blocks.get("camera_pose", 0)
            + n_blocks.get("reference_cloud", 0)
        )
    )
    n_free_camera = _free_camera_param_count(reconstruction, result, solver)
    n_unk = 6 * n_img + 3 * n_points + n_free_camera + 3 * n_marker_xyz
    n_datum = 7 if (result is not None and result.gauge is not None) else 0
    redundancy = n_eq - n_unk + n_datum

    sigma0 = (
        float(np.sqrt(residual_sum_sq / redundancy)) if redundancy > 0 else float("nan")
    )
    ratio = sigma0 / tie_points_sigma_px if redundancy > 0 else float("nan")
    assumed_intrinsics_free = source is None

    return VarianceFactor(
        sigma0=sigma0,
        ratio=ratio,
        redundancy=redundancy,
        n_equations=n_eq,
        n_unknowns=n_unk,
        n_datum=n_datum,
        residual_sum_sq=residual_sum_sq,
        tie_points_sigma_px=tie_points_sigma_px,
        n_observations=n_obs,
        n_points=n_points,
        n_behind=n_behind,
        assumed_intrinsics_free=assumed_intrinsics_free,
    )


# =============================================================================
# 3. Posterior: per-parameter covariance extraction, scaled by sigma-hat-0
# =============================================================================


def point_covariances(
    reconstruction: pycolmap.Reconstruction,
    covariance: pycolmap.BACovariance | None,
    sigma0: float,
    *,
    scale_by_sigma0: bool = True,
    point3D_ids: Iterable[int] | None = None,
) -> dict[int, np.ndarray | None]:
    """3x3 posterior covariance of each 3D point, world X/Y/Z, metres^2.

    **Conditional** on every pose, intrinsic and other point being held fixed
    at its optimum (COLMAP's Schur elimination reduces the point block that
    way) - not the marginal uncertainty, and not directly comparable to
    `marker_covariances`, which *is* marginal. This makes point sigmas
    systematically optimistic as a datum accuracy figure, but it is arguably
    the right quantity for a geometry-based filter (see
    `utils.filtering.filter_points_by_covariance`): it measures how well the
    rays intersect, which is exactly what a weak-geometry filter wants.

    Args:
        reconstruction: Source of the points (used only when `point3D_ids` is
            None, to iterate every point in the block).
        covariance: A `pycolmap.BACovariance`, or None (nothing was estimated).
        sigma0: The a-posteriori sigma to scale by - see `variance_factor`.
        scale_by_sigma0: Multiply the raw cofactor matrix by `sigma0^2`. True
            unless you specifically want the raw, unscaled cofactor.
        point3D_ids: Restrict to these ids. None uses every point in
            `reconstruction.points3D`.

    Returns:
        Covariance per point3D id; `{}` if `covariance` is None. A point held
        constant or excluded from the problem (e.g. by `min_track_length`)
        maps to None - kept, not omitted, so callers can tell "no information"
        from "not asked for".
    """
    if covariance is None:
        return {}
    ids = point3D_ids if point3D_ids is not None else reconstruction.points3D.keys()
    scale = sigma0**2 if scale_by_sigma0 else 1.0
    out: dict[int, np.ndarray | None] = {}
    for pid in ids:
        cov = covariance.get_point_cov(pid)
        out[pid] = None if cov is None else np.asarray(cov, dtype=float) * scale
    return out


def camera_pose_covariances(
    reconstruction: pycolmap.Reconstruction,
    covariance: pycolmap.BACovariance | None,
    sigma0: float,
    *,
    scale_by_sigma0: bool = True,
) -> dict[int, np.ndarray | None]:
    """6x6 tangent-space pose covariance `[rotation, translation]` per image.

    The rotation tangent is COLMAP's **left**, **half-angle** perturbation
    (`bundle_adjustment_ceres.cc`: `R' = exp([2*omega]_x) . R`) - an angular
    1-sigma is `2x` the tangent 1-sigma. See `docs/posterior_covariance.md` §3
    for the derivation and the numerical evidence for the factor of 2. Kept
    public because it is the only way to get rotation uncertainty; most
    callers want `camera_center_covariances` instead.

    Args:
        reconstruction: Source of the registered images.
        covariance: A `pycolmap.BACovariance`, or None.
        sigma0: The a-posteriori sigma to scale by - see `variance_factor`.
        scale_by_sigma0: Multiply by `sigma0^2`. True unless you want the raw
            cofactor.

    Returns:
        Covariance per registered image id; `{}` if `covariance` is None. None
        for an image whose pose block is not a full free 6x6 (the gauge-fixed
        camera under `TWO_CAMS_FROM_WORLD` is constant - `None` from pycolmap
        itself - or `SubsetManifold`d to 5x5 - detected and skipped here since
        the missing dimension cannot be attributed to rotation or translation
        without knowing which one pycolmap fixed).
    """
    if covariance is None:
        return {}
    scale = sigma0**2 if scale_by_sigma0 else 1.0
    out: dict[int, np.ndarray | None] = {}
    for image_id in reconstruction.reg_image_ids():
        cov = covariance.get_cam_cov_from_world(image_id)
        if cov is None or np.shape(cov) != (6, 6):
            out[image_id] = None
            continue
        out[image_id] = np.asarray(cov, dtype=float) * scale
    return out


def _camera_center_cov_from_pose_cov(
    rotation: np.ndarray, translation: np.ndarray, cov6: np.ndarray
) -> np.ndarray:
    """Propagate a `[rotation, translation]` tangent covariance to `C = -R^T t`.

    With `domega` the (left, half-angle) rotation tangent and `dt` the
    translation tangent, `R' = exp([2 domega]_x) R`, `t' = t + dt`, so to first
    order

        dC = -2 R^T [t]_x domega - R^T dt
        Sigma_C = R^T ( 4 [t]_x Sigma_ww [t]_x^T + 2 [t]_x Sigma_wt
                        + 2 Sigma_wt^T [t]_x^T + Sigma_tt ) R

    Verified by Monte Carlo against COLMAP's exact `Plus` (30k samples, ~1.5%
    agreement - MC noise floor at that sample size) in
    `sandbox/examples/check_covariance.py`. The naive `cov6[3:, 3:]` (kept only
    the translation block) drops the cross term, which here is the same order
    of magnitude as the translation term itself and not conservatively signed
    either way - not a safe approximation.

    Args:
        rotation: `cam_from_world.rotation.matrix()`, 3x3.
        translation: `cam_from_world.translation`, `(3,)`.
        cov6: The `[rotation, translation]` tangent covariance, 6x6.

    Returns:
        The 3x3 world-axis covariance of the camera centre, metres^2.
    """
    tx = _skew(translation)
    sigma_ww = cov6[:3, :3]
    sigma_wt = cov6[:3, 3:]
    sigma_tt = cov6[3:, 3:]
    sigma_c_cam = (
        4 * tx @ sigma_ww @ tx.T + 2 * tx @ sigma_wt + 2 * sigma_wt.T @ tx.T + sigma_tt
    )
    return rotation.T @ sigma_c_cam @ rotation


def camera_center_covariances(
    reconstruction: pycolmap.Reconstruction,
    covariance: pycolmap.BACovariance | None,
    sigma0: float,
    *,
    scale_by_sigma0: bool = True,
) -> dict[int, np.ndarray | None]:
    """3x3 world-axis posterior covariance of each camera centre, metres^2.

    Propagates the full 6x6 `camera_pose_covariances` block through
    `C = -R^T t`, including the rotation-translation cross term - see
    `_camera_center_cov_from_pose_cov`.

    Args:
        reconstruction: Source of the registered images and their poses.
        covariance: A `pycolmap.BACovariance`, or None.
        sigma0: The a-posteriori sigma to scale by - see `variance_factor`.
        scale_by_sigma0: Multiply by `sigma0^2`. True unless you want the raw
            cofactor.

    Returns:
        Covariance per registered image id; `{}` if `covariance` is None. None
        wherever `camera_pose_covariances` is (gauge-fixed or reduced pose
        block).
    """
    pose_covs = camera_pose_covariances(
        reconstruction, covariance, sigma0, scale_by_sigma0=scale_by_sigma0
    )
    out: dict[int, np.ndarray | None] = {}
    for image_id, cov6 in pose_covs.items():
        if cov6 is None:
            out[image_id] = None
            continue
        pose = reconstruction.images[image_id].cam_from_world()
        out[image_id] = _camera_center_cov_from_pose_cov(
            pose.rotation.matrix(), pose.translation, cov6
        )
    return out


def marker_covariances(
    marker_xyz: dict[str, np.ndarray],
    covariance: pycolmap.BACovariance | None,
    sigma0: float,
    *,
    scale_by_sigma0: bool = True,
) -> dict[str, np.ndarray | None]:
    """3x3 posterior covariance of each marker, world X/Y/Z, metres^2.

    **Marginal**, not conditional: markers are "other" parameters in COLMAP's
    reduced system, whose full inverse is taken - unlike `point_covariances`,
    which is conditional on the rest of the block. The two are not on the same
    statistical footing.

    Args:
        marker_xyz: The *live* parameter-block dict from `BundleResult` (e.g.
            `result.marker_xyz`) - `get_other_params_cov` looks the block up by
            pointer, so a copy or a re-cast array silently returns None instead
            of the marker's covariance.
        covariance: A `pycolmap.BACovariance`, or None.
        sigma0: The a-posteriori sigma to scale by - see `variance_factor`.
        scale_by_sigma0: Multiply by `sigma0^2`. True unless you want the raw
            cofactor.

    Returns:
        Covariance per marker name; `{}` if `covariance` is None.

    Raises:
        ValueError: A marker's array is not float64 / C-contiguous, so the
            pointer lookup would silently miss and return None instead of the
            real covariance.
    """
    if covariance is None:
        return {}
    scale = sigma0**2 if scale_by_sigma0 else 1.0
    out: dict[str, np.ndarray | None] = {}
    for name, xyz in marker_xyz.items():
        arr = np.asarray(xyz)
        if arr.dtype != np.float64 or not arr.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"marker {name!r} array is not float64/C-contiguous - "
                "get_other_params_cov does a pointer lookup and would silently "
                "return None instead of raising"
            )
        cov = covariance.get_other_params_cov(xyz)
        out[name] = None if cov is None else np.asarray(cov, dtype=float) * scale
    return out
