"""Frontend: thin wrappers around native pycolmap calls.

Only the stages whose *option plumbing* is tedious live here - features and
matching, seeding known intrinsics, writing pose priors, incremental mapping.
Everything else is a direct pycolmap call and should stay that way: this module
is deliberately small, and `pycolmap` remains the API of record.

The georeferenced bundle that comes after mapping lives in
`pycolmap_ba.core`, the scoring and export in `pycolmap_ba.utils`.

Examples:
    Free network with known intrinsics held fixed::

        frontend.extract_and_match(db_path, image_path, overwrite=True)
        frontend.seed_intrinsics(
            db_path,
            CameraIntrinsics(width=6016, height=4016, params=[5865.18, 3008, 2008]),
        )
        rec = frontend.run_mapping(db_path, image_path, sparse_path)

    Constrain the mapper with GNSS camera positions::

        frontend.write_pose_priors(db_path, camera_xyz, sigma_m=0.10)
        rec = frontend.run_mapping(
            db_path, image_path, sparse_path,
            MappingOptions(use_prior_position=True),
        )
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from .utils.report import banner

__all__ = [
    "CameraIntrinsics",
    "FeatureOptions",
    "MappingOptions",
    "extract_and_match",
    "remove_database",
    "run_mapping",
    "seed_intrinsics",
    "write_pose_priors",
]


@dataclass
class FeatureOptions:
    """Feature extraction and matching settings.

    Attributes:
        extractor: A `pycolmap.FeatureExtractorType`. None uses SIFT.
        matcher: A `pycolmap.FeatureMatcherType`. None uses SIFT_BRUTEFORCE,
            which is exhaustive and deterministic. Pair ALIKED_N16ROT with
            ALIKED_LIGHTGLUE.
        camera_model: COLMAP camera model for newly created cameras. Prefer a
            single-focal model on a flat nadir block: with PINHOLE, fx and fy
            refine independently and can diverge into an anisotropic stretch,
            which is not a similarity, so no later alignment can undo it.
        camera_mode: A `pycolmap.CameraMode`. None shares one camera across all
            images (SINGLE).
        max_image_size: Downscale limit for extraction; -1 = full resolution.
            COLMAP rescales keypoints back to original image coordinates, so
            marks measured at full resolution stay compatible either way.
        max_num_features: Feature cap per image, applied to whichever extractor
            was selected.
        use_gpu: Use the GPU for both extraction and matching.
    """

    extractor: Any = None  # None -> pycolmap.FeatureExtractorType.SIFT
    matcher: Any = None  # None -> pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    camera_model: str = "SIMPLE_PINHOLE"
    camera_mode: Any = None  # None -> pycolmap.CameraMode.SINGLE
    max_image_size: int = -1
    max_num_features: int = 8192
    use_gpu: bool = False


@dataclass
class CameraIntrinsics:
    """Known intrinsics to seed into the database before mapping.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        params: Model parameters, in COLMAP's order for `model`
            (SIMPLE_PINHOLE: `[f, cx, cy]`).
        model: Camera model name or `pycolmap.CameraModelId`.
        has_prior_focal_length: Marks the focal length as known. Without it the
            mapper may discard the seeded value as a bad EXIF guess.
    """

    width: int
    height: int
    params: Sequence[float]
    model: str | Any = "SIMPLE_PINHOLE"
    has_prior_focal_length: bool = True

    def model_id(self) -> Any:
        """Resolve `model` to a `pycolmap.CameraModelId`."""
        if isinstance(self.model, str):
            return getattr(pycolmap.CameraModelId, self.model)
        return self.model


@dataclass
class MappingOptions:
    """Incremental mapper settings.

    Attributes:
        use_prior_position: Let the mapper use the pose priors stored in the
            database (see `write_pose_priors`). COLMAP needs >= 3 priors or it
            silently ignores them.
        use_robust_loss_on_prior_position: Robustify the prior residuals.
        prior_position_loss_scale: Scale of that loss. The prior residuals are
            covariance-whitened, so the scale is in sigma units:
            `sqrt(7.815)` is the 95% quantile of chi-square with 3 DoF.
        refine_focal_length: Self-calibrate the focal during mapping. A flat
            nadir block cannot determine it from imagery - focal trades against
            scene depth - so keep it False and free it later in the controlled
            bundle, where the control breaks the degeneracy.
        refine_principal_point: Self-calibrate cx, cy during mapping.
        refine_extra_params: Self-calibrate distortion during mapping.
        filter_max_reproj_error: Drop observations above this error, pixels.
        filter_min_tri_angle_deg: Drop points with a weaker triangulation angle
            than this, degrees.
        ba_local_min_tri_angle_deg: Minimum triangulation angle for a point to
            enter local BA, degrees.
        extra: Any further attribute to set on the `IncrementalPipelineOptions`,
            as `{name: value}`. Dotted names reach into sub-options, e.g.
            `{"mapper.abs_pose_min_num_inliers": 20}`.

    The three filter thresholds are COLMAP's own defaults, restated so the tuning
    surface is visible rather than implicit; on a block with strong overlap they
    are typically no-ops.
    """

    use_prior_position: bool = False
    use_robust_loss_on_prior_position: bool = False
    prior_position_loss_scale: float = float(np.sqrt(7.815))
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    filter_max_reproj_error: float = 4.0
    filter_min_tri_angle_deg: float = 1.5
    ba_local_min_tri_angle_deg: float = 6.0
    extra: dict[str, Any] = field(default_factory=dict)


def remove_database(database_path: Path) -> bool:
    """Delete a COLMAP database and its SQLite side files.

    The `-wal` / `-shm` files hold committed data too, so leaving them behind can
    resurrect part of the old database.

    Args:
        database_path: Path to the `.db` file.

    Returns:
        True if a database was actually removed.
    """
    database_path = Path(database_path)
    if not database_path.exists():
        return False
    database_path.unlink()
    for suffix in (".db-wal", ".db-shm"):
        side = database_path.with_suffix(suffix)
        if side.exists():
            side.unlink()
    return True


def extract_and_match(
    database_path: Path,
    image_path: Path,
    options: FeatureOptions | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Extract features and match them exhaustively.

    Args:
        database_path: Database to create or extend. Parent dirs are created.
        image_path: Root folder of the images.
        options: Extraction/matching settings. None uses `FeatureOptions()`.
        overwrite: Delete an existing database first. Without this, COLMAP
            extends the existing one and skips images it already knows.
        verbose: Print a section header.
    """
    options = options or FeatureOptions()
    database_path, image_path = Path(database_path), Path(image_path)
    if verbose:
        banner("Feature extraction and matching")
    if overwrite and remove_database(database_path):
        print(f"  removed existing database {database_path.name}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = options.extractor or pycolmap.FeatureExtractorType.SIFT
    extraction_options = pycolmap.FeatureExtractionOptions(
        type=extractor,
        max_image_size=options.max_image_size,
        use_gpu=options.use_gpu,
    )
    # The feature cap lives on the extractor's own sub-options (`.sift`,
    # `.aliked`, ...), named after the first word of the enum member.
    family = extractor.name.split("_")[0].lower()
    getattr(extraction_options, family).max_num_features = options.max_num_features

    matching_options = pycolmap.FeatureMatchingOptions(
        type=options.matcher or pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE,
        use_gpu=options.use_gpu,
    )

    pycolmap.extract_features(
        database_path,
        image_path,
        camera_mode=options.camera_mode or pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model=options.camera_model),
        extraction_options=extraction_options,
    )
    pycolmap.match_exhaustive(database_path, matching_options=matching_options)


def seed_intrinsics(
    database_path: Path,
    camera: CameraIntrinsics,
    verbose: bool = True,
) -> int:
    """Overwrite every camera in the database with known intrinsics.

    Args:
        database_path: Database written in place.
        camera: The intrinsics to write.
        verbose: Print the resulting cameras.

    Returns:
        Number of cameras updated.
    """
    if verbose:
        banner("Seeding the known intrinsics")
    n_updated = 0
    with pycolmap.Database.open(Path(database_path)) as db:
        for entry in db.read_all_cameras():
            entry.model = camera.model_id()
            entry.width, entry.height = camera.width, camera.height
            entry.params = np.asarray(camera.params, dtype=float)
            entry.has_prior_focal_length = camera.has_prior_focal_length
            db.update_camera(entry)
            n_updated += 1
            if verbose:
                print(f"  camera {entry.camera_id}: {entry.model.name} {entry.params}")
    return n_updated


def write_pose_priors(
    database_path: Path,
    xyz: dict[str, np.ndarray],
    sigma_m: float | Sequence[float] | np.ndarray = 0.10,
    verbose: bool = True,
) -> int:
    """Store prior camera positions (e.g. GNSS) in the database.

    These are what `MappingOptions.use_prior_position` consumes; they also fix
    the gauge, so the free network comes out roughly in world coordinates.

    Args:
        database_path: Database written in place.
        xyz: Prior camera **centres** per image name, world coordinates, metres.
        sigma_m: 1-sigma accuracy, metres. Scalar or (x, y, z).
        verbose: Print how many priors were written.

    Returns:
        Number of priors written.
    """
    if verbose:
        banner("Seeding prior camera positions")
    sigma = np.broadcast_to(np.asarray(sigma_m, dtype=float), (3,))
    n_written = 0
    with pycolmap.Database.open(Path(database_path)) as db:
        # Priors persist in the database across sessions, and writing a second
        # one for the same image violates a UNIQUE constraint - so reuse the
        # existing id and update in place instead.
        existing = {p.corr_data_id: p.pose_prior_id for p in db.read_all_pose_priors()}
        for image in db.read_all_images():
            if image.name not in xyz:
                continue
            prior = pycolmap.PosePrior()
            prior.corr_data_id = image.data_id
            prior.position = np.asarray(xyz[image.name], dtype=float)
            prior.position_covariance = np.diag(sigma**2)
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            if image.data_id in existing:
                prior.pose_prior_id = existing[image.data_id]
                db.update_pose_prior(prior)
            else:
                db.write_pose_prior(prior)
            n_written += 1
        n_priors = db.num_pose_priors()

    if verbose:
        print(f"  wrote {n_written} pose priors, sigma {sigma} m")
    # Below three priors COLMAP cannot fit a similarity, so it drops the priors
    # and fixes the gauge on the first pair instead - silently.
    if n_priors < 3:
        print(
            f"  WARNING: only {n_priors} pose priors in the database; COLMAP "
            "needs >= 3 or it ignores them entirely"
        )
    return n_written


def _plain_mapping(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    options: pycolmap.IncrementalPipelineOptions,
) -> dict[int, pycolmap.Reconstruction]:
    """Run `incremental_mapping` with no progress bar."""
    return pycolmap.incremental_mapping(
        database_path, image_path, output_path, options=options
    )


def _mapping_with_progress(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    options: pycolmap.IncrementalPipelineOptions,
) -> dict[int, pycolmap.Reconstruction]:
    """Run `incremental_mapping` behind a registration progress bar.

    Falls back to a plain run when `enlighten` is not installed. Adapted from
    `pycolmap_examples/example.py`.
    """
    try:
        import enlighten
    except ImportError:
        return pycolmap.incremental_mapping(
            database_path, image_path, output_path, options=options
        )

    with pycolmap.Database.open(database_path) as db:
        num_images = db.num_images()
    with enlighten.Manager() as manager:
        with manager.counter(total=num_images, desc="Images registered:") as pbar:
            pbar.update(0, force=True)
            return pycolmap.incremental_mapping(
                database_path,
                image_path,
                output_path,
                options=options,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )


def run_mapping(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    options: MappingOptions | None = None,
    clean: bool = True,
    write_text: bool = False,
    seed: int | None = None,
    progress: bool = False,
    verbose: bool = True,
) -> pycolmap.Reconstruction:
    """Run incremental mapping and return the largest reconstruction.

    Args:
        database_path: Database with features and matches.
        image_path: Root folder of the images.
        output_path: Folder for the sub-models COLMAP writes (`0/`, `1/`, ...).
        options: Mapper settings. None uses `MappingOptions()`.
        clean: Delete `output_path` first, so a previous - possibly distorted -
            reconstruction is never picked up and reused.
        write_text: Also write the winning reconstruction as text format at the top of `output_path`.
        seed: Seed COLMAP's RNG for a reproducible run.
        progress: Show a registration progress bar (needs `enlighten`).
        verbose: Print a header and the resulting summary.

    Returns:
        The reconstruction with the most registered images.

    Raises:
        RuntimeError: If the mapper produced no reconstruction at all.
    """
    options = options or MappingOptions()
    output_path = Path(output_path)
    if verbose:
        banner("Incremental mapping")
    if seed is not None:
        pycolmap.set_random_seed(seed)

    pipeline = pycolmap.IncrementalPipelineOptions()
    pipeline.use_prior_position = options.use_prior_position
    pipeline.use_robust_loss_on_prior_position = (
        options.use_robust_loss_on_prior_position
    )
    pipeline.prior_position_loss_scale = options.prior_position_loss_scale
    pipeline.ba_refine_focal_length = options.refine_focal_length
    pipeline.ba_refine_principal_point = options.refine_principal_point
    pipeline.ba_refine_extra_params = options.refine_extra_params
    pipeline.mapper.filter_max_reproj_error = options.filter_max_reproj_error
    pipeline.mapper.filter_min_tri_angle = options.filter_min_tri_angle_deg
    pipeline.mapper.ba_local_min_tri_angle = options.ba_local_min_tri_angle_deg
    for name, value in options.extra.items():
        target = pipeline
        *parents, leaf = name.split(".")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)

    if clean and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    run = _mapping_with_progress if progress else _plain_mapping
    reconstructions = run(Path(database_path), Path(image_path), output_path, pipeline)
    if not reconstructions:
        raise RuntimeError(
            "incremental mapping produced no reconstruction - check the matches "
            "and, if pose priors are enabled, that there are >= 3"
        )
    best = reconstructions[
        max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    ]
    if write_text:
        # best.write(output_path)
        best.write_text(output_path)
    if verbose:
        if len(reconstructions) > 1:
            sizes = [r.num_reg_images() for r in reconstructions.values()]
            print(f"  {len(reconstructions)} sub-models, sizes {sizes} -> keeping best")
        print(best.summary())
        print("  camera:", next(iter(best.cameras.values())).params)
    return best
