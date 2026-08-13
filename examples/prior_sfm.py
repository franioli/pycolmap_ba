"""Prior-informed SfM from a COLMAP-format ground truth.

Reads the ground-truth model in `GT_PATH`, injects its exact intrinsics into a
fresh database, writes camera-centre pose priors with an explicit covariance,
and runs a prior-informed incremental reconstruction.

Optionally follows up with a bundle adjustment that also constrains the
rotations, which COLMAP's native pose priors cannot do.

Run with:
    uv run python examples/prior_sfm.py

The paths below point at a synthetic block that is not part of this repository -
set `GT_PATH` / `IMAGE_PATH` to your own COLMAP model and images before running.
"""

from pathlib import Path

import numpy as np
import pycolmap

from pycolmap_ba.backend import (
    CameraPoseConstraints,
    SolverOptions,
    run_bundle_adjustment,
)
from pycolmap_ba.frontend import (
    FeatureOptions,
    MappingOptions,
    extract_and_match,
    run_mapping,
    write_pose_priors,
)
from pycolmap_ba.utils import banner, camera_center_rmse, rotation_error_deg

ROOT = Path(__file__).resolve().parents[1]
GT_PATH = ROOT / "colmap/sparse/0"
IMAGE_PATH = ROOT / "colmap/data"  # needs >= 3 images for priors to be used
WORK = ROOT / "colmap/prior_run"
DB_PATH = WORK / "database.db"
OUT_PATH = WORK / "sparse"

POSITION_STDDEV_M = 0.10  # 1-sigma on the ground-truth camera centres
ROTATION_STDDEV_DEG = 0.05  # only used by the optional 6-DoF stage
REFINE_ROTATIONS_WITH_PRIORS = True


def copy_intrinsics(gt: pycolmap.Reconstruction, gt_images: dict) -> None:
    """Overwrite each database camera with its ground-truth intrinsics.

    Per-image rather than shared, so `frontend.seed_intrinsics` (one camera for
    the whole block) does not apply - this is the plain pycolmap route.

    Args:
        gt: The ground-truth reconstruction holding the true cameras.
        gt_images: Ground-truth images keyed by image name.
    """
    with pycolmap.Database.open(DB_PATH) as db:
        for image in db.read_all_images():
            gt_camera = gt.cameras[gt_images[image.name].camera_id]
            camera = db.read_camera(image.camera_id)
            camera.model = gt_camera.model
            camera.width, camera.height = gt_camera.width, gt_camera.height
            camera.params = gt_camera.params
            # Without this the mapper may discard them as bad EXIF guesses.
            camera.has_prior_focal_length = True
            db.update_camera(camera)


def report(rec: pycolmap.Reconstruction, gt_images: dict, label: str) -> None:
    """Print position, rotation and reprojection errors against the truth.

    Args:
        rec: The reconstruction to score.
        gt_images: Ground-truth images keyed by image name.
        label: Row label.
    """
    centres = {name: image.projection_center() for name, image in gt_images.items()}
    angles = [
        rotation_error_deg(
            image.cam_from_world().rotation.matrix(),
            gt_images[image.name].cam_from_world().rotation.matrix(),
        )
        * 3600
        for image in rec.images.values()
    ]
    print(
        f"{label:24s} pos RMSE = {camera_center_rmse(rec, centres) * 100:6.3f} cm   "
        f"rot RMSE = {np.sqrt(np.mean(np.square(angles))):7.2f} arcsec   "
        f"reproj = {rec.compute_mean_reprojection_error():.4f} px"
    )


def main() -> None:
    """Run the whole prior-informed pipeline."""
    WORK.mkdir(parents=True, exist_ok=True)

    gt = pycolmap.Reconstruction(GT_PATH)  # auto-detects .txt vs .bin
    gt_images = {im.name: im for im in gt.images.values()}
    print(f"ground truth: {gt.num_cameras()} cameras, {gt.num_images()} images")

    # ALIKED + LightGlue on CPU.
    extract_and_match(
        DB_PATH,
        IMAGE_PATH,
        FeatureOptions(
            extractor=pycolmap.FeatureExtractorType.ALIKED_N16ROT,
            matcher=pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE,
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            max_image_size=2000,
            max_num_features=4096,
        ),
        overwrite=True,
    )
    copy_intrinsics(gt, gt_images)
    write_pose_priors(
        DB_PATH,
        {name: image.projection_center() for name, image in gt_images.items()},
        sigma_m=POSITION_STDDEV_M,
    )

    rec = run_mapping(
        DB_PATH,
        IMAGE_PATH,
        OUT_PATH,
        MappingOptions(
            use_prior_position=True,
            use_robust_loss_on_prior_position=True,
        ),
    )
    banner("Accuracy against the ground truth")
    report(rec, gt_images, "after mapper")

    if REFINE_ROTATIONS_WITH_PRIORS:
        # 6-DoF priors: positions *and* rotations. The sigmas are in native units
        # against COLMAP's unweighted tie points, hence tie_points_sigma_px=1.
        run_bundle_adjustment(
            rec,
            solver=SolverOptions(refine_focal_length=False),
            camera_poses=CameraPoseConstraints(
                positions={
                    name: image.projection_center() for name, image in gt_images.items()
                },
                sigma_m=POSITION_STDDEV_M,
                cam_from_world={
                    name: image.cam_from_world() for name, image in gt_images.items()
                },
                sigma_rot_deg=ROTATION_STDDEV_DEG,
            ),
            tie_points_sigma_px=1.0,
        )
        report(rec, gt_images, "after 6-DoF prior BA")


if __name__ == "__main__":
    main()
