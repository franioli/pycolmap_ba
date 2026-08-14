"""Helpers: knock a reconstruction out of place, to test whether it recovers.

A recovery test is the only honest way to show that a constraint *works* rather
than merely leaves a good block alone. Break the block in a known way, adjust,
and see how much comes back.

The 7-DoF similarity is the important one. Applied to points **and** cameras
together it leaves every reprojection residual unchanged - it is exactly the
gauge freedom of a free network - so the tie points cannot possibly undo it and
whatever is recovered is attributable to the added constraint alone. That is not
true of the intrinsics: a focal error *is* visible to the tie points, just
weakly and degenerately on a flat block, so recovering it is a claim about
conditioning rather than about information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pycolmap

__all__ = ["Perturbation", "PerturbationReport", "perturb_reconstruction"]


@dataclass
class Perturbation:
    """How far to knock a reconstruction out of place.

    Every field defaults to zero, so a `Perturbation()` is a no-op and each test
    turns on only what it means to measure.

    Attributes:
        shift_m: Translate the block by this many metres, in a random direction.
        rot_deg: Rotate the block by this angle, degrees, about a random axis
            through its centroid.
        scale_ppm: Scale the block about its centroid, parts per million
            (1e6 ppm doubles it). 1000 ppm is 1 mm per metre.
        focal_pct: Scale every camera's focal length by this percentage.
        principal_point_px: Displace every principal point by this many pixels,
            in a random direction.
        extra_params_pct: Scale every camera's distortion coefficients by this
            percentage. Silently does nothing on a model without any, such as
            SIMPLE_PINHOLE.
        point_noise_m: Add isotropic Gaussian noise of this sigma to every 3D
            point, metres. Unlike the others this is *not* recoverable by any
            datum constraint - it destroys information rather than moving it -
            so use it to test robustness, not recovery.
        seed: Seed for every random direction, axis and noise draw.
    """

    shift_m: float = 0.0
    rot_deg: float = 0.0
    scale_ppm: float = 0.0
    focal_pct: float = 0.0
    principal_point_px: float = 0.0
    extra_params_pct: float = 0.0
    point_noise_m: float = 0.0
    seed: int = 0

    def __bool__(self) -> bool:
        """True when this would actually change anything."""
        return any(
            (
                self.shift_m,
                self.rot_deg,
                self.scale_ppm,
                self.focal_pct,
                self.principal_point_px,
                self.extra_params_pct,
                self.point_noise_m,
            )
        )


@dataclass
class PerturbationReport:
    """What `perturb_reconstruction` actually did.

    Attributes:
        sim3d: The applied similarity, or None if the datum was left alone.
            Its inverse is the transform a perfect recovery would find.
        lever_arm: RMS distance of the 3D points from their centroid, metres.
        reach: Displacement the datum perturbation causes at the block edge,
            metres. **Size the first correspondence gate from this**: a gate
            narrower than `reach` matches the wrong surface and the block can
            never pull itself back.
        intrinsics_before: Camera parameter vectors before, by camera id.
        intrinsics_after: The same after.
    """

    sim3d: pycolmap.Sim3d | None = None
    lever_arm: float = 0.0
    reach: float = 0.0
    intrinsics_before: dict[int, np.ndarray] = field(default_factory=dict)
    intrinsics_after: dict[int, np.ndarray] = field(default_factory=dict)

    def summary(self) -> str:
        """Multi-line recap of everything that was changed."""
        lines = ["perturbation applied"]
        if self.sim3d is not None:
            lines.append(
                f"  datum      : {np.linalg.norm(self.sim3d.translation):.3f} m "
                f"translation, scale {(self.sim3d.scale - 1) * 1e6:+.0f} ppm"
            )
            lines.append(
                f"  reach      : {self.reach:.3f} m at the block edge "
                f"(L = {self.lever_arm:.0f} m) <- first gate must exceed this"
            )
        for camera_id, before in self.intrinsics_before.items():
            after = self.intrinsics_after[camera_id]
            if not np.allclose(before, after):
                lines.append(
                    f"  camera {camera_id}   : {np.round(before, 3)} -> "
                    f"{np.round(after, 3)}"
                )
        return "\n".join(lines)


def perturb_reconstruction(
    reconstruction: pycolmap.Reconstruction, perturbation: Perturbation
) -> PerturbationReport:
    """Apply every requested perturbation to a reconstruction, in place.

    One entry point for the whole family - datum, intrinsics, point noise - so a
    recovery test states what it broke in one place and the caller can size the
    correspondence gate from the returned `reach`.

    Rotation and scale act about the block centroid, not the origin, so
    `shift_m` is the only source of bulk translation and the three datum
    magnitudes stay independent. Without that, a 0.4 deg rotation about a distant
    origin would also translate the block by hundreds of metres.

    Args:
        reconstruction: Modified in place - points, cameras and images.
        perturbation: What to change and by how much.

    Returns:
        A `PerturbationReport` describing what was applied.

    Raises:
        ValueError: If the reconstruction has no 3D points to perturb about.
    """
    points = list(reconstruction.points3D.values())
    if not points:
        raise ValueError("reconstruction has no 3D points")

    xyz = np.array([p.xyz for p in points], dtype=float)
    centre = xyz.mean(axis=0)
    lever = float(np.sqrt(np.mean(np.sum((xyz - centre) ** 2, axis=1))))

    rng = np.random.default_rng(perturbation.seed)
    report = PerturbationReport(
        lever_arm=lever,
        intrinsics_before={
            camera_id: np.array(camera.params, dtype=float)
            for camera_id, camera in reconstruction.cameras.items()
        },
    )

    if perturbation.shift_m or perturbation.rot_deg or perturbation.scale_ppm:
        direction = _unit(rng)
        rotation = pycolmap.Rotation3d(_unit(rng) * np.deg2rad(perturbation.rot_deg))
        scale = 1.0 + perturbation.scale_ppm * 1e-6
        # Sim3d maps x -> s R x + t, so pinning the centroid means solving
        # s R c + t = c + shift for t.
        translation = (
            centre - scale * (rotation * centre) + perturbation.shift_m * direction
        )
        transform = pycolmap.Sim3d(scale, rotation, translation)
        reconstruction.transform(transform)
        report.sim3d = transform
        report.reach = (
            perturbation.shift_m
            + lever * np.deg2rad(perturbation.rot_deg)
            + lever * abs(perturbation.scale_ppm) * 1e-6
        )

    if perturbation.point_noise_m:
        # Points only: the cameras stay put, so this is a genuine loss of
        # consistency rather than a movement of the datum.
        for point, offset in zip(
            points,
            rng.normal(0.0, perturbation.point_noise_m, (len(points), 3)),
            strict=True,
        ):
            point.xyz = point.xyz + offset

    for camera in reconstruction.cameras.values():
        params = np.array(camera.params, dtype=float)
        if perturbation.focal_pct:
            params[list(camera.focal_length_idxs())] *= (
                1.0 + perturbation.focal_pct / 100.0
            )
        if perturbation.principal_point_px:
            idxs = list(camera.principal_point_idxs())
            params[idxs] += perturbation.principal_point_px * _unit(rng)[: len(idxs)]
        if perturbation.extra_params_pct:
            params[list(camera.extra_params_idxs())] *= (
                1.0 + perturbation.extra_params_pct / 100.0
            )
        camera.params = params

    report.intrinsics_after = {
        camera_id: np.array(camera.params, dtype=float)
        for camera_id, camera in reconstruction.cameras.items()
    }
    return report


def _unit(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly distributed unit 3-vector."""
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)
