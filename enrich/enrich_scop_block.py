"""Assemble the ENRICH-Aerial `nadir-2` block as a scop `Optimizer`.

This is `ENRICH-Aerial_scop_sfm_gcp.ipynb` cells 2-17 lifted into a function, so
experiments that need the same block do not have to re-derive fifteen cells of
object assembly. The notebook stays the annotated tutorial; every quirk it
documents is reproduced here, with a pointer back to it.

The input is the Sim3-aligned *free* network in `scop/colmap_sparse_free`. scop
is not an SfM engine, and its BA is a local optimiser, so the block has to sit
in the world frame before the GCP and position constraints mean anything.

    from enrich_scop_block import build
    ctx = build(max_points=5000)
    ctx.opt.fit(method="LM", iter_max=25, schur=True, compute_Qxx=False)
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from scop.utils.core.Camera import Camera, ModelType
from scop.utils.core.Image import Image, Point2D
from scop.utils.core.Optimizer import Optimizer
from scop.utils.core.Point3D import Point3D
from scop.utils.geometry.Rotation import opk_to_rmat, rmat_to_opk

import enrich_data

# COLMAP -> scop camera frame: a 180 deg rotation about x.
FLIP = np.diag([1.0, -1.0, -1.0])

# Id ranges, kept disjoint from the COLMAP ids (tie point ids stop at 45 571).
GCP_ID0 = 10_000_000
MARK_ID0 = 30_000_000


@dataclass
class BlockContext:
    """Everything the notebooks need to read results back out of `opt`."""

    opt: Optimizer
    cam: Camera
    images: dict[int, Image]
    tie_points: dict[int, Point3D]
    markers: dict[str, Point3D]
    id2name: dict[int, str]
    name2id: dict[str, int]
    data: Any
    truth: Any
    rec: Any
    gcp_id0: int = GCP_ID0
    flip: np.ndarray = field(default_factory=lambda: FLIP)
    n_tie_obs: int = 0
    n_marks: int = 0

    def reproj_rmse(self):
        """(tie RMSE px, n_tie, marker RMSE px, n_marker) -- RMS of the norm.

        COLMAP's `mean_reprojection_error` is the *mean* of the norm, so the two
        differ by ~10 % on identical geometry.
        """
        x = self.opt.to_jax_vector()
        acc = {False: [0.0, 0], True: [0.0, 0]}  # keyed by "is a marker mark"
        for img in self.opt.dict_obj["Image"].values():
            out = img.compute_batched_residuals(x)
            if out is None:
                continue
            res, idx, *_ = out
            res = np.asarray(res)
            for k, i in enumerate(idx):
                is_marker = img._lst_pt2[i].get_pt3().get_id() >= self.gcp_id0
                acc[is_marker][0] += float((res[k] ** 2).sum())
                acc[is_marker][1] += 1
        out = []
        for key in (False, True):
            s, n = acc[key]
            out += [math.sqrt(s / n) if n else float("nan"), n]
        return tuple(out)

    def camera_errors(self):
        """(Nx3 centre deltas [m], N rotation angles [deg]) against the truth EO."""
        gt_pose = enrich_data.cam_from_world_all(self.truth)
        dc, dr = [], []
        for img_id, img in self.opt.dict_obj["Image"].items():
            name = self.id2name[img_id]
            est = np.array([img.get_estimated_value(a) for a in ("Xs", "Ys", "Zs")])
            dc.append(est - self.truth.camera_xyz[name])
            # back to COLMAP axes: R_colmap = diag(1,-1,-1) @ R_scop
            R_est = FLIP @ opk_to_rmat(
                *[img.get_estimated_value(a) for a in ("omega", "phi", "kappa")],
                np,
                np,
            )
            cos = (np.trace(R_est @ gt_pose[name].rotation.matrix().T) - 1.0) / 2.0
            dr.append(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))))
        return np.array(dc), np.array(dr)

    def marker_errors(self, names):
        """Nx3 deltas [m] of the named markers against their true coordinates."""
        return np.array([
            [self.markers[n].get_estimated_value(a) for a in "XYZ"]
            - self.truth.marker_xyz[n]
            for n in names
        ])

    def marker_rmse(self, names):
        """RMS of the norm of the named marker errors."""
        e = self.marker_errors(names)
        return float(np.sqrt((e**2).sum(1).mean()))


def build(
    sparse_path: Path,
    min_track_length: int = 3,
    max_points: int | None = None,
    refine_focal: bool = True,
    use_camera_priors: bool = True,
    obs_noise_px: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> BlockContext:
    """Build the block.

    `obs_noise_px` is the one switch here that touches the **observations**: it
    adds white noise to every image mark (`Point2D.x/y`) at build time. Leave it
    at 0 for a perturbation experiment that must search for an unchanged
    minimum.
    """
    rng = np.random.default_rng(seed)

    truth = enrich_data.load(subblock="nadir-2", split_control_check=True)
    data = enrich_data.load(subblock="nadir-2", split_control_check=True)

    rec = pycolmap.Reconstruction(sparse_path)
    d = np.array([
        im.cam_from_world().inverse().translation - truth.camera_xyz[im.name]
        for im in rec.images.values()
    ])
    centre_rmse = float(np.sqrt((d**2).sum(1).mean()))
    assert centre_rmse < 1.0, (
        f"free network is not in the world frame ({centre_rmse:.2f} m) - align it first"
    )

    # --- camera -----------------------------------------------------------
    # COLMAP SIMPLE_PINHOLE (f, cx, cy) == scop Frame/SimpleRadial with k = 0.
    # Only `f` may be freed: scop projects v = h - ((y + dy)*f + cy), so
    # refining cy moves the principal point opposite to COLMAP (notebook cell 7).
    ccam = rec.cameras[1]
    f0, cx0, cy0 = (float(v) for v in ccam.params)
    cam = Camera.frame(
        width=ccam.width,
        height=ccam.height,
        focal_px=f0,
        model=ModelType.SimpleRadial,
        cx=cx0,
        cy=cy0,
        k=0.0,
        variables=["f"] if refine_focal else None,
    )
    cam._id = 1

    # --- images -----------------------------------------------------------
    # `Image.at` assigns `observations` verbatim, so the "Point2D" sentinel that
    # switches on the reprojection residuals must be listed explicitly.
    obs_names = ["Xs", "Ys", "Zs", "Point2D"] if use_camera_priors else ["Point2D"]

    images, id2name, name2id = {}, {}, {}
    for im in rec.images.values():
        cfw = im.cam_from_world()
        R_scop = FLIP @ cfw.rotation.matrix()
        S = cfw.inverse().translation  # camera centre
        omega, phi, kappa = rmat_to_opk(R_scop, np, np)  # radians

        img = Image.at(
            camera=cam,
            x=float(S[0]),
            y=float(S[1]),
            z=float(S[2]),
            omega=float(omega),
            phi=float(phi),
            kappa=float(kappa),
            variables=["Xs", "Ys", "Zs", "omega", "phi", "kappa"],
            observations=list(obs_names),
            _id=im.image_id,
        )

        if use_camera_priors:
            # Observed value only - the estimated pose stays at the COLMAP
            # solution. float() because enrich_data returns float32 arrays.
            img.Xs, img.Ys, img.Zs = (float(v) for v in data.camera_xyz[im.name])
            for axis in ("Xs", "Ys", "Zs"):
                img.set_sigma_var(axis, enrich_data.camera_position_sigma_m)

        images[im.image_id] = img
        id2name[im.image_id] = im.name
        name2id[im.name] = im.image_id

    assert all("Point2D" in i._observations for i in images.values()), (
        "the Point2D sentinel was dropped - the images have no reprojection residuals"
    )

    # --- tie points and their marks ---------------------------------------
    # Longest tracks first: the best-constrained points survive the subsampling.
    pts_sorted = sorted(
        rec.points3D.items(), key=lambda kv: len(kv[1].track.elements), reverse=True
    )
    tie_points = {}
    for pid, p in pts_sorted:
        if len(p.track.elements) < min_track_length:
            continue
        if max_points is not None and len(tie_points) >= max_points:
            break
        tie_points[pid] = Point3D.tie(
            x=float(p.xyz[0]), y=float(p.xyz[1]), z=float(p.xyz[2]), _id=int(pid)
        )

    lst_pt2, n_tie_obs, n_behind = [], 0, 0
    for im in rec.images.values():
        img = images[im.image_id]
        cfw = im.cam_from_world()
        R_scop, S = FLIP @ cfw.rotation.matrix(), cfw.inverse().translation

        for p2 in im.points2D:
            pid = int(p2.point3D_id)
            if pid not in tie_points:
                continue
            pt3 = tie_points[pid]

            # scop's camera looks down -Z: a visible point has Zc < 0.
            if (R_scop @ (np.array([pt3.X, pt3.Y, pt3.Z]) - S))[2] >= 0:
                n_behind += 1
                continue

            u, v = float(p2.xy[0]), float(p2.xy[1])  # raw u, v - no flip
            if obs_noise_px:
                u += float(rng.normal(0.0, obs_noise_px))
                v += float(rng.normal(0.0, obs_noise_px))
            pt2 = Point2D(_id=im.image_id * 100_000 + pid, x=u, y=v, _m_pt3=pt3)
            pt2.create_estimated()
            img.add_pt2(pt2)
            lst_pt2.append(pt2)
            n_tie_obs += 1

    # --- markers ----------------------------------------------------------
    markers = {}
    for i, name in enumerate(data.marker_names):
        x, y, z = (float(v) for v in data.marker_xyz[name])
        if name in data.control_names:
            pt3 = Point3D.control(x=x, y=y, z=z, _id=GCP_ID0 + i)
            for axis, sigma in zip("XYZ", enrich_data.gcp_sigma_m, strict=True):
                pt3.set_sigma_var(axis, float(sigma))  # per-axis, not isotropic
        else:
            pt3 = Point3D.check(x=x, y=y, z=z, _id=GCP_ID0 + i)
        markers[name] = pt3

    n_marks = 0
    for j, row in enumerate(data.marker_data.itertuples(index=False)):
        img_id = name2id.get(row.image_name)
        if img_id is None:  # image outside the sub-block
            continue
        u, v = float(row.u), float(row.v)
        if obs_noise_px:
            u += float(rng.normal(0.0, obs_noise_px))
            v += float(rng.normal(0.0, obs_noise_px))
        pt2 = Point2D(_id=MARK_ID0 + j, x=u, y=v, _m_pt3=markers[row.marker_name])
        pt2.create_estimated()
        images[img_id].add_pt2(pt2)
        lst_pt2.append(pt2)
        n_marks += 1

    # --- optimizer --------------------------------------------------------
    opt = Optimizer()
    opt.add_object(cam)
    for img in images.values():
        opt.add_object(img)
    for pt3 in list(tie_points.values()) + list(markers.values()):
        opt.add_object(pt3)
    for pt2 in lst_pt2:
        opt.add_object(pt2)

    opt.set_sigma("Point2D", enrich_data.tie_points_sigma_px)  # px, ties AND marks

    opt.update_nb_obs_var()
    opt.__dict__.pop("_cached_identity_l", None)  # never invalidated

    n_identity = len(opt._identity_obs_rows)
    expected = 3 * len(images) * use_camera_priors + 3 * len(data.control_names)
    assert n_identity == expected, (n_identity, expected)

    if verbose:
        print(
            f"{len(images)} images, {len(tie_points)} tie points, "
            f"{n_tie_obs} tie marks ({n_behind} behind camera), {n_marks} marker marks"
        )
        print(
            f"{len(markers)} markers = {len(data.control_names)} control "
            f"+ {len(data.check_names)} check"
        )
        print(
            f"variables {opt.nb_variables} | observations {opt.nb_observations} | "
            f"redundancy {opt.nb_observations - opt.nb_variables} | "
            f"identity rows {n_identity}"
        )
        if obs_noise_px:
            print(f"!! image observations carry {obs_noise_px} px of added white noise")

    return BlockContext(
        opt=opt,
        cam=cam,
        images=images,
        tie_points=tie_points,
        markers=markers,
        id2name=id2name,
        name2id=name2id,
        data=data,
        truth=truth,
        rec=rec,
        n_tie_obs=n_tie_obs,
        n_marks=n_marks,
    )
