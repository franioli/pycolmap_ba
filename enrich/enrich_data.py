"""ENRICH-Aerial nadir-2 data: load the truth, optionally perturb it.

Built on the dataset's official loaders in `ENRICH_data_loading_demo.py`
(`load_camera_gt`, `load_gcp_locations`, `load_gcp_visibility`,
`get_image_list`) rather than re-parsing the CSVs.

`load()` reads the data as it is. `add_noise()` takes that object and returns
another of the same type with a simulated survey applied. Both hold the same
fields - just different numbers - so downstream code never has to care which
it is:

    truth = enrich_data.load()
    noisy = enrich_data.add_noise(truth)
"""

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

# Dataset root folder. A default, not a fixture: every loader takes it as an
# argument, so the data can live anywhere.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data/ENRICH-Aerial"

# Default camera/sub-block filter, e.g. "nadir-2". None loads the whole dataset.
SUBBLOCK = None

# --- camera -----------------------------------------------------------------
# The README quotes "35mm / 5,882px" for the nadir camera. That figure comes
# from the nominal 5.95 um pixel size; the sensor spec in the same document
# (35.9 mm / 6016 px = 5.96742 um) would instead give 5865.181 px. The
# difference is 0.29% and matters little - what matters is that fx and fy are
# not free to diverge, which is why the pipeline uses SIMPLE_PINHOLE.
IMAGE_WIDTH, IMAGE_HEIGHT = 6016, 4016
SENSOR_MM = (35.9, 24.0)
FOCAL_MM = 35.0
PIXEL_SIZE_MM = SENSOR_MM[0] / IMAGE_WIDTH  # 0.00596742
FOCAL_PX = FOCAL_MM / PIXEL_SIZE_MM  # 5865.181, from the sensor spec
PRINCIPAL_POINT = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)  # (3008, 2008)

# Blender cameras look down -Z with +Y up; COLMAP/Metashape use +Z forward,
# +Y down.
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0])

# --- noise applied to the Blender truth ------------------------------------
tie_points_sigma_px = 0.5  # 1-sigma on the Blender tie points in pixels
camera_position_sigma_m = 1  # 1-sigma on the Blender camera centres
gcp_sigma_m = [0.02, 0.02, 0.03]  # [x,y,z] or [xyz] 1-sigma on the Blender GCPs
gcp_sigma_px = 1.0  # 1-sigma on the Blender GCP image marks

CHECK_EVERY = 3  # every Nth GCP is held out as a check point

# Accept a scalar or a 3-vector for the GCP sigma.
gcp_sigma_m = np.broadcast_to(np.asarray(gcp_sigma_m, dtype=float), (3,)).copy()

SEED = 42


@dataclass
class EnrichData:
    """One set of coordinates - true or perturbed, with optional control/check split."""

    marker_data: pd.DataFrame  # image_name, marker_name, u, v (SUBBLOCK only)
    marker_xyz: dict[str, np.ndarray]  # marker world coordinates, metres
    cameras: dict[str, dict]  # load_camera_gt format
    camera_xyz: dict[str, np.ndarray]  # camera centres, metres
    marker_names: list[str]
    control_names: list[str] = field(default_factory=list)  # ground control points
    check_names: list[str] = field(default_factory=list)  # check points
    marker_type: dict[str, str] = field(default_factory=dict)  # 'round' | 'cross'
    image_paths: list[Path] = field(default_factory=list, repr=False)
    depth_paths: list[Path] = field(default_factory=list, repr=False)


def load(
    subblock: str | None = SUBBLOCK,
    split_control_check: bool = False,
    data_root: Path = DATA_ROOT,
) -> EnrichData:
    """Read the dataset as it is, via the official loaders. No noise.

    Args:
        subblock: camera / sub-block name to keep, e.g. "nadir-2". Matched as a
            substring of the image name. None (the default) loads every camera.
        split_control_check: if True, subdivide markers into ground control
            points and check points. If False, leave that subdivision unset.
        data_root: dataset root folder.
    """
    cameras = load_camera_gt(data_root)
    gcp_world = load_gcp_locations(data_root)
    visibility = load_gcp_visibility(data_root)

    # Flatten {image: {marker: (u,v)}} into a tidy frame, subblock only.
    rows = [
        (image_name, marker_name, float(uv[0]), float(uv[1]))
        for image_name, markers in visibility.items()
        if subblock is None or subblock in image_name
        for marker_name, uv in markers.items()
    ]
    marks = pd.DataFrame(rows, columns=["image_name", "marker_name", "u", "v"])
    marks = marks.sort_values(["image_name", "marker_name"]).reset_index(drop=True)

    marker_names = sorted(marks.marker_name.unique())
    marker_xyz = {
        name: np.asarray(gcp_world[name]["position"], dtype=float)
        for name in marker_names
    }
    marker_type = {name: gcp_world[name]["type"] for name in marker_names}
    camera_xyz = {
        label: np.asarray(entry["position"], dtype=float)
        for label, entry in cameras.items()
    }

    control_names: list[str] | None
    check_names: list[str] | None

    if split_control_check:
        # A check point seen by a single camera cannot be triangulated, so it
        # cannot be scored as a check point - force those into the control set.
        check = set(marker_names[::CHECK_EVERY])
        views = marks.groupby("marker_name").size()
        check -= set(views[views < 2].index)

        control_names = [name for name in marker_names if name not in check]
        check_names = [name for name in marker_names if name in check]
    else:
        control_names = None
        check_names = None

    return EnrichData(
        marker_xyz=marker_xyz,
        marker_data=marks,
        camera_xyz=camera_xyz,
        cameras=cameras,
        marker_names=marker_names,
        control_names=control_names,
        check_names=check_names,
        marker_type=marker_type,
        image_paths=get_image_list(data_root, camera_name=subblock),
        depth_paths=get_image_list(data_root, camera_name=subblock, is_depth=True),
    )


def add_noise(
    data: EnrichData,
    gcp_sigma_m: np.ndarray = gcp_sigma_m,
    gcp_sigma_px: float = gcp_sigma_px,
    camera_position_sigma_m: float = camera_position_sigma_m,
    seed: int = SEED,
) -> EnrichData:
    """Return a copy of `data` with a simulated survey applied.

    The input is left untouched, so the caller keeps the truth for scoring.
    """
    rng = np.random.default_rng(seed)
    sigma_m = np.broadcast_to(np.asarray(gcp_sigma_m, dtype=float), (3,))

    marker_xyz = {
        name: xyz + rng.normal(0.0, sigma_m) for name, xyz in data.marker_xyz.items()
    }
    marks = data.marker_data.copy()
    marks["u"] = marks.u + rng.normal(0.0, gcp_sigma_px, len(marks))
    marks["v"] = marks.v + rng.normal(0.0, gcp_sigma_px, len(marks))
    camera_xyz = {
        label: c + rng.normal(0.0, camera_position_sigma_m, 3)
        for label, c in data.camera_xyz.items()
    }

    return replace(
        data,
        marker_xyz=marker_xyz,
        marker_data=marks,
        camera_xyz=camera_xyz,
    )


def cam_from_world(data: EnrichData, label: str):
    """Ground-truth pose as a pycolmap.Rigid3d, in COLMAP axes.

    cameras.csv stores the quaternion as (w, x, y, z) for a *world-from-camera*
    rotation in Blender axes (camera looks down -Z, +Y up). COLMAP wants
    cam-from-world with +Z forward and +Y down, hence the BLENDER_TO_OPENCV flip.
    """
    import pycolmap  # local import: keeps this module importable without it

    entry = data.cameras[label]
    w, x, y, z = entry["quaternion"]
    world_from_cam = pycolmap.Rotation3d(np.array([x, y, z, w], dtype=float)).matrix()
    R = BLENDER_TO_OPENCV @ world_from_cam.T
    center = np.asarray(entry["position"], dtype=float)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ center)


def cam_from_world_all(data: EnrichData) -> dict:
    """Ground-truth poses for every camera, keyed by image name.

    Ready to hand to `pycolmap_ba.utils.GroundTruth(cam_from_world=...)`.
    """
    return {label: cam_from_world(data, label) for label in data.cameras}


def load_depth(image_name: str, data_root: Path = DATA_ROOT) -> np.ndarray:
    """Ground-truth depth map for an image, in metres.

    Values are **z-depth along the optical axis**, not ray length (verified: a
    nadir frame's corner/centre ratio is 0.991, whereas ray length would give
    1/cos(31.6 deg) = 1.174).

    The enrich's load_image() uses cv2, but the opencv build here reports
    `OpenEXR: NO` and returns None for .exr, so read with imageio instead.
    """
    import imageio.v3 as iio

    stem = Path(image_name).stem
    path = data_root / "depth" / "exr" / f"{stem}_depth.exr"
    depth = np.asarray(iio.imread(path), dtype=np.float32)
    if depth.ndim == 3:  # some writers store a single channel as HxWx1
        depth = depth[..., 0]
    return depth


def sample_depth(depth: np.ndarray, u: float, v: float, max_range: float = 1.0):
    """Ground-truth depth at (u, v), or None if the neighbourhood is not smooth.

    A depth discontinuity (roof edge, tree) makes sampling ambiguous - a
    sub-pixel shift changes the value by metres - so pixels whose 3x3
    neighbourhood spans more than `max_range` metres are rejected instead.

    Args:
        depth: Depth map, metres.
        u: Column coordinate, pixels.
        v: Row coordinate, pixels.
        max_range: Maximum depth spread tolerated over the 3x3 window, metres.

    Returns:
        The depth at the nearest pixel, or None.
    """
    col, row = int(u), int(v)
    if not (1 <= col < depth.shape[1] - 1 and 1 <= row < depth.shape[0] - 1):
        return None
    window = depth[row - 1 : row + 2, col - 1 : col + 2]
    if not np.isfinite(window).all() or window.max() - window.min() > max_range:
        return None  # depth discontinuity: sampling would be ambiguous
    return float(window[1, 1])


def summary(data: EnrichData) -> str:
    head = (
        f"{len(data.marker_data)} marker image marks, {len(data.marker_names)} markers, "
        f"{data.marker_data.image_name.nunique()} images"
    )
    if data.control_names is None or data.check_names is None:
        return f"{head}\ncontrol/check split: disabled"
    return (
        f"{head}\n"
        f"control: {len(data.control_names)} points | "
        f"check: {len(data.check_names)} points ({', '.join(data.check_names)})"
    )


### === ENRICH Data loader Helpers ===


def get_image_list(
    directory: Path, camera_name: str | None = None, is_depth: bool = False
) -> list:
    """Get the list of images in a directory.
    Camera name can be optionally specified to get only images of a specific camera.

    Args:
        directory (Path): path to the dataset directory.
        camera_name (str, optional): optional name of a camera, if set only images from that camera are returned.
                                     If not set all images from all cameras are returned. Defaults to None.
        is_depth (bool, optional): if True return the list of depth images instead of RGB images. Defaults to False.

    Returns:
        List: list of path to images.
    """
    if is_depth:
        directory = directory.joinpath("depth/exr")
        suffix = "_depth.exr"
    else:
        directory = directory.joinpath("images")
        suffix = ".jpg"
    pattern = "*" + camera_name if camera_name else "*"
    pattern += suffix
    return sorted(directory.glob(pattern))


def load_image(
    path: Path,
    camera_name: str = None,
    frame_number: int = None,
    is_depth: bool = False,
) -> np.ndarray:
    """Load an image by specifying the full path (including image name) or the camera name and frame number.

    Args:
        path (Path): path to the dataset directory (if camera_name and frame_number are set) or path to the image.
        camera_name (str, optional): name of the camera. Defaults to None.
        frame_number (int, optional): frame number of the camera. Defaults to None.
        is_depth (bool, optional): if True the depth image is loaded instead of the RGB one. Defaults to False.

    Returns:
        np.ndarray:
    """
    if camera_name is not None and frame_number is not None:
        if is_depth:
            path = path.joinpath("depth/exr")
            camera_name += "_depth"
            extension = ".exr"
        else:
            path = path.joinpath("images")
            extension = ".jpg"
        path = path.joinpath(f"{frame_number:02d}_{camera_name}")
        path = path.with_suffix(extension)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:  # depth only has 1 channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB conversion
    return img


def colorize_depth(depth: np.ndarray, color_lut: Path, min_depth: float) -> np.ndarray:
    """Compute colorized previews of depth maps.
    The result is the same as the images contained in the `png-preview` folders.

    Args:
        depth (np.ndarray): depth map.
        color_lut (Path): path to the color lookup table file
                          (<dataset-folder>/depth/png-preview/lut.png).
        min_depth (float): depth value of the lower bound of the color lookup table.
                           For depth preview images included in the datasets this
                           value is 114 for ENRICH-Aerial dataset, 0 for ENRICH-Square
                           and ENRICH-Statue.

    Returns:
        np.ndarray: RGB colorized preview of the input depth map.
    """
    lut = cv2.imread(str(color_lut), cv2.IMREAD_UNCHANGED)  # load color lookup table
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB)  # BGR to RGB conversion
    lut = lut[0, ...]  # drop unnecessary rows
    step = 0.25  # one color each 25cm
    levels = lut.shape[0] - 1
    depth = (depth - min_depth) / step  # shift & scale depth to the lut range
    depth_mask = depth > levels  # pixels further than lut max are at infinite (black)
    depth = np.round(depth).astype(np.int32)
    depth[depth_mask] = levels  # set infinite pixels
    depth_colorized = lut[depth]  # map depth to colormap
    return depth_colorized


def load_camera_gt(directory: Path, fmt: Literal["csv", "tsv"] = "csv") -> dict:
    """Load camera ground truths from cameras.csv/.tsv file.

    Args:
        directory (Path): path to the dataset directory.
        fmt (Literal['csv', 'tsv'], optional): use comma or tab separated values file. Defaults to 'csv'.

    Returns:
        Dict: root level keys are labels in the ground truth file (first column).
              For each key another dictionary provides ground truth camera pose information (remaining columns).
              Sample structure:
                 {
                    '01_nadir.jpg': {
                        'position': array([-227.39603, -...e=float32),     # X,Y,Z position
                        'opk': array([-0.,  0., -0....e=float32),          # Omega, Phi, Kappa rotation degrees
                        'ypr': array([0., 0., 0.], ...e=float32),          # Yaw, Pitch, Roll rotation degrees
                        'quaternion': array([1., 0., 0., 0...e=float32),   # W,X,Y,Z rotation quaternion
                        'lookat': array([ 0.,  0., -1....e=float32)        # X,Y,Z lookat direction vector
                    },
                    '01_forward.jpg': {...},
                    ...
                 }
    """
    camera_gt = {}
    with open(
        directory.joinpath("cameras." + fmt), encoding="utf-8", newline=""
    ) as gt_file:
        reader = csv.DictReader(gt_file, delimiter="," if fmt == "csv" else "\t")
        for row in reader:
            label = row.pop("label")
            entry = {
                "position": np.array(
                    (row["position_x"], row["position_y"], row["position_z"]),
                    dtype=np.float32,
                ),
                "opk": np.array(
                    (row["omega"], row["phi"], row["kappa"]), dtype=np.float32
                ),
                "ypr": np.array(
                    (row["yaw"], row["pitch"], row["roll"]), dtype=np.float32
                ),
                "quaternion": np.array(
                    (
                        row["rotation_w"],
                        row["rotation_x"],
                        row["rotation_y"],
                        row["rotation_z"],
                    ),
                    dtype=np.float32,
                ),
                "lookat": np.array(
                    (row["lookat_x"], row["lookat_y"], row["lookat_z"]),
                    dtype=np.float32,
                ),
            }
            camera_gt[label] = entry
    return camera_gt


def load_gcp_locations(directory: Path, fmt: Literal["csv", "tsv"] = "csv") -> dict:
    """Load GCP ground truths from gcp_list.csv/.tsv file.

    Args:
        directory (Path): path to the dataset directory.
        fmt (Literal['csv', 'tsv'], optional): use comma or tab separated values file. Defaults to 'csv'.

    Returns:
        Dict: root level keys are gcp names in the ground truth file (first column).
              For each key a dictionary specifies the 3D position (numpy array) and type (string) of the GCP.
              Sample structure:
                 {
                    'gcp.001': {'position': array([-218.01, -112.48, 18.23], dtype=float32), 'type': 'round'},
                    'gcp.002': {'position': array([-103.46, -112.33, 33.24], dtype=float32), 'type': 'round'},
                    ...
                 }
    """
    gcps = {}
    with open(
        directory.joinpath("gcp_list." + fmt), encoding="utf-8", newline=""
    ) as gcp_file:
        reader = csv.DictReader(gcp_file, delimiter="," if fmt == "csv" else "\t")
        for row in reader:
            label = row.pop("gcp_name")
            entry = {
                "position": np.array(
                    (row["x_east"], row["y_north"], row["z_altitude"]), dtype=np.float32
                ),
                "type": row["type"],
            }
            gcps[label] = entry
    return gcps


def load_gcp_visibility(directory: Path, fmt: Literal["csv", "tsv"] = "csv") -> dict:
    """Load GCP visibility ground truths from gcp_images_list.csv/.tsv file.

    Args:
        directory (Path): path to the dataset directory.
        fmt (Literal['csv', 'tsv'], optional): use comma or tab separated values file. Defaults to 'csv'.

    Returns:
        Dict: root level keys are image names in the ground truth file (first column).
              For each key another dictionary includes the visible GCPs and their image coordinates.
              Sample structure:
                 {
                    '01_nadir.jpg': {
                        'gcp.015': array([4310.5337, 15...e=float32),
                        'gcp.026': array([2935.8276,  8...e=float32)
                    },
                    '01_forward.jpg': { ... },
                    ...
                 }
    """
    camera_gcp_visibility = {}
    with open(
        directory.joinpath("gcp_images_list." + fmt),
        encoding="utf-8",
        newline="",
    ) as gt_file:
        reader = csv.DictReader(gt_file, delimiter="," if fmt == "csv" else "\t")
        for row in reader:
            label = row.pop("image_name")
            gcp_coordinate = np.array(
                (row["image_u"], row["image_v"]), dtype=np.float32
            )
            if label not in camera_gcp_visibility:
                camera_gcp_visibility[label] = {row["gcp_name"]: gcp_coordinate}
            else:
                camera_gcp_visibility[label][row["gcp_name"]] = gcp_coordinate
    return camera_gcp_visibility


def load_3d_mesh(
    directory: Path, print_progress: bool = False
) -> "o3d.geometry.TriangleMesh":  # noqa: F821
    """Load the 3D mesh model of the dataset.
    Includes only geometry without textures.

    Args:
        directory (Path): path to the dataset directory.
        print_progress (bool): whether to print mesh loading progress to console. Defaults to False.

    Returns:
        o3d.geometry.TriangleMesh: 3D tri-mesh geometry model of the dataset.
    """
    import open3d as o3d

    path = directory.joinpath("3D/geometry").joinpath(directory.name + ".obj")
    mesh = o3d.io.read_triangle_mesh(str(path), print_progress=print_progress)
    return mesh


if __name__ == "__main__":
    truth = load()
    noisy = add_noise(truth)
    print(summary(truth))
    print(
        f"\nnoise: GCP xy {gcp_sigma_m[0] * 100:.0f} cm, "
        f"z {gcp_sigma_m[2] * 100:.0f} cm | marks {gcp_sigma_px} px | "
        f"camera centres {camera_position_sigma_m * 100:.0f} cm"
    )
    name = truth.marker_names[0]
    print(
        f"{name}: type {truth.marker_type[name]}, shifted by "
        f"{np.round(noisy.marker_xyz[name] - truth.marker_xyz[name], 4)} m"
    )
    print(
        f"marks shifted by "
        f"{np.abs(noisy.marker_data.u - truth.marker_data.u).mean():.3f} px mean |du|"
    )
    print(f"\n{len(truth.image_paths)} images, {len(truth.depth_paths)} depth maps")
    depth = load_depth(truth.image_paths[0].name)
    print(f"depth[0] {depth.shape} range {depth.min():.1f}..{depth.max():.1f} m")
    print(f"\nfocal: {FOCAL_PX:.3f} px | principal point: {PRINCIPAL_POINT}")
