"""Smoke test for the Docker image: every extractor/matcher pair, then a bundle.

By default the images are rendered from a synthetic reconstruction, so no
dataset is needed. Each pair runs in its own subprocess: COLMAP reports a
missing ONNX Runtime or CUDA provider by aborting the process, which Python
cannot catch.

A pair that runs but verifies no matches is a WARN, not a failure: SIFT +
LightGlue, LoMa brute-force and LoMa B128 find nothing on the synthetic blob
textures but work on real images. Pass `--images` to check on real data.

Usage::

    python docker/smoke_test.py                  # CPU, plus GPU if built with CUDA
    python docker/smoke_test.py --device cpu     # e.g. on a machine without a GPU
    python docker/smoke_test.py --only ALIKED    # pairs whose name contains ALIKED
    python docker/smoke_test.py --images /data/some_block/images
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pycolmap

from pycolmap_ba import run_bundle_adjustment
from pycolmap_ba.frontend import FeatureOptions, extract_and_match

# (extractor, matcher). LoMa matchers only accept descriptors of their own
# extractor variant: LOMA_B128 pairs with LOMA_B128, the rest with LOMA_B.
PAIRS = [
    ("SIFT", "SIFT_BRUTEFORCE"),
    ("SIFT", "SIFT_LIGHTGLUE"),
    ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE"),
    ("ALIKED_N16ROT", "ALIKED_BRUTEFORCE"),
    ("LOMA_B", "LOMA_B"),
    ("LOMA_B", "LOMA_BRUTEFORCE"),
    ("LOMA_B128", "LOMA_B128"),
]


def synthesize(image_dir: Path) -> pycolmap.Reconstruction:
    """Render a small synthetic block into `image_dir`."""
    options = pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_frames_per_rig=6, num_points3D=800
    )
    reconstruction = pycolmap.synthesize_dataset(options)
    image_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.synthesize_images(
        pycolmap.SyntheticImageOptions(), reconstruction, image_dir
    )
    return reconstruction


def run_pair(
    work_dir: Path, image_dir: Path, extractor: str, matcher: str, use_gpu: bool
) -> None:
    """Extract and match one pair, printing the counts as the last stdout line."""
    device = "gpu" if use_gpu else "cpu"
    options = FeatureOptions(
        extractor=getattr(pycolmap.FeatureExtractorType, extractor),
        matcher=getattr(pycolmap.FeatureMatcherType, matcher),
        max_image_size=1600,
        max_num_features=2048,
        use_gpu=use_gpu,
    )
    database_path = work_dir / f"{extractor}-{matcher}-{device}.db"
    start = time.perf_counter()
    extract_and_match(database_path, image_dir, options, overwrite=True, verbose=False)
    elapsed = time.perf_counter() - start
    with pycolmap.Database.open(database_path) as db:
        print(
            f"keypoints={db.num_keypoints()} matches={db.num_matches()} "
            f"inliers={db.num_inlier_matches()} time={elapsed:.1f}s"
        )


def run_bundle(reconstruction: pycolmap.Reconstruction) -> bool:
    """Perturb the synthetic points and check the custom bundle pulls them back."""
    rng = np.random.default_rng(0)
    for point in reconstruction.points3D.values():
        point.xyz += rng.normal(scale=0.01, size=3)
    result = run_bundle_adjustment(reconstruction, verbose=False)
    summary = result.summary.ceres_summary
    ok = result.usable and summary.final_cost < summary.initial_cost
    print(
        f"  {'PASS' if ok else 'FAIL'} bundle: cost {summary.initial_cost:.3g} -> "
        f"{summary.final_cost:.3g}"
    )
    return ok


def main() -> None:
    """Run every pair on the requested devices and report PASS/WARN/FAIL."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "both"],
        default="both" if pycolmap.has_cuda else "cpu",
    )
    parser.add_argument("--only", default="", help="Substring filter on pair names.")
    parser.add_argument("--images", type=Path, help="Real images instead of synthetic.")
    parser.add_argument("--keep", action="store_true", help="Keep the work dir.")
    # Internal: run a single pair in this process.
    parser.add_argument("--pair", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.pair:
        work_dir, image_dir, pair, gpu = args.pair
        extractor, matcher = pair.split(":")
        run_pair(Path(work_dir), Path(image_dir), extractor, matcher, gpu == "1")
        return

    print(f"pycolmap {pycolmap.__version__}, has_cuda={pycolmap.has_cuda}")
    work_dir = Path(tempfile.mkdtemp(prefix="pycolmap_ba_smoke_"))
    reconstruction = synthesize(work_dir / "images")
    image_dir = args.images or work_dir / "images"
    print(f"  images: {image_dir}, work dir: {work_dir}")

    devices = {"cpu": [False], "gpu": [True], "both": [False, True]}[args.device]
    failures = 0
    for extractor, matcher in PAIRS:
        if args.only not in f"{extractor}:{matcher}":
            continue
        if not (
            hasattr(pycolmap.FeatureExtractorType, extractor)
            and hasattr(pycolmap.FeatureMatcherType, matcher)
        ):
            print(f"  SKIP {extractor} + {matcher}: not in this pycolmap version")
            continue
        for use_gpu in devices:
            label = f"{extractor} + {matcher} [{'gpu' if use_gpu else 'cpu'}]"
            command = [
                sys.executable,
                __file__,
                "--pair",
                str(work_dir),
                str(image_dir),
                f"{extractor}:{matcher}",
                "1" if use_gpu else "0",
            ]
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode != 0:
                failures += 1
                print(f"  FAIL {label} (exit code {proc.returncode})")
                lines = (proc.stdout + proc.stderr).strip().splitlines()
                print("\n".join(f"       {line}" for line in lines[-8:]))
                continue
            counts = proc.stdout.strip().splitlines()[-1]
            status = "WARN" if "inliers=0 " in counts else "PASS"
            print(f"  {status} {label}: {counts}")

    if not run_bundle(reconstruction):
        failures += 1
    if not args.keep:
        shutil.rmtree(work_dir, ignore_errors=True)
    print("no failures" if failures == 0 else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
