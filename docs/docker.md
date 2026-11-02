# Docker image (CUDA, ONNX Runtime, optional GUI)

A ready-to-run environment for pycolmap_ba with every COLMAP feature extractor
and matcher: SIFT, ALIKED, LightGlue and LoMa, on CPU or an NVIDIA GPU.

## What is in it

The image starts from the **official `colmap/colmap` release image**. That image
already ships COLMAP built with CUDA, ONNX Runtime (with the CUDA provider), the
GUI, and the headers and static libraries needed to build against it. Only the
pycolmap bindings are compiled, against that install; COLMAP itself is never
rebuilt.

| | |
|---|---|
| Base | `colmap/colmap`, COLMAP 4.2.1 pinned by digest (Ubuntu 24.04, CUDA 12.9) |
| pycolmap | built from the source tag matching the base image's COLMAP |
| Added | Python 3.12, cuDNN 9, MKL dispatcher, the locked project dependencies |
| Python env | `/opt/venv`, on `PATH`; pycolmap_ba installed editable from `/workspace/src` |

PyPI has no usable alternative: `pycolmap` lacks both CUDA and ONNX, and
`pycolmap-cuda12` has CUDA but no ONNX, so ALIKED/LightGlue/LoMa abort with it.

## Files

| File | Purpose |
|---|---|
| [docker/Dockerfile](../docker/Dockerfile) | builder (pycolmap wheel) and runtime stages |
| [docker-compose.yml](../docker-compose.yml) | the `ba` service; runs anywhere, on the CPU |
| [docker-compose.gpu.yml](../docker-compose.gpu.yml) | opt-in: NVIDIA GPU |
| [docker-compose.gui.yml](../docker-compose.gui.yml) | opt-in: COLMAP GUI over X11 |
| [docker-compose.wslg.yml](../docker-compose.wslg.yml) | opt-in: GPU-accelerated OpenGL on WSL2 |
| [.env.example](../.env.example) | every setting, documented |
| [docker/smoke_test.py](../docker/smoke_test.py) | checks every extractor/matcher pair and a bundle |

## Quick start

```bash
cp .env.example .env          # set UID/GID (id -u, id -g) and DATA_DIR
mkdir -p data                 # or point DATA_DIR at an existing folder
docker compose build
docker compose run --rm ba python docker/smoke_test.py
docker compose run --rm ba    # interactive shell
```

There is one service, `ba`, and the commands are the same on every machine.
What differs is which compose files are loaded, set once in `.env` (see
"Choosing a setup"). With nothing set it runs on the CPU. On a machine without
an NVIDIA GPU you can also set `WITH_CUDNN=0` to leave out cuDNN (about 1 GB).

## Choosing a setup

`docker-compose.yml` is the base. The other files each add one thing and are
merged on top of it, later files winning. `COMPOSE_FILE` in `.env` lists the
files to load, separated by colons, and applies to every `docker compose`
command. Uncomment **one** line of `.env.example`:

| Machine | `COMPOSE_FILE` |
|---|---|
| CPU only (default, nothing to set) | `docker-compose.yml` |
| NVIDIA GPU | `docker-compose.yml:docker-compose.gpu.yml` |
| GPU + GUI, native Linux | `...:docker-compose.gpu.yml:docker-compose.gui.yml` |
| GPU + GUI, WSL2 | `...:docker-compose.gpu.yml:docker-compose.gui.yml:docker-compose.wslg.yml` |
| GUI without a GPU | `docker-compose.yml:docker-compose.gui.yml` |

`docker compose config` prints the merged result. Without a `gpu` file the
container runs on the CPU even on a GPU machine. To try a combination once, pass
the files explicitly, which overrides `.env`:
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm ba`.

Inside the container, run Python directly (`python script.py`), not `uv run`
or `uv sync`: `pyproject.toml` points pycolmap at a local wheel path that only
exists on the development laptop.

The unit tests are not copied into the image; they run from the repo mount
(pytest is installed):

```bash
docker compose run --rm ba python -m pytest
```

## Mounts

| Container path | Source | Notes |
|---|---|---|
| `/workspace` | the repo | edits to `src/` apply without rebuilding |
| `/data` | `DATA_DIR` | must exist; keep it on the Linux/WSL filesystem, not `/mnt/c` |
| `~/.cache/colmap` | `COLMAP_CACHE` | model weights, downloaded on first use (~2 GB for all) |
| `~/.nv` | volume | CUDA JIT cache, see "First GPU run" below |

`COLMAP_CACHE` can be a volume name (default) or a host path such as
`/home/francesco/.cache/colmap`, to reuse weights a local install downloaded. A
host path must exist before the first run; otherwise Docker creates it owned by
root.

## Optional dependencies

`EXTRAS` in `.env` picks the dependency groups from `pyproject.toml`, then
`docker compose build`:

- `cloud` (default): open3d, for reference-cloud constraints, pre-alignment and
  point-cloud export.
- `examples`: open3d, opencv, imageio, matplotlib, ipykernel, for the `enrich/`
  scripts and notebooks. `enrich/` itself is not copied into the image; it is
  reached through the `/workspace` mount.
- empty: the core only (pycolmap, pyceres, numpy, pandas, enlighten).

## GUI

The GUI libraries (Qt, Mesa) are part of the base image, so enabling the GUI
needs no rebuild: it is a choice of mounts and environment variables. Pick a
setup with the GUI in "Choosing a setup", then:

```bash
docker compose run --rm ba colmap gui
```

- **WSL2 (WSLg)**: add `docker-compose.wslg.yml` for GPU-accelerated OpenGL
  through Mesa's d3d12 driver, or leave it out for software OpenGL.
  Windows appear on the Windows desktop of that machine, also when compose is
  started from an SSH session (`DISPLAY` defaults to WSLg's `:0`).
- **Native Linux**: allow the container user first:
  `xhost +SI:localuser:$(id -un)`.

## Updating COLMAP

Images on Docker Hub are pushed only for COLMAP releases, so `latest` is the
newest release. To move to it, set `COLMAP_IMAGE=colmap/colmap:latest` in `.env`
and run `docker compose build --pull`. The builder reads the COLMAP version from
the image, clones the matching source tag, and stops if the image's headers do
not match that tag. To pin a new release, replace the digest in the Dockerfile
(`docker buildx imagetools inspect colmap/colmap:latest` prints it).

## Things to know

- **First GPU run.** Release images compile COLMAP's CUDA kernels (SIFT-GPU,
  dense MVS) for compute capabilities 5.0 to 7.5 and 9.0 only. On other GPUs,
  e.g. an RTX A2000 (8.6), the driver JIT-compiles them from PTX on first use,
  which is slow once and then cached in the `~/.nv` volume. ALIKED, LightGlue
  and LoMa run in ONNX Runtime and are not affected.
- **MKL preload.** `LD_PRELOAD` is set to MKL's dispatcher in the image: without
  it, SIFT brute-force matching aborts with `Cannot load libmkl_def.so` (see
  [building_pycolmap_onnx.md](building_pycolmap_onnx.md)).
- **Crashes have no traceback.** A missing ONNX or CUDA backend aborts the
  process from a COLMAP worker thread. The smoke test runs each pair in its own
  process for that reason.
- **WARN in the smoke test.** On its synthetic textures, SIFT + LightGlue,
  LoMa brute-force and LoMa B128 run but find no matches; all three match
  normally on real images. Use `--images /data/<block>/images` to check on
  real data.
- **`use_gpu` is honoured.** pycolmap ignores `use_gpu` in the option structs
  and decides from its `device` argument, whose default means GPU in any CUDA
  build. `pycolmap_ba.frontend` passes `device` explicitly, so
  `FeatureOptions(use_gpu=False)` really runs on the CPU (and `True` fails
  loudly without a GPU). Direct `pycolmap.extract_features` calls need
  `device=pycolmap.Device.cpu` on a machine without a GPU.
- **Build memory.** The pybind11 compile is heavy. If the build is killed, lower
  `BUILD_JOBS` in `.env`.
- **Image size** is about 15 GB: ~11 GB is the official base image (CUDA
  libraries, COLMAP), ~1.8 GB cuDNN, ~1.8 GB the Python environment (mostly
  open3d). `WITH_CUDNN=0` and an empty `EXTRAS` trim the last two.
- **Docker Desktop on Windows** stores images on its own virtual disk (C: by
  default), so check the free space there, not on the WSL disk.
