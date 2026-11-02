# Docker image (CUDA, ONNX Runtime, COLMAP GUI)

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
| [docker-compose.yml](../docker-compose.yml) | the `ba` service; runs anywhere, on the CPU, GUI-capable |
| [docker-compose.gpu.yml](../docker-compose.gpu.yml) | opt-in: NVIDIA GPU |
| [docker-compose.wsl.yml](../docker-compose.wsl.yml) | opt-in: Windows/WSL2 with NVIDIA GPU and accelerated GUI drawing |
| [.env.example](../.env.example) | every setting, documented |
| [docker/smoke_test.py](../docker/smoke_test.py) | checks every extractor/matcher pair and a bundle |

## Installation

You need an x86-64 machine with Docker and the Compose plugin (`docker compose
version` should work). Apple Silicon and other ARM machines are not supported: the
base image is amd64 only. Pick your case below, install its prerequisites, then
follow "Quick start". The `COMPOSE_FILE` line each case sets in `.env` is explained
in "Choosing a setup".

### Linux, CPU only (laptop or server)

- Install [Docker Engine and the Compose plugin](https://docs.docker.com/engine/install/),
  and add your user to the `docker` group to run it without `sudo`.
- Nothing to set for `COMPOSE_FILE`. Set `WITH_CUDNN=0` in `.env` to leave out cuDNN
  (about 1 GB), which is only used on a GPU.
- The GUI needs a desktop session. On a headless server just don't run it.

### Linux with an NVIDIA GPU

- Install the NVIDIA driver on the host (575 or newer is safe for the image's CUDA
  12.9; older 12.x drivers usually work too), Docker Engine, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Then register the toolkit with Docker and restart it:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

- In `.env`: `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml`.
- Only the host driver is needed; CUDA and cuDNN come with the image.

### Windows with an NVIDIA GPU (WSL2)

- Install the normal **Windows** NVIDIA driver. Do not install a Linux driver
  inside WSL: it uses the Windows one.
- Install WSL2 with a Linux distribution (e.g. Ubuntu), and
  [Docker Desktop](https://docs.docker.com/desktop/features/wsl/) with its WSL
  integration enabled for that distribution.
- Do everything below **inside the WSL distribution**: clone the repo into its home
  directory (not under `/mnt/c`, which is much slower) and run `docker compose`
  there. Images live on Docker Desktop's own disk, on C: by default.
- In `.env`: `COMPOSE_FILE=docker-compose.yml:docker-compose.wsl.yml`.
- GUI windows appear on the Windows desktop through WSLg, which comes with current
  WSL; no extra X server is needed. Reaching the machine over SSH works too.

## Quick start

```bash
git clone https://github.com/franioli/pycolmap_ba.git && cd pycolmap_ba
cp .env.example .env          # set UID/GID (id -u, id -g), DATA_DIR and your COMPOSE_FILE line
mkdir -p data                 # or point DATA_DIR at an existing folder
docker compose build          # first time ~10 min
docker compose run --rm ba python docker/smoke_test.py --device cpu   # CPU setup
docker compose run --rm ba python docker/smoke_test.py                # GPU setup: CPU and GPU
docker compose run --rm ba    # interactive shell
```

The smoke test checks every extractor and matcher pair plus a bundle. The image is
always built with CUDA, so without `--device cpu` it also tries the GPU and fails
on a machine that has none. A failing GPU line on a GPU setup usually means the
toolkit or driver is not set up (see the case above).

There is one service, `ba`, and the commands are the same on every machine. What
differs is which compose files are loaded, chosen once per machine in `.env`.

## Choosing a setup

`docker-compose.yml` is the base and runs anywhere, on the CPU. The other files
each add one thing and are merged on top of it, later files winning. Set
`COMPOSE_FILE` in `.env` to the files you want, separated by colons; it applies to
every `docker compose` command. Uncomment **one** line of `.env.example`:

| Machine | `COMPOSE_FILE` |
|---|---|
| CPU only (default, nothing to set) | `docker-compose.yml` |
| NVIDIA GPU | `docker-compose.yml:docker-compose.gpu.yml` |
| Windows (WSL2) with an NVIDIA GPU | `docker-compose.yml:docker-compose.wsl.yml` |

Without the `gpu` file the container runs on the CPU, even on a GPU machine.
`docker compose config` prints the merged result, so use it to check what is loaded.

## Working with Docker Compose

**Running.** `run` starts a fresh container, and `--rm` deletes it when it ends:

```bash
docker compose run --rm ba                        # interactive shell
docker compose run --rm ba python my_script.py    # one command, then exit
docker compose run --rm ba python -m pytest       # unit tests (-v, -x, -k name)
docker compose run --rm ba colmap gui             # COLMAP GUI, see below
```

Your repo and data are bind-mounts, so files written under `/workspace` and
`/data` stay on the host. Anything else done inside a container (e.g. `pip install`)
is lost when it exits: add the package to `pyproject.toml` and rebuild instead.
Run Python directly (`python`), not `uv run` or `uv sync`, since `pyproject.toml`
points pycolmap at a wheel path that exists only on the development laptop.

**When to rebuild.**

| You changed | Do |
|---|---|
| `src/`, `tests/`, scripts, notebooks | nothing: the repo is mounted |
| `DATA_DIR`, `COLMAP_CACHE`, `COMPOSE_FILE` | nothing: the next run uses them |
| `pyproject.toml`, `uv.lock`, the Dockerfile, or a build option below | `docker compose build` |
| a new COLMAP release | `docker compose build --pull`, see "Updating COLMAP" |

**Trying another setup once.** Override the files for a single command with
`-f`, which wins over `COMPOSE_FILE`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm ba
```

**Overriding `.env` for one command.** A variable set in the shell beats `.env`,
which beats the defaults in the files:

```bash
DATA_DIR=/mnt/data/block1 docker compose run --rm ba python my_script.py
EXTRAS=examples docker compose build
docker compose build --build-arg EXTRAS=examples      # same, for build options
docker compose --env-file .env.gpu run --rm ba        # a whole alternative file
```

`docker compose run -e NAME=value` only sets a variable *inside* the container; it
does not change the `${...}` values Compose substitutes into the files. Keep
`UID` and `GID` in `.env`: shells do not export them reliably.

**GUI.** `colmap gui` works with any setup wherever there is a display; the
GUI libraries are in the base image and the X11 socket is mounted by the base
file, so there is nothing to enable. On a headless Linux server just don't run it.

- **Windows (WSL2)**: windows appear on the Windows desktop of that machine, also
  when compose is started from an SSH session (`DISPLAY` defaults to WSLg's `:0`).
  `docker-compose.wsl.yml` makes the drawing use the GPU (Mesa's d3d12 driver). If
  the window fails to open or draws badly, use `docker-compose.gpu.yml` instead:
  the same GUI with software drawing.
- **Native Linux**: allow the container user once: `xhost +SI:localuser:$(id -un)`.
- **Over SSH with X forwarding**, `DISPLAY` points at the forwarded display, which
  the container cannot reach. Run with `DISPLAY=:0` or on the machine itself.

## Settings

All are set in `.env` (see `.env.example`) or overridden as shown above.

| Variable | Used at | Default | Meaning |
|---|---|---|---|
| `UID`, `GID` | build | 1000 | host user, so files in the mounts belong to you |
| `DATA_DIR` | run | `./data` | datasets, mounted at `/data`; must exist; keep it on the Linux/WSL filesystem, not `/mnt/c` |
| `COLMAP_CACHE` | run | `colmap-models` | model weights (~2 GB for all), see below |
| `COMPOSE_FILE` | every command | base file | which compose files to load |
| `EXTRAS` | build | `cloud` | Python extras to install, see below |
| `COLMAP_IMAGE` | build | 4.2.1 by digest | base image |
| `WITH_CUDNN` | build | `1` | `0` leaves out cuDNN (~1 GB) on machines without a GPU |
| `BUILD_JOBS` | build | `4` | parallel compile jobs; lower it if the build is killed |

`EXTRAS` picks groups from `pyproject.toml`. `cloud` (open3d) is needed for
reference-cloud constraints, pre-alignment and point-cloud export. `examples` adds
what the `enrich/` notebooks need (`enrich/` itself is reached through the repo
mount, not copied). An empty `EXTRAS=` installs the core only: pycolmap, pyceres,
numpy, pandas, enlighten.

`COLMAP_CACHE` is a volume name by default. Set it to a host path such as
`/home/francesco/.cache/colmap` to reuse weights a local install downloaded; that
path must exist before the first run, or Docker creates it owned by root.

Mounts inside the container: the repo at `/workspace` (edits apply without a
rebuild), `DATA_DIR` at `/data`, the weights at `~/.cache/colmap`, and a volume at
`~/.nv` that keeps the CUDA JIT cache (see "First GPU run").

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
