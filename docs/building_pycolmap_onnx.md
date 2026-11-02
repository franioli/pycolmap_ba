# Building pycolmap with ONNX (ALIKED + LightGlue)

The pycolmap wheels on PyPI are built **without ONNX Runtime**. ALIKED and
LightGlue are ONNX models, so using them throws in a COLMAP worker thread and
the process dies with `SIGABRT` (in Jupyter: "the kernel crashed", no traceback).
It is a compile-time feature, so it fails the same way on a GPU machine.

This project therefore installs pycolmap from a **locally built wheel**, not
from PyPI. Do not "fix" `pyproject.toml` by deleting the path source: `uv sync`
would silently pull the PyPI wheel back in and ALIKED/LightGlue would crash.

Full machine-specific notes (MKL, gotchas, timings) live outside the repo in
`~/software/pycolmap-cpu-setup.md`. This page is the short, repeatable version.

## Check whether a wheel has ONNX

```bash
ldd .venv/lib/python3.12/site-packages/pycolmap/_core*.so | grep onnx
```

No output means ALIKED and LightGlue will abort.

## How the build works

COLMAP's root `pyproject.toml` uses `scikit_build_core` with
`cmake.source-dir = "python/"`. Installing the repo runs CMake on
`python/CMakeLists.txt`, which calls `find_package(colmap REQUIRED)`. It does
**not** rebuild COLMAP: it links the bindings against an already-built,
**installed** COLMAP and inherits its feature set (ONNX included).

## Steps

Paths below are this machine's; adapt as needed.

### 1. Build COLMAP with ONNX

In `~/software/colmap` (Ninja build, `build/`). Flags that matter, as found in
`build/CMakeCache.txt`:

```bash
cmake -S . -B build -GNinja -DONNX_ENABLED=ON -DFETCH_ONNX=ON -DTESTS_ENABLED=OFF
cmake --build build -j$(nproc)
```

`FETCH_ONNX=ON` downloads ONNX Runtime during configure. (I did not re-run this
configure while writing this page; the exact original command was not recorded.
`CUDA_ENABLED` and `GUI_ENABLED` are also ON in the cache; the machine has no
CUDA, and `pycolmap.has_cuda` is `False`.)

### 2. Install COLMAP to a private prefix

```bash
cmake --install ~/software/colmap/build --prefix ~/software/colmap-install
```

This only copies headers, static libs and a consumable `colmap-config.cmake`.
Pointing `colmap_DIR` at `build/` does **not** work: the config there is an
install template whose `PACKAGE_PREFIX_DIR` resolves wrongly
(`include could not find requested file: .../FindDependencies.cmake`).

**Keep `~/software/colmap-install`.** ONNX Runtime is found at runtime through a
`RUNPATH` (`$ORIGIN/../lib:~/software/colmap-install/lib`) baked into the
module; deleting the prefix breaks `import pycolmap`. The COLMAP libs themselves
are static.

### 3. Build the pycolmap wheel

Run from this project directory (or set `VIRTUAL_ENV`), so uv uses the right
venv and Python version:

```bash
CMAKE_PREFIX_PATH=~/software/colmap-install \
  uv pip install --no-deps --force-reinstall ~/software/colmap
```

`--no-deps` stops uv from pulling the PyPI pycolmap back in. The build takes
about 2 minutes and the wheel is cached by uv.

### 4. Stash the wheel

```bash
cp ~/.cache/uv/sdists-v9/path/*/*/pycolmap-*-cp312-*.whl ~/software/wheels/
```

The result is `~/software/wheels/pycolmap-4.2.0.dev0-cp312-cp312-linux_x86_64.whl`.
Stashing it avoids recompiling on every `uv sync`.

## Linking the wheel in this project

[pyproject.toml](../pyproject.toml):

```toml
requires-python = ">=3.12,<3.13"   # the wheel is tagged cp312 only

dependencies = ["pycolmap>=4.2.0.dev0", ...]

[tool.uv.sources]
pycolmap = { path = "/home/francesco/software/wheels/pycolmap-4.2.0.dev0-cp312-cp312-linux_x86_64.whl" }
```

- Pin the **wheel**, not the COLMAP source directory. A source pin recompiles on
  every sync and needs `CMAKE_PREFIX_PATH` in the environment.
- The `requires-python` cap is mandatory: uv refuses to resolve while the project
  claims 3.13+ and the only pycolmap source is a cp312 wheel.
- If you rebuild and the version changes, update the wheel filename.

## Using a newer Python (3.13 / 3.14)

PyPI pycolmap (4.2.1) has cp310 to cp314 wheels, but they lack ONNX. To move
this project to a newer Python, build the wheel with that interpreter:

1. Redo step 3 with `uv ... --python 3.14` (the COLMAP install prefix from
   steps 1 and 2 is reusable).
2. Copy the new `cp314` wheel to `~/software/wheels/`.
3. Point `[tool.uv.sources]` at it and raise `requires-python` (e.g.
   `>=3.14,<3.15`). A single local wheel only covers one Python version.

This has not been tried yet. The SIFT-only fallback (PyPI wheel, no ALIKED or
LightGlue) works on 3.14 but is not what this project's examples use.

## Rebuilding after a COLMAP `git pull`

```bash
cd ~/software/colmap && git pull
cmake --build build -j$(nproc)
cmake --install build --prefix ~/software/colmap-install
cd ~/dev/pycolmap_ba
CMAKE_PREFIX_PATH=~/software/colmap-install \
  uv pip install --no-deps --force-reinstall ~/software/colmap
cp ~/.cache/uv/sdists-v9/path/*/*/pycolmap-*-cp312-*.whl ~/software/wheels/
```

Then update the wheel filename in `pyproject.toml` if the version changed.

## SIFT matching needs an MKL preload

SIFT brute-force matching (not extraction, not ALIKED/LightGlue) kills the
interpreter with `Intel MKL FATAL ERROR: Cannot load libmkl_def.so`, because
Python loads extension modules with `RTLD_LOCAL`. Fix:

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libmkl_rt.so
```

For Jupyter, set it before the kernel starts (shell that launches VS Code, or
`env` in `.venv/share/jupyter/kernels/python3/kernel.json`, which is wiped when
`ipykernel` is reinstalled).

## Using ALIKED + LightGlue

```python
extraction_options = pycolmap.FeatureExtractionOptions(
    type=pycolmap.FeatureExtractorType.ALIKED_N16ROT, max_image_size=2000
)
extraction_options.aliked.max_num_features = 4096  # default is only 2048

matching_options = pycolmap.FeatureMatchingOptions(
    type=pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
)
matching_options.aliked.lightglue.min_score = 0.1
```

Weights download once to `~/.cache/colmap/`. If you switch extractor type,
delete the database first: `extract_features` skips images already in it and
matching then aborts on mixed descriptor types.

## Verify

```bash
ldd .venv/lib/python3.12/site-packages/pycolmap/_core*.so | grep onnx
uv run python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
```
