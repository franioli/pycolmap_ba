# pycolmap_ba

Custom bundle adjustment on top of pycolmap: ground control points (GCPs), pose priors, reference-cloud constraints, and posterior covariance estimation.

- Code: [src/pycolmap_ba/](src/pycolmap_ba/)

## Setup

```bash
uv sync                    # core: pycolmap, pyceres, numpy, pandas
uv sync --extra cloud      # + open3d, for reference-cloud constraints and export
uv sync --extra examples   # + what the enrich/ examples and notebooks need
```

pycolmap comes from a local ONNX-enabled wheel (PyPI wheels crash on ALIKED/LightGlue), which pins Python to 3.12. See [docs/building_pycolmap_onnx.md](docs/building_pycolmap_onnx.md).

Or skip the local build and use Docker, with CUDA and an optional COLMAP GUI: see [docs/docker.md](docs/docker.md).

## Tests and example

```bash
uv run pytest
```

The tests ([tests/](tests/)) run on a synthetic pycolmap block with known truth: each one moves the block by a 7-DoF similarity the tie points cannot see, and checks that markers, camera pose priors or a reference cloud bring it back. [examples/synthetic_custom_bundle_adjustment.ipynb](examples/synthetic_custom_bundle_adjustment.ipynb) walks through the same setup step by step, with all three constraints in one bundle (needs the `examples` extra).

## Docs

- [GCP bundle adjustment](docs/gcp_bundle_adjustment.md)
- [Reference-cloud bundle adjustment](docs/reference_cloud_bundle_adjustment.md)
- [Posterior covariance](docs/posterior_covariance.md)
- [Pre-alignment and reporting plan](docs/prealign_compatibility_and_report_plan.md)
- [Building pycolmap with ONNX](docs/building_pycolmap_onnx.md)
- [Docker image](docs/docker.md)
