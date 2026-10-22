# pycolmap_ba

Custom bundle adjustment on top of pycolmap: ground control points (GCPs), pose priors, reference-cloud constraints, and posterior covariance estimation.

- Code: [src/pycolmap_ba/](src/pycolmap_ba/)

## Setup

```bash
uv sync
```

pycolmap comes from a local ONNX-enabled wheel (PyPI wheels crash on ALIKED/LightGlue), which pins Python to 3.12. See [docs/building_pycolmap_onnx.md](docs/building_pycolmap_onnx.md).

## Docs

- [GCP bundle adjustment](docs/gcp_bundle_adjustment.md)
- [Reference-cloud bundle adjustment](docs/reference_cloud_bundle_adjustment.md)
- [Posterior covariance](docs/posterior_covariance.md)
- [Pre-alignment and reporting plan](docs/prealign_compatibility_and_report_plan.md)
- [Building pycolmap with ONNX](docs/building_pycolmap_onnx.md)
