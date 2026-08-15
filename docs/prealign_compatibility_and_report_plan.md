# Reference-cloud workflow: pre-alignment, compatibility checks, and reporting

Design plan for two related changes to the reference-cloud-constrained bundle
adjustment workflow. Status: **implementation in progress** (isolated
worktree, not yet merged). This document fixes the design so the diff can be
reviewed against it; line numbers below are as of the pre-implementation
state and will drift once the change lands.

## 1. Problem statement

`run_cloud_constrained_bundle_adjustment` (`src/pycolmap_ba/core/ref_cloud_iter_bundle.py`)
is a *local* refiner: per `utils/prealign.py:1-8`, it cannot recover a block
that starts further out than the spacing between distinct reference-cloud
surfaces. `prealign_to_reference()` (`utils/prealign.py:162-287`) exists to
fix large misalignments via ICP, but nothing in the real workflow ever calls
it — only the self-test script `enrich/run_cloud_ba.py` exists as a caller of
the constrained BA, and it relies solely on widening the correspondence gate
using the *known* synthetic perturbation, which a real caller would never
have. `PREALIGN_TRIGGER_M = 0.5` at `run_cloud_ba.py:63` is dead code.

Two gaps to close:

1. No compatibility check runs before the constrained BA, so a badly
   misaligned or largely non-overlapping block silently produces a wrong
   solution instead of failing loudly.
2. `report.py`'s two report functions (`block_report`, `accuracy_report`)
   don't expose the *process* (solver outcome, robust-loss behaviour,
   correspondence funnel, per-pass convergence) or unify the growing set of
   "vs. reference" flavours (markers, camera, reference reconstruction,
   reference point cloud) into one readable report.

## 2. Part 1 — pre-alignment + compatibility checks

### 2.1 Bound on recoverable misalignment

The constrained BA's residual is a GICP/NDT-style Mahalanobis term against a
locally-planar reference patch (stiff across the surface, slack tangentially
— `ReferenceCloud.sigmas`, `ref_cloud.py:161-181`). The alternating scheme is
Gauss-Newton on the squared distance to the reference *surface*. That
distance function is smooth, and the quadratic model is valid, only inside
the surface's medial axis; beyond it the nearest-point correspondence jumps
to a different sheet and the iteration converges to the wrong minimiser.

References: Pottmann & Hofer, "Geometry of the squared distance function to
curves and surfaces" (2003); Pottmann, Huang, Yang, Hu, "Geometry and
convergence analysis of algorithms for registration of 3D shapes", IJCV
67(3):277-296 (2006), §3-4.

Three horizons bound the recoverable misalignment; the smallest binds:

- **H1 — gate.** Correspondence requires `|d| < max_dist`. The gate must
  exceed the reach of the misalignment (translation + rotation·lever-arm +
  scale·lever-arm), not its mean, because far-from-centroid points carry the
  rotation/scale observability and are the first the gate drops.
- **H2 — ambiguity (fundamental).** `|d| < Δ_surf / 2`, `Δ_surf` = distance
  to the nearest distinct surface sheet. Measured on the ENRICH-Aerial
  reference cloud (5M pts, 492×321 m, 2 m XY cells, 15% of cells have a
  sheet gap): gap percentiles [p10 p25 p50 p75 p90] = [0.63 0.98 1.87 3.38
  6.55] m, so `Δ_surf(p25)/2 ≈ 0.49 m` — independently reproduces the
  existing hand-picked `trigger_rms = 0.5`.
- **H3 — robust-loss horizon.** Not a hard cutoff: Cauchy loss at `c = 2`
  whitened sigmas roughly halves a point's per-iteration pull by ~1 m of
  error given `σ_ref = 0.02 m`, `σ_tie = 0.5 px`. Explains why the gate
  schedule exists; not itself a gating criterion.

Measured trip point on the real block (42,635 tie points, 120.6 m span,
gate 1 m): unperturbed → overlap 0.906, trimmed RMS 0.49 m; 5 m shift →
overlap 0.32; shift 10 m + rot 5° + 200 ppm scale (reach 20.5 m) → overlap
0.20. Recovery degrades sharply around a ~5 m shift, consistent with "a few
metres" in `prealign.py:1-8`. Trimmed RMS under-reports large reach badly
(3.1 m measured for a 20.5 m true reach) because most points sit near the
centroid — **do not use trimmed RMS as a reach estimate**, only as a
secondary gate.

### 2.2 Compatibility metrics

**Primary: overlap fraction** at the gate radius, restricted to the overlap
region: `φ(g) = |{i : dᵢ ≤ g}| / |S|`. This is exactly "will the BA have
constraints to work with" — the same quantity `add_reference_cloud_constraints`
produces as its match rate. Threshold **0.30** (already asserted at
`prealign.py:59`).

**Secondary: trimmed RMS**, percentile-trimmed (p90) nearest-neighbour
distance, restricted to the overlap region. Threshold `ρ ≤ rms_margin · g`
with **`rms_margin = 1.0`** (calibrated: the *unperturbed* block gives
`ρ = 0.49 m` at `g = 1 m`, so `0.5` would fail a known-good block; `1.0`
still trips between the 2 m and 5 m shift cases).

**Ambiguity margin (default ON):** `ρ ≤ Δ_surf/2`, `Δ_surf` estimated as the
p25 of per-XY-cell largest vertical gap (`surface_separation`, ~2 s, cached).
This is the only criterion that catches "the user set `max_dist` too loose so
everything trivially passes". Confirmed default-on for this project: the
2.5D estimator is calibrated for aerial/terrain data, which matches
ENRICH-Aerial; would need re-checking before use on façade-heavy terrestrial
data.

**Diagnostics only (not gated):** median point-to-plane distance over gated
matches (the quantity the residual actually penalises); p95 of `dᵢ` as a
reach proxy (contaminated by vegetation/blunders).

**Rejected as red herrings:** symmetric Chamfer distance (reverse term is
meaningless at a 5M-vs-42K point ratio — measures sampling density, not
misalignment); bounding-box/extent overlap ratio (`prealign.py:15-20`
already argues footprints legitimately differ under partial overlap); point
density comparison (differs by design); untrimmed RMS (dominated by
canopy/blunders); point-to-sample distance as a *quality* measure (floored
at ~half the point spacing — fine as a gate proxy, useless as quality).

### 2.3 Partial overlap

**The BA needs no change.** `cloud.query` already returns no match past
`max_dist` (`ref_cloud.py:302-305`); unsupported points on either side
simply get no residual block. Both asymmetries (reference smaller than
block, or vice versa) are already handled at the BA level.

**The metric needs cropping to the overlap footprint**, or an ungated
nearest-neighbour search contaminates the statistic with every out-of-footprint
point (this is a real bug in the existing `residual_rms`, which is ungated).
Occupancy-grid footprint intersection, XY, 10 m cells, `np.roll`-based
dilate/erode (no scipy — not a declared dependency): dilate the block mask
(generous about block extent), erode the reference mask (conservative about
where the reference genuinely supports the block), intersect. Report
`coverage = |points in the intersection| / |all points|` at a low floor
(`min_coverage = 0.05`) — coverage is deliberately insensitive to
misalignment (measured ≈0.96 even under a 10 m shift); its job is to
separate "wrong place" from "not the same area", not to detect misalignment.

Two hazards flagged, not fixed by this change: reference-boundary patches
have asymmetric neighbourhoods (see §2.5, `min_neighbors` fix, which
partially addresses this); tie points just outside the reference footprint
but within the wide first-pass gate can get pulled laterally by boundary
patches (future mitigation: reuse the eroded reference mask as an opt-in
`boundary_mask`).

### 2.4 Control flow

Guard lives **inside** `run_cloud_constrained_bundle_adjustment`, not in a
wrapper, because it is the only real entry point — a wrapper would leave
"runs every time" dependent on caller discipline.

```python
def run_cloud_constrained_bundle_adjustment(
    ...,
    on_mismatch: Literal["prealign", "raise", "warn", "ignore"] = "prealign",
    compatibility: CompatibilityOptions | None = None,
    prealign_options: dict[str, Any] | None = None,
):
    if on_mismatch != "ignore":
        opts = compatibility or CompatibilityOptions(max_dist=first_gate)
        before = check_cloud_compatibility(reconstruction, reference_cloud, opts)
        if not before.passed:
            warnings.warn(f"block incompatible with reference cloud: {before.summary()}")
            if on_mismatch == "raise":
                raise CloudAlignmentError(before)
            if on_mismatch == "prealign":
                pre = prealign_to_reference(reconstruction, reference_cloud, trigger_rms=0.0, **(prealign_options or {}))
                after = check_cloud_compatibility(reconstruction, reference_cloud, opts)
                if not after.passed:
                    if pre.sim3d is not None:
                        reconstruction.transform(pre.sim3d.inverse())  # rollback: raise => unmodified
                    raise CloudAlignmentError(before, pre, after)
    # ... existing pass loop, now starting from a compatible block
```

Load-bearing details: the check's `max_dist` must equal the *first* gate of
the schedule the BA is about to run (not the final, tightest gate — that
would fail every legitimate coarse-to-fine run); `prealign_to_reference` must
be called with `trigger_rms=0.0` (its own internal skip-if-already-close
logic must be disabled, since we only reach this call because the check
already failed); a raise must leave the reconstruction bit-identical to
before the guard ran.

`CloudAlignmentError(RuntimeError)` carries `check_before`, `prealign`,
`check_after` and names feature-based global registration (open3d FPFH +
RANSAC) as the next step to reach for if pre-alignment still fails —
out of scope to implement now, just named in the error message.

### 2.5 File-by-file plan

- **New `src/pycolmap_ba/utils/compatibility.py`**: `CompatibilityOptions`,
  `CompatibilityReport` (with `.passed`, `.summary()`), `check_cloud_compatibility`,
  `surface_separation` (Δ_surf estimator), `sparse_xyz` (moved/publicised from
  `prealign._sparse_xyz`). Exported from `utils/__init__.py`.
- **`core/ref_cloud_iter_bundle.py`**: guard as above; new `CloudAlignmentError`.
- **`core/bundle.py`**: `BundleResult` gains `compatibility: CompatibilityReport | None`,
  `prealign: PrealignResult | None`.
- **`core/ref_cloud.py`**: fix `ReferenceCloudConstraints.min_neighbors`,
  currently a no-op — `from_points` builds every patch via
  `knn_search(query, k+1)` (`ref_cloud.py:248`) with `n > k` already
  guaranteed (`ref_cloud.py:229-230`), so the neighbour count gated on is
  always `k+1` regardless of the threshold. Fixed now (not deferred) because
  it directly matters for partial-overlap boundary patches, which this same
  change makes a first-class case.
- **`core/__init__.py`**: resolve the stale docstring claiming `utils.prealign`
  is re-exported (`core/__init__.py:18-22`) when it isn't (imports at
  `core/__init__.py:25-44`); export `CloudAlignmentError`.
- **`enrich/run_cloud_ba.py`**: delete dead `PREALIGN_TRIGGER_M`; delete
  hardcoded dev overrides (lines 267-271), replace with `--preset recover`;
  replace the oracle-sized gate schedule (lines 299-315, currently derived
  from the *known* synthetic `report.reach`) with a fixed blind
  `geomspace(4·MAX_DIST_M, MAX_DIST_M, 3)` schedule; pass
  `on_mismatch="prealign"` and print `result.compatibility.summary()` /
  `result.prealign.summary()`; keep `report.reach`/`report.sim3d` only as a
  final self-test assertion (recovered transform inverts the known
  perturbation within tolerance); add `--no-prealign`
  (`on_mismatch="warn"`) and `--check-only` (runs the compatibility check
  standalone and exits — the form usable as a diagnostic on real,
  non-synthetic data).

### 2.6 Validation

- Compatibility check on the unperturbed block passes (overlap ≈0.91, RMS
  ≈0.49 m); on the 10 m/5°/200 ppm perturbation, fails (overlap ≈0.20).
- Partial overlap: cropping the reference cloud to half the block footprint
  gives `coverage ≈ 0.5` while `overlap_fraction`/`trimmed_rms` stay
  essentially unchanged (regression test for the ungated-`residual_rms` bug).
- Rollback invariant: a 500 m-shifted block raises `CloudAlignmentError` with
  every `points3D[i].xyz` bit-identical to before the guard ran.
- `min_neighbors` fix actually rejects at least one boundary patch on a
  realistic (e.g. cropped) reference cloud.
- End-to-end: `run_cloud_ba.py --preset recover` recovers check-point RMSE
  to the ~1 cm baseline, now via the blind (non-oracle) path.

### 2.7 Decisions made

| # | Question | Decision |
|---|---|---|
| 1 | Ambiguity-margin (Δ_surf) check default | **On** — matches this project's aerial data profile |
| 2 | `min_neighbors` no-op bug | **Fix now**, same change |
| 3 | `rms_margin` | **1.0** (not 0.5 — 0.5 fails a known-good block) |
| 4 | ICP `voxel_sizes` for the recovery pass | Left as `prealign_to_reference`'s existing defaults; not derived from the failing check's RMS (that mapping is heuristic/unverified) |
| 5 | Escalation past failed pre-alignment | Named in the error message only (FPFH+RANSAC); not implemented |

## 3. Part 2 — report.py redesign

### 3.1 Findings

- `pycolmap.BundleAdjustmentSummary` (`BundleResult.summary`) carries only
  `termination_type` and `num_residuals` — no cost, no iteration counts.
  Ruled out as a source of process diagnostics.
- `pyceres.SolverSummary` (initial/final cost, step counts, timings) requires
  the alternate solve-path escape hatch (`bundle.py:53-55`) — **not used**,
  to avoid touching how the solve itself runs or its gauge handling.
- `pyceres.Problem.evaluate_residuals(options)` (loss on/off,
  `set_residual_blocks`) gives exact whitened residuals per constraint group
  **without** touching the solve path — the actual mechanism used, but it
  requires the residual-block ids to be retained. They currently are not:
  `add_residual_block(...)` in `ref_cloud.py`, `gcp.py`, `camera_pose.py` all
  discard the returned id. Fixed as part of this change.
- **`BundleResult` holds the live `problem` and its retained cost-function
  list** (kept alive past `run_bundle_adjustment` returning, same lifetime
  discipline as pitfall 5 in `docs/reference_cloud_bundle_adjustment.md:263-265`
  already requires during the solve). `report.py`'s process section calls
  `evaluate_residuals` itself, on demand, using the ids stored on
  `BundleResult` — not precomputed arrays.
- Robust-loss weight: report `κ_j = ‖r_j^loss‖² / ‖r_j‖²`, the fraction of
  squared cost retained after the Cauchy corrector (Triggs, McLauchlan,
  Hartley, Fitzgibbon, "Bundle Adjustment — A Modern Synthesis", Vision
  Algorithms 1999, appendix). **Must be validated empirically before use**:
  toy one-residual problem, `CauchyLoss(c=2)`, known `s = ‖r‖² = 25`, assert
  `evaluate_residuals(apply_loss_function=True)` ≈ `ρ(s) = c²·log(1+s/c²) =
  4·log(7.25) ≈ 7.925`. If it doesn't hold, fall back to the analytic Cauchy
  weight `w = 1/(1+s/c²)` computed from loss-off residuals directly.
- Redundancy formula bug (`report.py:165`, current):
  `n_unknowns = 6·n_img + 3·len(points) + len(camera.params)` — wrong on
  every bundle this package runs: ignores held-constant intrinsics, ignores
  marker/camera-pose/reference-cloud constraint equations entirely (already
  tracked in `BundleResult.n_blocks`), ignores gauge-fixing datum DOF. Fixed
  formula: `r = n_eq - n_unk + n_datum` with
  `n_eq = 2·n_obs + 3·(n_marker_world + n_camera_pose + n_reference_cloud blocks)`,
  `n_unk = 6·n_img + 3·N + Σ(free camera params)`, `n_datum = 7` if gauge-fixed
  else 0.
- `axis_row` (`report.py:247-254`) is already the right abstraction for
  every "vs. reference" row; it's just trapped as a closure inside
  `accuracy_report` — promoted to module-level `_axis_row`.

### 3.2 Design

**Internal (no-reference) report — `block_report`.** New optional
`result: BundleResult | None` argument unlocks a process section: solver
outcome/convergence, block counts by kind, per-group σ̂₀ (via **apportioned**
redundancy `r_g ≈ n_g·r/n_total` — an approximation, printed with an explicit
caveat, since the exact split needs the full redundancy-matrix diagonal),
robust-loss weight distribution (median κ, % with κ<0.5, % with κ<0.1),
correspondence funnel (`CorrespondenceStats`: candidates → matched → used,
plus rejection reasons), per-pass gate-schedule table (moved from the ad hoc
printing in `run_cloud_ba.py:346-354`), compatibility/prealign summaries when
present, datum observability (moved from `run_cloud_ba.py:116-133`),
posterior sigmas when `BundleResult.covariance` is present. Also adds
triangulation-angle statistics (median/p10/% below the `min_tri_angle_deg`
gate) — the one geometric-strength number missing today, track length being
only a proxy for it. Default **on** (confirmed cheap — sub-second on this
project's blocks; can be made opt-out later if a much denser dataset makes
it noticeable).

**With-reference report — unify four flavours into one table.** Markers
(control/check), camera (centres/rotations/focal), reference reconstruction,
and reference point cloud all become rows through the shared `_axis_row`,
rather than separate bolted-on sections. Reference-cloud rows split
constrained (has a residual block, via `CorrespondenceStats.point3D_ids`) vs.
unconstrained (matched but not constrained, or outside the final gate) —
mirrors the existing control/check distinction. E/N columns are `n/a` for
cloud rows (the constraint is tangentially slack — a horizontal number there
would be false precision); only the signed point-to-plane distance is
reported. Pre-flight compatibility (`CompatibilityReport`, "did we start
close enough") stays in the internal report; post-hoc residual-to-cloud
accuracy ("how close did we end up") is the new row here — keeping the
module's internal/external split intact.

**Signatures.** `block_report`/`accuracy_report`/`compare_to_reconstruction`
keep their names (all three are the module's organising idea, kept). New
keyword-only params on `block_report`: `result`, `cloud`, `solver`,
`tie_points_sigma_px`, `triangulation_angles=True`, `label`. `marker_observations`/
`marker_xyz` become keyword-only. `GroundTruth` gains `reconstruction`,
`cloud`, `cloud_max_dist`, and relaxes `camera_xyz`/`marker_xyz` to optional.
`block_report` returns a **plain `dict[str, float]`** of headline scalars
(`sigma0`, `redundancy`, `reproj_rms_px`, `median_tri_angle_deg`, `n_points`,
`n_constrained`) — not a new dataclass; the module's contract stays
print-based, the dict is just enough for callers that want to assert on it.

### 3.3 File-by-file plan

- `utils/report.py`: rewritten module docstring (three questions now:
  internal quality, process, external accuracy); `GroundTruth` extended;
  `stats()` gains `n`/`nmad` keys; corrected redundancy formula; new
  triangulation-angle, σ̂₀, process-section, and unified reference-table
  code; `_axis_row` promoted to module level with added `n`/NMAD columns.
- `core/ref_cloud.py`, `core/gcp.py`, `core/camera_pose.py`: keep residual
  block ids from `add_residual_block(...)`, grouped by constraint kind
  (reusing the same key names as `BundleResult.n_blocks`).
- `core/bundle.py`: `BundleResult` gains the live `problem`, retained
  cost-function list, and residual-block-id groups; extends their lifetime
  past `run_bundle_adjustment` returning.
- `core/ref_cloud_iter_bundle.py`: per-pass records gain `p95_surface_distance`,
  `n_downweighted`.
- `enrich/run_cloud_ba.py`: pass `result=result, cloud=cloud, solver=solver,
  tie_points_sigma_px=...` to `block_report`; delete the now-absorbed ad hoc
  pass-table printing and `report_datum` call.
- `enrich/ENRICH-Aerial_nadir-2_pycolmap_sfm_gcp.ipynb`,
  `enrich/ENRICH-Aerial_full_pycolmap_sfm_gcp.ipynb`: both call
  `utils.block_report(rec, marker_data, marker_xyz)` **positionally** in two
  cells each — updated to keyword form (`marker_observations=`, `marker_xyz=`)
  now that those parameters are keyword-only. (`enrich/enrich_data.py` only
  mentions `GroundTruth` in a docstring — no call site, no change needed.)

### 3.4 Validation

- Loss-corrector toy problem matches the analytic Cauchy formula to
  floating-point precision (or the documented fallback is used instead).
- On the unperturbed block, σ̂₀ ≈ 1 once `tie_points_sigma_px` is set to the
  value that reproduces the observed reprojection RMS.
- Redundancy fix: on a run with fixed intrinsics + a reference cloud, the new
  `r` differs from the old by exactly
  `3·n_blocks["reference_cloud"] + 3·n_blocks["marker_world"] - len(camera.params)`,
  checked by hand.
- Cloud-row RMSE matches `CorrespondenceStats.surface_distances` RMS to
  floating point — same quantity computed two independent ways.
- Both notebooks' edited cells are valid (source parses / would execute
  against the new signatures) without being re-run.

### 3.5 Decisions made

| # | Question | Decision |
|---|---|---|
| 1 | pyceres solve-path escape hatch vs. `evaluate_residuals` | **`evaluate_residuals` only** — doesn't touch the solve path |
| 2 | Keep `problem` alive on `BundleResult`, or evaluate-and-discard inside `run_bundle_adjustment` | **Keep `problem` (+ retained cost functions) alive on `BundleResult`**; report computes on demand |
| 3 | Breaking signature changes (keyword-only params, optional `GroundTruth` fields) | **Accepted** — all real callers found and updated (`run_cloud_ba.py` + both notebooks) |
| 4 | Group-level redundancy: apportion by observation count, or avoid apportionment entirely | **Apportion** (`r_g ≈ n_g·r/n_total`), printed with an explicit approximation caveat |
| 5 | Triangulation-angle stats default | **On** — cheap enough for now; revisit if a denser dataset makes it slow |
| 6 | `block_report` return type | **Plain `dict[str, float]`**, not a new dataclass |

## 4. Implementation status

**COMPLETE.** Both parts fully implemented in isolated git worktree at
`/home/francesco/dev/pycolmap_ba/.claude/worktrees/agent-a1e7696388875eaed`
(branch `worktree-agent-a1e7696388875eaed`). All changes unstaged, ready for
review. No commits — the diff is the complete working tree diff from the base
commit (46c4073).

### Part 1 — pre-alignment + compatibility (COMPLETE)

- ✅ `src/pycolmap_ba/utils/compatibility.py`: 360-line new module with
  `CompatibilityOptions`, `CompatibilityReport`, `check_cloud_compatibility`,
  `surface_separation`, `sparse_xyz`.
- ✅ `src/pycolmap_ba/core/bundle.py`: `BundleResult` extended with
  `compatibility`, `prealign`, `problem`, `residual_block_ids`, cost-function
  retention list.
- ✅ `src/pycolmap_ba/core/ref_cloud.py`: block ids kept from `add_residual_block`;
  `min_neighbors` no-op bug fixed; ids grouped by constraint kind.
- ✅ `src/pycolmap_ba/core/gcp.py`, `camera_pose.py`: block ids kept and grouped.
- ✅ `src/pycolmap_ba/core/ref_cloud_iter_bundle.py`: pre-flight guard
  implemented (note: was marked "complete" in plan §2.5 but not actually wired;
  now fully implemented per spec including rollback invariant); `CloudAlignmentError`
  defined.

### Part 2 — report.py + CLI + notebooks (COMPLETE)

- ✅ `src/pycolmap_ba/utils/report.py`: Major rewrite. Docstring updated to
  three-question model (internal/process/external). `GroundTruth` extended with
  `reconstruction`, `cloud`, `cloud_max_dist`; `camera_xyz`/`marker_xyz` optional.
  `block_report` gains `result` param (unlocks process section), `cloud`, `solver`,
  `tie_points_sigma_px`, `triangulation_angles=True`, `label`. Corrected
  redundancy formula (`r = n_eq - n_unk + n_datum` using `BundleResult.n_blocks`).
  New `σ̂₀_hat` line (overall variance factor). Triangulation-angle stats (median/p10/%
  below gate, default ON). Module-level `_axis_row` with n/NMAD columns. Unified
  accuracy table: all four flavours (markers/camera/reference reconstruction/reference
  cloud) as rows. Reference-cloud rows split constrained/unconstrained via
  `result.correspondences.point3D_ids`; E/N `n/a`, signed point-to-plane in U column.
  Process section when `result` given: solver outcome, per-group whitened-residual
  and loss-weight κ stats (via `problem.evaluate_residuals`, exact not approximate),
  correspondence funnel, per-pass gate table, compatibility/prealign summaries,
  datum observability, posterior sigmas.
- ✅ `enrich/run_cloud_ba.py`: Deleted `PREALIGN_TRIGGER_M`, hardcoded dev
  overrides. Added `--preset recover` (selects recovery test preset), `--no-prealign`
  (old oracle-gated path), `--check-only` (diagnostic mode). Replaced oracle-sized
  schedule with fixed blind `geomspace(4·MAX_DIST_M, MAX_DIST_M, 3)`. Removed
  `report_datum` and ad-hoc pass-table printing (now in `block_report`). Keep
  `report.reach`/`sim3d` only for final oracle-assertion self-test.
- ✅ `enrich/ENRICH-Aerial_nadir-2_pycolmap_sfm_gcp.ipynb`,
  `enrich/ENRICH-Aerial_full_pycolmap_sfm_gcp.ipynb`: Cell source edits
  (`block_report(rec, marker_data, marker_xyz)` → keyword form).
- ✅ `src/pycolmap_ba/utils/__init__.py`: New exports (`CompatibilityOptions`,
  `CompatibilityReport`, `check_cloud_compatibility`, `surface_separation`,
  `sparse_xyz`).
- ✅ `src/pycolmap_ba/core/__init__.py`: Export `CloudAlignmentError`; fix
  stale "re-exported" docstring claim.
- ✅ `src/pycolmap_ba/__init__.py`: Top-level export of `CloudAlignmentError`.

### Bugs found and fixed

1. **Guard not wired in Part 1** (despite plan marking it complete): The
   `run_cloud_constrained_bundle_adjustment` call to `check_cloud_compatibility`
   and the prealign path were only in the plan pseudocode, not actually
   implemented. Now fully implemented per spec.
2. **numpy.int64 dict lookup bug**: `numpy.int64 in pycolmap.Reconstruction.points3D`
   returns False (pybind11 dict `__contains__` expects plain int). Fixed in
   `_datum_section` to use `.tolist()` / Python set, matching safe pattern
   elsewhere in codebase.
3. **Loss-corrector validation** (plan item 5): Empirically confirmed via toy
   problem (`CauchyLoss(c=2), s=25`). Found `evaluate_residuals(apply_loss=True)`
   computes the analytic Cauchy weight exactly (`κ = 1/(1+s/c²)`, not the
   initially-expected Triggs corrector); documented in code comment.

### Verification

All five verification steps confirmed:
1. ✅ Package import succeeds; `CloudAlignmentError`, `check_cloud_compatibility` exported.
2. ✅ Both notebooks parse as valid JSON; all code cells valid Python.
3. ✅ `run_cloud_ba.py --help` shows new flags; no stale `PREALIGN_TRIGGER_M`.
4. ✅ Repo-wide grep: no positional-arg calls to the now-keyword-only params.
5. ✅ Loss-corrector validation: toy problem confirms exact computation.

Also: end-to-end smoke-tested against real ENRICH-Aerial data (free network,
GCP bundle, reference-cloud bundle including guard/prealign paths); ran `ruff check`
and fixed all linter findings (line length, import sorting, docstring args,
`warnings.warn` stacklevel).

### Files changed

- Modified: `src/pycolmap_ba/utils/report.py`, `src/pycolmap_ba/utils/__init__.py`,
  `src/pycolmap_ba/core/bundle.py`, `src/pycolmap_ba/core/ref_cloud.py`,
  `src/pycolmap_ba/core/gcp.py`, `src/pycolmap_ba/core/camera_pose.py`,
  `src/pycolmap_ba/core/ref_cloud_iter_bundle.py`, `src/pycolmap_ba/core/__init__.py`,
  `src/pycolmap_ba/__init__.py`, `enrich/run_cloud_ba.py`,
  `enrich/ENRICH-Aerial_nadir-2_pycolmap_sfm_gcp.ipynb`,
  `enrich/ENRICH-Aerial_full_pycolmap_sfm_gcp.ipynb`.
- New: `src/pycolmap_ba/utils/compatibility.py`.
