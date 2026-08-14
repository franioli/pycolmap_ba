# Reference-cloud constrained bundle adjustment — the logic

Reference note for
[`src/pycolmap_ba/backend/reference_cloud.py`](../src/pycolmap_ba/backend/reference_cloud.py)
and its validation in
[`enrich/run_cloud_ba.py`](../enrich/run_cloud_ba.py) (ENRICH-Aerial, nadir-2).

Companion to [`gcp_bundle_adjustment.md`](gcp_bundle_adjustment.md), whose
weighting convention (§3–4) applies here unchanged.

---

## 1. What the constraint is for

An already georeferenced and scaled LiDAR / TLS survey is a third kind of anchor,
alongside markers and pose priors. Every triangulated tie point should lie on the
surface the scanner measured. Turning that into residuals buys two things:

| | Effect | Mechanism |
|---|---|---|
| 1 | **Datum** — 7-DoF fix without a GCP survey | the cloud is already in the target frame at the target scale |
| 2 | **Deformation / self-calibration** | a wrong focal or distortion bends the sparse cloud into a dome; the reference surface is deformation free, so the residual field *is* the correction signal |

(2) is the interesting one. `SolverOptions.refine_focal_length` is documented as
degenerate on a flat nadir block — focal trades against scene depth. A reference
surface breaks that trade, because a focal error does not merely rescale the
block, it *curves* it, and the cloud sees the curvature.

---

## 2. The residual

### Why not nearest-neighbour point-to-point

A LiDAR cloud **samples** a surface. The correspondent of a tie point lies *on*
that surface and is essentially never one of the samples. Snapping to the nearest
sample therefore

- biases every point by ~half the point spacing, and
- pulls it **tangentially**, along the surface,

which is exactly the systematic lateral distortion the intrinsics are supposed to
be cleaned of. Point-to-point makes (2) above worse, not better.

### The GICP form

Each reference point carries the distribution of its neighbourhood instead, and
the residual is the Mahalanobis distance to it (Segal 2009 GICP, Magnusson 2009
NDT). Following GICP, the measured eigenvalues decide *which* directions are
free; they are never used as variances directly:

$$
\Sigma_j = V\operatorname{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2)V^\top,
\qquad
\sigma_i = \sigma_{\min}\left(\frac{\sigma_{\text{slack}}}{\sigma_{\min}}\right)^{t_i}
$$

$$
t_i=\sqrt{\frac{\lambda_i-\lambda_1}{\lambda_3-\lambda_1}}\in[0,1],
\qquad
\sigma_{\min}=\sqrt{\sigma_{\text{ref}}^2+\lambda_1}
$$

with $\lambda_1\le\lambda_2\le\lambda_3$ the neighbourhood covariance
eigenvalues, so $t_1=0$ (across the surface — always stiffest) and $t_3=1$
(most spread-out direction — always freest).

| Patch | $\lambda$ spectrum | $\sigma$ assigned | Behaviour |
|---|---|---|---|
| plane | $(\epsilon,\ell,\ell)$ | $(\sigma_{\min},\sigma_{\text{slack}},\sigma_{\text{slack}})$ | point-to-plane, free to slide |
| breakline | $(\epsilon,\epsilon,\ell)$ | $(\sigma_{\min},\sigma_{\min},\sigma_{\text{slack}})$ | stiff in 2, free along the line |
| blob (vegetation) | $(\ell,\ell,\ell)$ | all slack | contributes nothing |

**Do not use $\lambda_i$ as the variance directly.** It measures the *patch
radius* — about 0.1 m for a 20-neighbour patch — not the freedom to slide along
the surface, so the tangential pull comes straight back. Measured on a synthetic
plane: a 0.5 m tangential displacement cost **10.1** under the raw-$\lambda$ form
and **0.05** under the form above, a 200× difference.

Because a planar patch really is soft tangentially, the correspondence only has
to find the right **surface patch**, not the right **point**. That is what makes
rare refresh viable.

---

## 3. Why no C++ was needed

`pyceres.factors.NormalPrior(mean, cov)` evaluates
$\operatorname{chol}(\Sigma^{-1})^\top(x-\mu)$ — which *is* the residual above,
in C++. Verified numerically:

```
n=[0,0,1], sigma_n=0.02, sigma_t=10; point 0.10 m off-plane, (+0.5,+0.7) in-plane
0.5|r|^2 = 12.5037      point-to-plane 0.5*(0.10/0.02)^2 = 12.5    (in-plane leak 0.0037)
```

This matters: `gcp_bundle_adjustment.md` §8 records that a **Python** cost
function holds the GIL and serialises Ceres's multithreaded pass, turning a 5 s
solve into a 30 min timeout. The same C++-factor discipline as the marker world
priors applies.

The parameter blocks are shared the same way as everywhere else in this package:
`Point3D.xyz` is a live `OWNDATA=False` view of COLMAP's memory, so
`problem.add_residual_block(cost, loss, [point.xyz])` binds to the block COLMAP
is already optimising. Nothing is added to the reconstruction.

---

## 4. Correspondence: found once per pass, never inside the solve

Ceres's trust region assumes a **fixed** cost function within a solve. Mutating
correspondences between evaluations makes cost values non-comparable across
steps, so step quality is meaningless and the solver oscillates. It is also
unnecessary — after the first pass the points move by centimetres.

`run_cloud_constrained_bundle_adjustment` uses ICP's alternating (E/M) split
instead: freeze correspondences → solve to convergence → re-search → repeat. A
fresh adjuster per pass also preserves the single-adjuster invariant of
`run_bundle_adjustment`.

Convergence is tested on the **correspondence assignment**, not the cost: the
fraction of tie points whose matched patch changed. Measured on nadir-2:
100 % → 9.4 % → 1.5 % → 1.1 %.

### The statistic that lies

Report the distance **to the local plane**, not to the matched sample. The
sample distance has a floor of roughly half the point spacing even for a tie
point lying exactly on the surface. On nadir-2 it sat at 0.213 m for every pass
and showed no convergence at all, while the point-to-plane distance went
0.025 → 0.011 m. `CorrespondenceStats` carries both, and `summary()` leads with
the plane distance.

---

## 5. Datum observability — measure it, don't assume it

A flat scene constrains height, roll and pitch and leaves easting, northing and
yaw free, **however dense the cloud**. What breaks the degeneracy is *normal
diversity*: façades, sloped roofs, kerbs. Density does nothing. Corners do not
help much either — a point-sampled patch at a corner has an isotropic spectrum,
which the formula above correctly treats as uninformative.

So `datum_information` measures it. For each constraint with whitener
$A_j=\operatorname{chol}(\Sigma_j^{-1})$, the Jacobian with respect to a 7-DoF
similarity about the block centroid $c$ is

$$
J_j = A_j\left[\,I \;\; -[X_j-c]_\times \;\; (X_j-c)\,\right]/L,
\qquad H=\sum_j J_j^\top J_j
$$

with $L$ the RMS distance from the centroid, so all seven columns are in metres
and $1/\sqrt{\lambda(H)}$ reads directly as "how far the block could still move,
at its edge, in this mode".

Synthetic check:

| Scene | weak modes | strong modes |
|---|---|---|
| plane only | tx, ty, rz, scale ≈ 30 mm | tz, rx, ry ≈ 0.4 mm |
| plane + 2 façades | — | all seven ≤ 18 mm, most < 1 mm |

A 70× separation on the flat scene, correctly flagged. Run this before trusting
a cloud-only datum, and keep `solver.gauge` or camera priors if a mode is weak.

---

## 6. Gating

Tie points sit on texture corners, which on an aerial block may be canopy,
moving objects or roof edges rather than the surface the scanner saw. The gates
are load-bearing:

| Gate | Purpose |
|---|---|
| `max_dist` | reject wild correspondences. Set it from the **initial misalignment**, not the target accuracy — a gate tighter than the starting error leaves the block unable to pull itself in |
| `max_roughness` | the main clutter filter. A façade is millimetres rough, vegetation decimetres |
| `min_track_length` | short tracks are poorly triangulated and would import their own error into the datum |
| `min_neighbors` | the covariance of a sparse patch is meaningless |
| `robust_loss_scale` | Cauchy, in whitened units. Without it a single gross correspondence drags the datum |
| `max_normal_angle_deg` | optional; drops grazing / occluded matches. Costs a per-track loop, off by default |

`min_planarity` exists but is usually the wrong tool: breaklines are stiff in two
directions and worth keeping, and planarity does *not* protect the horizontal
datum — normal diversity does.

---

## 7. Weighting

Unchanged from `gcp_bundle_adjustment.md` §4: COLMAP adds its tie-point
residuals unweighted (implicit $\sigma=1$ px), so the whole objective is
multiplied by $\sigma_{\text{tie}}^2$ and every other covariance divided by it.
`covs_in_tie_units` is the matrix counterpart of `cov_in_tie_units`, since a
GICP covariance is not diagonal.

$\sigma_{\text{ref}}$ should be the **real** accuracy of the reference cloud.
Setting it tighter than the survey achieved does not make the block better, it
forces the block onto survey noise — and, because $\sigma_{\text{ref}}$ also
controls the noise floor of the patch normals, it makes the tangential leak
worse.

$\sigma_{\text{slack}}/\sigma_{\text{ref}}$ is the dynamic range of the
constraint. ~500–1000 is comfortable; 1e6 makes the covariance ill-conditioned
for no benefit.

---

## 8. Results (ENRICH-Aerial nadir-2)

Free network aligned to the noisy camera priors, then adjusted with **no ground
control at all**; the reference cloud is the GT mesh sampled to 5 M points with
2 cm Gaussian noise. All 26 markers are check points.

| | check-point 3D RMSE | camera-centre RMSE |
|---|---|---|
| GCP bundle (18 control, 8 check) | 1.29 cm (n=8) | 1.82 cm |
| reference-cloud bundle, **no control at all** | **1.02 cm** (n=26) | 1.78 cm |

The cloud-only datum is at least as good as the GCP-controlled one, on three
times as many check points — which is the expected outcome, since the cloud
supplies orders of magnitude more control than 18 surveyed markers.

### Self-calibration

The focal length was deliberately scaled by +2 % (5865.181 → 5982.485 px) before
adjusting. The control run — same perturbation, refinement disabled — shows what
that error costs and proves the perturbation was real:

| `--perturb-focal 2` | recovered focal | check points | camera centres |
|---|---|---|---|
| refinement **off** (control) | 5982.485 (+117.304) | 32.08 cm | 318.61 cm |
| refinement **on** | 5864.874 (**−0.307**) | **1.10 cm** | 1.75 cm |

A 117.3 px focal error is recovered to 0.31 px — 0.005 % — on a **flat nadir
block**, where the focal is otherwise degenerate against depth. That degeneracy
is visible in the control row: with the points held down by the cloud and the
focal frozen wrong, the cameras absorb the error by flying 3.18 m upward.

Note the camera-centre residual is *not* an independent check here — the block
was pre-aligned to the (noisy) prior centres. The 26 check points are.

Per pass: ~8–14 s on 221 660 tie-point blocks + ~17 500 cloud blocks. The
correspondence search itself is a rounding error (5 M-point index 0.3 s, 43 k
queries 0.02 s).

Caveat: this reference cloud is *synthetic* — perfectly registered, no
occlusion, no vegetation, no moving objects, and derived from the same model that
generated the images. It answers "is the estimator right", not "will this survive
real data".

---

## 9. Pitfalls

1. **Do not use the raw eigenvalues as variances** (§2). Silently reintroduces
   tangential pull; the block still converges and still looks fine.
2. **Do not report point-to-sample distance as fit quality** (§4). It is
   dominated by the point spacing and hides both convergence and divergence.
3. **Do not refresh correspondences inside the solve** (§4).
4. **Do not assume the datum is fixed because many points are constrained**
   (§5). Density is not diversity.
5. **Keep the cost functions alive.** pyceres does not take ownership; the
   `retained` list is why this does not segfault.
6. **One adjuster.** Building a second one over the same config orphans every
   block added here and the solve silently runs on tie points alone. The
   `num_residual_blocks` assertion in `run_bundle_adjustment` guards it.
7. **Align before constraining.** Bundle adjustment is a local optimiser. From
   an arbitrary free-network datum the cloud is nowhere near the sparse points
   and every correspondence is noise. Fit the block to prior camera centres (or
   a handful of markers) first, then constrain.

Checks 1–4 and the two datum cases in §5 are automated in
[`examples/check_reference_cloud.py`](../examples/check_reference_cloud.py);
run it after touching the residual.

---

## 10. Not implemented (and why)

| Approach | Appeal | Cost |
|---|---|---|
| **SDF / distance field** — residual = interpolated `SDF(X)`, smooth, analytic gradient, *no correspondence at all* | theoretically cleanest; nothing to refresh | the one variant that genuinely needs C++ (`ceres::Grid3D` + `CubicInterpolator`) |
| **Point-to-mesh** via `open3d` `RaycastingScene.compute_closest_points` | exact surface distance, zero sampling bias | needs a meshable reference; awkward for raw noisy TLS |
| **Image-space depth residuals** — ray-cast the reference into each image once, constrain per-observation depth | couples to intrinsics even more directly | correspondence must follow the poses |

---

## 11. References

- A. Segal, D. Haehnel, S. Thrun, *Generalized-ICP*, RSS 2009.
- M. Magnusson, *The Three-Dimensional Normal-Distributions Transform*, PhD
  thesis, Örebro, 2009.
- M. Weinmann et al., *Semantic point cloud interpretation based on optimal
  neighborhoods*, ISPRS J. 2015 (the planarity / linearity features).
