# Posterior covariance — scaling, propagation, and what it means

Reference note for
[`src/pycolmap_ba/core/covariance.py`](../src/pycolmap_ba/core/covariance.py),
validated by
[`sandbox/examples/check_covariance.py`](../sandbox/examples/check_covariance.py).
Companion to
[`gcp_bundle_adjustment.md`](gcp_bundle_adjustment.md) (the weighting
convention this note assumes throughout) and
[`reference_cloud_bundle_adjustment.md`](reference_cloud_bundle_adjustment.md).

---

## 1. `pycolmap.BACovariance` is a cofactor matrix, not a covariance

`pycolmap.estimate_ba_covariance_from_problem` (`estimate_covariance` here)
evaluates the Jacobian `J` of the solved Ceres problem, forms `H = J^T J`,
Schur-eliminates the requested block, and inverts. Read directly from
`colmap/src/colmap/estimators/covariance.cc`: no residual, no sigma, no
redundancy anywhere in that computation. What it returns is the **cofactor
matrix**

$$
Q_{xx} = (A^\top P A)^{-1}
$$

which is the *correct* covariance only if the weight matrix `P` used the
*true* observation variances. In practice the assumed sigmas
(`tie_points_sigma_px` and friends, see §4 of `gcp_bundle_adjustment.md`) are
a guess, and the honest posterior covariance is

$$
\hat\Sigma_{xx} = \hat\sigma_0^2\, Q_{xx}, \qquad
\hat\sigma_0^2 = \frac{v^\top P v}{r}, \quad r = n_{eq} - n_{unk} + n_{datum}
$$

the classical variance-component estimate of Gauss-Markov adjustment theory
(Förstner & Wrobel, *Photogrammetric Computer Vision*, Springer 2016,
§2.4/§4.2; Koch, *Parameter Estimation and Hypothesis Testing in Linear
Models*, 2nd ed., Springer 1999, ch. 3). `variance_factor()` computes
$\hat\sigma_0$; every `*_covariances()` function multiplies the raw matrix by
`sigma0 ** 2` before returning it, unless `scale_by_sigma0=False` is passed
explicitly.

**Never feed the raw matrix into a metric export as if it were already in
metres.** It is the cofactor for an *assumed* `sigma_tie = 1 px`, and reading
it directly understates the true uncertainty by a factor of `1 / sigma0_hat`
- roughly +25% at `sigma0_hat ≈ 0.8`, dramatically more the further the
assumed sigmas were from the truth.

---

## 2. Why no separate unit conversion is needed

`core.covariance.cov_in_tie_units` scales the whole objective by
`sigma_tie^2` so COLMAP's own unweighted tie-point blocks match exactly (see
`gcp_bundle_adjustment.md` §4). This changes the residuals and Jacobian the
solved problem actually sees:

$$
r_{\text{code}} = \sigma_{\text{tie}}\, r_{\text{true}}, \qquad
J_{\text{code}} = \sigma_{\text{tie}}\, J_{\text{true}}
$$

so the raw cofactor and the code's own $\hat\sigma_{0,\text{code}}$ pick up
compensating factors of $\sigma_{\text{tie}}$:

$$
Q_{\text{code}} = \left(J_{\text{code}}^\top J_{\text{code}}\right)^{-1}
                 = \frac{Q_{\text{true}}}{\sigma_{\text{tie}}^2}, \qquad
\hat\sigma_{0,\text{code}}^2 = \frac{\sum r_{\text{code}}^2}{r}
                 = \sigma_{\text{tie}}^2\,\hat\sigma_{0,\text{true}}^2
$$

Multiplying them back together, the $\sigma_{\text{tie}}$ factors cancel
exactly:

$$
\hat\sigma_{0,\text{code}}^2 \cdot Q_{\text{code}}
  = \hat\sigma_{0,\text{true}}^2 \cdot Q_{\text{true}} = \hat\Sigma_{xx}
$$

**So the number `variance_factor(...).sigma0` returns is exactly the right
scale factor to square and multiply into COLMAP's raw matrix — no further
unit conversion, regardless of what `tie_points_sigma_px` was.** This is
`sandbox/examples/check_covariance.py`'s strongest check: solving the same
block at `tie_points_sigma_px = 0.5` and `1.0` gives raw `Q` differing by
`4x` and identical scaled `Sigma_hat`.

Scalar sanity check (canonical vs. this package's convention), for `n`
observations of one unknown with assumed sigma $\sigma_a$:

| | canonical (`P = 1/sigma_a^2`) | code convention (`sigma_a = sigma_tie`) |
|---|---|---|
| `Q` | $\sigma_a^2/n$ | $1/n$ |
| $\hat\sigma_0^2$ | $s^2/\sigma_a^2$ | $s^2$ |
| product | $s^2/n$ | $s^2/n$ | ✓ same

---

## 3. `sigma0_hat`: exact residual sum, not `Point3D.error`

COLMAP never hands back the tie-point residual-block ids — they are added and
owned entirely inside `create_default_ceres_bundle_adjuster` — so an early
version of this diagnostic approximated the tie-point sum of squares as
`sum(track_length * Point3D.error**2)`. Two problems with that:

1. **`Point3D.error` is the mean, not the RMS**, per-observation reprojection
   error (`observation_manager.cc`: `error_sum / track.Length()`). For 2-D
   isotropic Gaussian image noise, `|r|` is Rayleigh-distributed, so
   $(\mathbb E|r|)^2 / \mathbb E|r|^2 = (\pi/2)/2 = \pi/4 \approx 0.785$ - the
   mean-based sum understates the true sum of squares by exactly that
   Jensen-gap factor, biasing $\hat\sigma_0$ low by $\sqrt{\pi/4}\approx
   11\%$ and the covariance scale low by $\approx 21\%$.
2. **It is stale unless a reprojection filter ran after the solve** - that
   assignment is the only place COLMAP updates `Point3D.error`.

`_tie_point_residual_sum_sq` fixes both by reprojecting every observation
directly: `image.cam_from_world() * xyz` then `camera.img_from_cam(..., True)`,
summing `|uv - xy|^2` and excluding cheirality failures. Checked on a
synthetic block against the old approximation: ratio `4/pi = 1.273` (predicted)
vs. `1.267` observed - see check 4 in the validation script.

---

## 4. Camera-centre propagation: left convention, factor of 2

`camera_pose_covariances` returns the **tangent-space** covariance of
`rig_from_world` in COLMAP's `ProductManifold(EigenQuaternionManifold,
Euclidean/Subset(3))` (`bundle_adjustment_ceres.cc`), order
`[rotation, translation]`. Converting that to a *world-axis* covariance of the
camera centre $C = -R^\top t$ needs the exact tangent convention, which is
**not** the SE(3) exponential (so `Rigid3d::AdjointInverse` is the wrong tool
here, even though `rigid3.h` documents COLMAP as left-convention SE(3)
elsewhere):

$$
R' = \exp([2\,\delta\omega]_\times)\, R \quad\text{(left, half-angle)}, \qquad
t' = t + \delta t \quad\text{(plain Euclidean)}
$$

Verified directly (not from documentation) with a one-block `pyceres` problem
under `EigenQuaternionManifold`, comparing the Jacobian Ceres reports against
candidate conventions:

```
ceres tangent jacobian:  [[-0.,      -0.458774, -1.22563 ], ...]
right  -R[v]x         :  [[ 0.46183,  0.463518,  0.004945], ...]   WRONG
left   -[Rv]x         :  [[-0.,      -0.229387, -0.612815], ...]   exact x2
```

Differentiating $C(\delta\omega,\delta t) = -R'^\top t'$ at the origin:

$$
\delta C = -2\,R^\top[t]_\times\,\delta\omega \;-\; R^\top\,\delta t
\qquad\Longrightarrow\qquad
J = -R^\top\big[\,2[t]_\times \;\big|\; I\,\big]
$$

$$
\Sigma_C = R^\top\Big(4\,[t]_\times\,\Sigma_{\omega\omega}\,[t]_\times^\top
                     + 2\,[t]_\times\,\Sigma_{\omega t}
                     + 2\,\Sigma_{\omega t}^\top\,[t]_\times^\top
                     + \Sigma_{tt}\Big) R
$$

implemented in `_camera_center_cov_from_pose_cov`. **The cross term matters.**
An earlier version of this report took only `cov[-3:, -3:]` (the translation
sub-block) and dropped the rest. Because `tr(R^\top\Sigma R) = tr(\Sigma)`,
that sub-block's `sqrt(mean(diag))` is only correct if the rotation were
noiseless; with `|t|` of a few hundred metres and `sigma_omega ~ 1e-5 rad`,
the `[t]_x Sigma_ωω [t]_x^T` and cross terms are millimetre-to-centimetre
scale — the same order as the translation term — and the sign of the total
is not determined by the translation term alone. Dropping them is not a
conservative approximation.

Validated by Monte Carlo (30k samples of the *exact* `Plus`, fixed pose,
random SPD `Sigma`, compare sample covariance of `C` to `J Sigma J^T`):
agreement to ~1.5%, at the expected Monte-Carlo noise floor for that sample
size — check 3 in the validation script.

**Also:** `camera_pose_covariances` returns `None`, not a mis-sliced matrix,
for any image whose pose block is not a full free 6x6 - the
`TWO_CAMS_FROM_WORLD` gauge fixes image 1 entirely (`get_cam_cov_from_world`
returns `None` from pycolmap itself) and gives image 2 a `SubsetManifold(3,
{fixed_dim})`, i.e. a 5x5 matrix that an earlier version silently mis-sliced
as `cov[-3:, -3:]`.

---

## 5. Conditional points, marginal markers - not the same statistic

`point_covariances` reads `BACovariance.get_point_cov`, computed as
`(H_pp + lambda I)^-1` on the point block *conditional on every pose,
intrinsic and other point held fixed* (the header says so explicitly).
`marker_covariances` reads `get_other_params_cov`, which lands markers in the
reduced system `S = H_aa - H_ap H_pp^-1 H_pa` whose full inverse is taken -
i.e. **marginal** over the points and poses.

Point sigmas are therefore systematically optimistic relative to marker
sigmas, and the two are not comparable on the same line without saying so -
`report._posterior_sigma_summary` labels each group accordingly. It is
arguably the *right* quantity for a geometry-based filter, though
(`filter_points_by_covariance`): conditional-on-the-rest is exactly "how well
do these rays intersect," independent of whatever the rest of the block's
uncertainty happens to be.

---

## 6. Pitfalls

| Pitfall | Detail |
|---|---|
| **Raw `BACovariance` is not in metres** | It is `Q`, the cofactor for an assumed `sigma_tie = 1 px`. Always go through `variance_factor()` and `scale_by_sigma0=True` (the default on every function here) unless you specifically want the unscaled cofactor. |
| **`Point3D.error`-based `sigma0_hat` is biased low** | Mean, not RMS, of a Rayleigh-distributed quantity - understates by `sqrt(pi/4) ≈ 11%`, more once the assumed sigmas are off. Fixed by exact reprojection (§3). |
| **Camera-centre covariance needs the full 6x6** | The translation-only sub-block silently drops the rotation-translation cross term, which is not negligible and not conservatively signed (§4). |
| **`get_other_params_cov` is a pointer lookup** | Pass the *live* parameter-block array (e.g. `result.marker_xyz[name]`, not a copy or a re-cast). A non-float64 or non-C-contiguous array is silently copied by pybind11's `forcecast`, and the lookup then returns `None` instead of raising - `marker_covariances` checks `dtype`/`flags` up front and raises instead. |
| **Points get conditional covariance, markers get marginal** | Not the same statistical quantity - see §5. Point sigmas cannot be read as a datum accuracy figure the way marker sigmas can. |
| **Deleting points after covariance estimation** | `pycolmap.BACovariance` itself survives point deletion (it is a snapshot), but any residual block this package added that referenced a deleted point (e.g. `BundleResult.residual_blocks["reference_cloud"]`) now points at freed memory. `filter_points_by_covariance` and `run_bundle_adjustment`'s own post-solve filtering both prune through `utils.filtering.surviving_blocks` immediately after deleting. |
| **`SolverOptions.min_track_length` excludes points from the problem, not the reconstruction** | Such points keep their pre-bundle coordinates and get `None` covariance - `variance_factor`, `point_covariances` and `filter_points_by_covariance` all treat "no covariance" as "kept, not evidence of a bad point," never as an implicit deletion. |

---

## 7. Validation

`sandbox/examples/check_covariance.py` (`uv run python
sandbox/examples/check_covariance.py`), following the pattern of
`check_reference_cloud.py`:

1. **`sigma_tie` invariance** (§2, strongest check): the same block solved at
   `tie_points_sigma_px = 0.5` and `1.0` gives raw `Q` differing by `4x` and
   scaled `Sigma_hat` agreeing to solver tolerance.
2. **Textbook scalar**: a weighted mean of `n` noisy observations as a tiny
   `pyceres` problem; `sigma0_hat^2 * Q == s^2/n` to `1e-10`.
3. **Camera-centre propagation, Monte Carlo** (§4): sample covariance of `C`
   under the exact `Plus` vs. the closed-form `J Sigma J^T`.
4. **Jensen bias** (§3): exact reprojection sum vs. the old
   `Point3D.error`-based sum, ratio checked against `4/pi`.
5. **Prior-tightening invariant**: on a synthetic block with surveyed control
   markers, the posterior marker sigma must be no larger than what an
   unconstrained (check) marker shows - adding image observations plus a
   prior can only tighten, never loosen, the posterior relative to prior-only
   information.

---

## 8. References

- Förstner & Wrobel, *Photogrammetric Computer Vision*, Springer 2016,
  §2.4 (variance-component estimation), §4.2 (redundancy and reliability).
- Koch, *Parameter Estimation and Hypothesis Testing in Linear Models*,
  2nd ed., Springer 1999, ch. 3.
- Triggs, McLauchlan, Hartley, Fitzgibbon, *Bundle Adjustment - A Modern
  Synthesis*, Vision Algorithms 1999/2000.
- Local: `/home/francesco/software/colmap/src/colmap/estimators/covariance.{h,cc}`
  (what `BACovariance` actually computes),
  `src/colmap/estimators/bundle_adjustment_ceres.cc` (the manifold and gauge
  choices §4 depends on), `src/pycolmap/scene/camera.cc` (`img_from_cam`
  batch signature used by `_tie_point_residual_sum_sq`).
