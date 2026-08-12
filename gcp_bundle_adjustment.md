# GCP-based bundle adjustment — the logic

Reference note for `ba_gcp.py` (ENRICH-Aerial, nadir-2 subblock).

---

## 1. What a GCP actually is

A ground control point is **not** a special kind of observation. It is an
ordinary tie point that happens to carry **two independent pieces of
information**:

| | Information | Residual type | Units |
|---|---|---|---|
| 1 | its image measurements $(u,v)$ in one or more photos | reprojection | px |
| 2 | its surveyed world coordinate $\tilde{X}$ with covariance | position prior | m |

Only (2) needs anything that ordinary SfM does not already have, and it is just
a 3-residual Gaussian prior. This is why no library ships a monolithic "GCP
cost function": there is nothing to ship beyond a prior.

A point given only (1) is a plain tie point. A point given only (2) is a
constraint on nothing. A point given both is control.

---

## 2. The objective

Let $\theta = \{T_i, K, X_j\}$ be poses, intrinsics and 3D points.

$$
J(\theta)=\underbrace{\sum_{ij}\rho\!\left(\lVert \pi(K,T_i,X_j)-x_{ij}\rVert^2_{\Sigma_{\text{tie}}}\right)}_{\text{tie points}}
+\underbrace{\sum_{g,i}\rho\!\left(\lVert \pi(K,T_i,X_g)-x_{gi}\rVert^2_{\Sigma_{\text{mark}}}\right)}_{\text{GCP image marks}}
+\underbrace{\sum_{g\in\mathcal{C}}\lVert X_g-\tilde{X}_g\rVert^2_{\Sigma_{\text{gcp}}}}_{\text{control priors}}
$$

where $\mathcal{C}$ is the **control** set. Check points contribute their image
marks but *not* the prior — see §6.

$\lVert r\rVert^2_\Sigma \equiv r^\top\Sigma^{-1}r$ is the squared Mahalanobis
norm.

---

## 3. Why weighting is mandatory, and what "correct" means

The three terms are in **pixels²**, **pixels²** and **metres²**. Summing them
raw is meaningless. Whitening fixes this: factor the inverse covariance as

$$
A^\top A=\Sigma^{-1},\qquad A=\operatorname{chol}(\Sigma^{-1})^\top
$$

and replace each residual by $r\mapsto Ar$. Then

$$
\lVert Ar\rVert^2=r^\top\Sigma^{-1}r
$$

is **dimensionless** — a count of standard deviations. A 2 cm world error
against $\sigma=2$ cm and a 1 px image error against $\sigma=1$ px both
evaluate to $1.0$, and are now directly comparable. **Metres are never
converted to pixels.**

**Any valid square root works.** $J$ depends on $A$ only through $A^\top A$, so
upper Cholesky, lower Cholesky, the symmetric root, or any of these times an
orthogonal matrix all give identical cost, minimum and estimate covariance.
Only the residual *vector* differs, not its norm.

### Statistical meaning

For $X_g\sim\mathcal{N}(\tilde{X}_g,\Sigma_{\text{gcp}})$,

$$
-\log L=\tfrac12 (X_g-\tilde{X}_g)^\top\Sigma^{-1}_{\text{gcp}}(X_g-\tilde{X}_g)+\text{const}
$$

and Ceres's per-block cost is $\tfrac12\lVert r\rVert^2$. So the control prior
**is** the Gaussian negative log-likelihood — hence `ceres::NormalPrior`. The
whole bundle is a maximum-likelihood estimate, and $J$ at the optimum is a
$\chi^2$ statistic.

---

## 4. The COLMAP-specific scaling detail

`create_default_ceres_bundle_adjuster` adds its tie-point reprojections
**unweighted**, i.e. with an implicit $\sigma_{\text{tie}}=1$ px. Rebuilding
115k residuals in Python to fix that would be prohibitively slow.

Since scaling $J$ by a positive constant does not move the minimum, multiply
through by $\sigma_{\text{tie}}^2$:

$$
\sigma_{\text{tie}}^2 J=\sum r_{\text{tie}}^2
+\sum\left(\frac{r_{\text{mark}}}{\sigma_{\text{mark}}/\sigma_{\text{tie}}}\right)^{2}
+\sum\left(\frac{r_{\text{world}}}{\sigma_{\text{gcp}}/\sigma_{\text{tie}}}\right)^{2}
$$

The first term now matches COLMAP's blocks exactly, and every other $\sigma$ is
simply **expressed in units of $\sigma_{\text{tie}}$**:

```python
gcp_image_cov = np.diag([(gcp_sigma_px / tie_points_sigma_px) ** 2] * 2)
gcp_world_cov = np.diag((gcp_sigma_m  / tie_points_sigma_px) ** 2)
```

The resulting `0.02/0.5 = 0.04` has odd units (m/px); that is a harmless
artefact of the global rescaling. Omitting the division would over-weight the
GCPs by $(\sigma_{\text{tie}}^{-1})^2 = 4\times$.

---

## 5. Choosing the sigmas

Only **ratios** matter. Practical guidance:

| Term | Source of the number |
|---|---|
| $\sigma_{\text{tie}}$ | feature localisation, typically 0.3–1 px |
| $\sigma_{\text{mark}}$ | how precisely the target was identified in the image |
| $\sigma_{\text{gcp}}$ | what the survey actually achieved (GNSS RTK ≈ 1–3 cm) |

**Set $\sigma_{\text{gcp}}$ to the real survey accuracy — tighter is not
safer.** Over-tight control forces the block to honour coordinates the imagery
cannot support and *degrades* independent check points.

Use anisotropic covariance where justified: GNSS height is typically 1.5–2×
worse than horizontal, so $\operatorname{diag}(\sigma_{xy}^2,\sigma_{xy}^2,\sigma_z^2)$
rather than isotropic.

### GSD sanity check

GSD is **not** needed for weighting, but is the right tool to check that the
sigmas are mutually sensible:

$$
\text{GSD}=\frac{H}{f}
$$

For this block: $H=172.5$ m above the GCPs, $f=5865$ px → **2.94 cm/px**.

| sigma | ground equivalent |
|---|---|
| tie 0.5 px | 1.47 cm |
| mark 1.0 px | 2.94 cm |
| survey 2 cm | 0.68 px |

Balanced — neither source dominates. If the survey converted to ~0.01 px the
control would steamroll the imagery; at ~20 px it would be decoration.

**Do not read $\sigma_z$ through GSD.** Photogrammetric height precision scales
as $\text{GSD}\times H/B$ (base-to-height ratio), several times worse than
planimetry for a nadir block.

---

## 6. Datum, gauge and check points

A free network is determined only up to a **7-DoF similarity** (3 translation,
3 rotation, 1 scale). Reprojection error is completely blind to it: the normal
equations $J^\top J$ are rank-deficient by 7.

Consequences:

- **≥3 non-collinear GCPs fix the datum**, replacing arbitrary gauge fixing.
- Along the 7 gauge directions the imagery contributes *no curvature*, so the
  minimum there is set by the control alone — **any** non-zero weight pins the
  datum equally well. Weight only matters where control *competes* with image
  information.
- BA is a local optimiser. If the free network is in an arbitrary frame it must
  first be brought close by a Sim3 alignment (`align_reconstruction_to_locations`),
  otherwise the GCP residuals start hopeless and the solve diverges.

**Check points** are GCPs whose image marks enter the bundle but whose world
prior does not. They are triangulated by the adjustment, and their distance to
the surveyed coordinate is the only **unbiased** accuracy estimate. Control
residuals always flatter you — those points were pulled to fit. Hold out ~1/3.

A GCP seen in a single image cannot be triangulated and therefore cannot serve
as a check point.

---

## 7. Implementation

No `Point3D`/`Track` surgery. Each GCP is a parameter block we own, wired into
COLMAP's existing Ceres problem:

```
gcp_xyz (3,)                      <- our numpy array, init from the survey
  ├── ReprojErrorCost(model, Σ_mark, uv)     per image mark
  │     blocks: [gcp_xyz, rig_from_world.params, camera.params]
  └── NormalPrior(surveyed, Σ_gcp)           control points only
        blocks: [gcp_xyz]
```

This works because `camera.params` and `frames[fid].rig_from_world.params` are
**live views** into COLMAP's memory (`owndata=False`, stable addresses), so
these are the same parameter blocks COLMAP optimises — with the quaternion
manifold already set. For a single-camera rig `cam_from_world == rig_from_world`.

Robust loss (Cauchy) on the image marks absorbs mis-identified targets; the
world priors stay quadratic.

---

## 8. Pitfalls encountered

| Pitfall | Detail |
|---|---|
| **pyceres does not own cost functions** | Keep every cost object in a Python list or the solver reads freed memory (hard SIGSEGV). |
| **Python cost functions hold the GIL** | They serialise Ceres's multithreaded pass over 115k C++ residuals: a 5 s solve became a 30 min timeout. Use `pyceres.factors.NormalPrior` (C++), verified bit-identical to a hand-written whitened prior. |
| **Spurious `NO_CONVERGENCE`** | COLMAP sets `function_tolerance = 0` and `parameter_tolerance = 0` ([bundle_adjustment_ceres.cc:102](/home/francesco/software/colmap/src/colmap/estimators/bundle_adjustment_ceres.cc#L102)), *disabling* both, leaving only `gradient_tolerance = 1e-4`. Every solve runs to the 100-iteration cap and reports failure although the cost converged. Measured: defaults → NO_CONVERGENCE 4.8 s; 500 iters → NO_CONVERGENCE 27.3 s; `ftol=1e-8` → **CONVERGENCE 0.4 s**. More iterations do not help; re-enabling `function_tolerance` does. |
| **Flat-block self-calibration is degenerate** | This nadir block cannot determine focal length from tie points: free-network self-calibration ran to `fx=3227.6, fy=3818.1` while still achieving 0.820 px reprojection — identical to the correct model. Focal trades against scene depth. Control is what breaks the degeneracy. |
| **Coordinate conventions** | Blender cameras look down $-Z$ with $+Y$ up; COLMAP looks down $+Z$ with $+Y$ down. Convert with $R_{cw}=\operatorname{diag}(1,-1,-1)\,R_{wc}^\top$. |
| **Keypoint frame** | COLMAP rescales keypoints back to **original** image coordinates after `max_image_size` downsampling, so GCP marks in full-resolution pixels are directly compatible. |

---

## 9. Dataset caveat (ENRICH-Aerial)

Fitting $f,c_x,c_y$ to `cameras.csv` + `gcp_list.csv` + `gcp_images_list.csv`
gives $f=5865.177$, $c=(3008.0,2008.0)$ at **0.0011 px RMS** — essentially
exact. The nominal render focal is $f=5882$, which leaves **5.47 px** even with
the principal point free. The GCP marks were therefore generated with a focal
~0.29 % shorter than the render, or from slightly different poses.

Since the imagery cannot constrain focal (§8), the GCP marks are the only thing
driving it, and the bundle will be pulled toward 5865.2. Fixing the camera at
5882 leaves an irreducible ~5.5 px systematic in the GCP residuals (≈16 cm on
the ground) that leaks into the check points. Worth resolving at source.

---

## 10. References

- Triggs, McLauchlan, Hartley, Fitzgibbon, *Bundle Adjustment — A Modern
  Synthesis*, Vision Algorithms 1999/2000. The canonical treatment of gauge
  freedom and weighting.
- Förstner & Wrobel, *Photogrammetric Computer Vision*, Springer 2016.
  Statistical formulation, variance components, datum definition.
- Luhmann, Robson, Kyle, Boehm, *Close-Range Photogrammetry and 3D Imaging*.
  Control/check point practice, base-to-height ratio.
- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016 (COLMAP).
- Ceres Solver documentation — `NormalPrior`, `LossFunction`, `Manifold`,
  solver options: <http://ceres-solver.org/>
- Local: `ba_gcp.py`, and `ba_tutorial.py` in the belvedere project for a
  from-scratch BA walkthrough including a GCP stage.