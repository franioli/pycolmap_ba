"""Backend: constrain the sparse cloud to a georeferenced reference cloud.

A LiDAR / TLS survey that is already referenced and scaled is a strong datum:
every triangulated tie point should lie on the surface it samples. This module
turns that statement into residual blocks on COLMAP's own Ceres problem, exactly
like `constraints.py` does for markers and pose priors.

It anchors the block in two ways at once:

* **Datum** - position, orientation and scale come from the cloud, so no GCP
  survey is needed. Use `datum_information` to check the 7 DoF really are
  observable before trusting it.
* **Deformation / intrinsics** - a wrong focal length or radial distortion bends
  the sparse cloud into a dome or a bowl. The reference surface is deformation
  free, so the residual field carries the signal needed to correct the
  intrinsics. This is what makes `SolverOptions.refine_focal_length` observable
  on a flat nadir block, where it is otherwise degenerate.

The residual
------------
Point-to-point nearest neighbour is the wrong metric. A LiDAR cloud *samples* a
surface; the correspondent of a tie point lies *on* that surface and is almost
never a LiDAR sample. Snapping to the nearest sample biases every point by ~half
the point spacing and pulls it **tangentially** - precisely the systematic
lateral distortion we are trying to remove from the intrinsics.

So each reference point carries the local distribution of its neighbourhood
instead, and the residual is the Mahalanobis distance to it (Segal 2009 GICP /
Magnusson 2009 NDT). Following GICP, the measured eigenvalues decide *which*
directions are free, and the sigmas are then assigned - the eigenvalues are
never used as variances directly::

    t_i     = sqrt((lambda_i - lambda_1) / (lambda_3 - lambda_1))   in [0, 1]
    Sigma   = V diag(sigma_1^2, sigma_2^2, sigma_3^2) V^T
    sigma_i = sigma_min * (sigma_slack / sigma_min) ** t_i
    sigma_min = sqrt(sigma_ref^2 + lambda_1)

with eigenvalues ascending, so `t_1 = 0` (the across-surface direction, always
the stiffest) and `t_3 = 1` (the most spread-out direction, always the freest).
Using the raw `lambda_i` as a variance instead would make the tangential sigma
the *patch radius* - about 0.1 m for a 20-neighbour patch - which quietly
reintroduces the tangential pull this design exists to remove.

One formula then covers every surface type: a plane gets `sigma_min` along its
normal and `sigma_slack` in both tangential directions (point-to-plane, genuinely
free to slide); a breakline is stiff in two directions and free along the line;
an isotropic blob (vegetation, noise) goes fully slack and contributes nothing,
and is better removed with the `max_roughness` gate anyway.

Because a planar patch is soft tangentially, the correspondence only has to find
the right *surface patch*, not the right *point* - which is why correspondences
can be refreshed rarely (see `bundle.run_cloud_constrained_bundle_adjustment`)
instead of every iteration.

Note what this means for the datum: the horizontal datum does **not** come from
corners, it comes from surfaces whose normals are not vertical - facades, sloped
roofs, kerbs. A strictly flat scene constrains height, roll and pitch and leaves
easting, northing and yaw free, however dense the cloud. `datum_information`
reports which modes actually got constrained.

`pyceres.factors.NormalPrior(mean, cov)` evaluates `chol(cov^-1)^T (x - mean)`,
so it *is* that residual, in C++. Never write this as a Python cost function:
one holds the GIL and serialises Ceres's multithreaded pass over the tie points
(`docs/gcp_bundle_adjustment.md` measured 5 s -> 30 min).

Weighting follows the module convention of `constraints.py`: COLMAP adds its
tie-point residuals unweighted, so every sigma here is divided by
`tie_points_sigma_px` to keep one consistent whitened system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyceres
import pycolmap

from .common import covs_in_tie_units

__all__ = [
    "CorrespondenceStats",
    "DatumInformation",
    "ReferenceCloud",
    "ReferenceCloudConstraints",
    "add_reference_cloud_constraints",
    "datum_information",
]

# Chunk size for the neighbourhood pass. k=20 neighbours of 250k points is
# 250k x 21 x 3 float64 = 126 MB; chunking keeps a 10M-point cloud in bounds.
_CHUNK = 250_000


@dataclass
class ReferenceCloud:
    """A reference cloud with its per-point local surface distribution.

    Built once and never mutated: the cloud is the fixed datum, so the k-NN
    index and every geometric quantity are precomputed and reused across all
    outer passes of the adjustment. Only the *query* changes as the tie points
    move.

    Use `from_points` rather than constructing this directly.

    Attributes:
        points: `(N, 3)` reference coordinates, metres, in the same frame as the
            reconstruction.
        mu: `(N, 3)` centroid of each point's neighbourhood. This, not the
            reference point itself, is the mean of the residual - it is the
            better estimate of the underlying surface.
        eigenvalues: `(N, 3)` neighbourhood covariance eigenvalues, **ascending**
            (so `[:, 0]` is across the surface and `[:, 2]` along it).
        eigenvectors: `(N, 3, 3)` matching eigenvectors as columns.
        counts: `(N,)` number of neighbours each patch was built from.
        sigma_ref: Noise floor, metres - the 1-sigma accuracy of the reference
            cloud itself. Sets how stiff the constraint is across the surface,
            and so how hard the tie points are pulled onto it.
        sigma_slack: Sigma assigned to the free directions, metres. Should be
            large enough that sliding along the surface is effectively free
            (tens of metres for an aerial block), but finite so the covariance
            stays well conditioned. `sigma_slack / sigma_ref` is the dynamic
            range of the constraint; ~500 is comfortable, 1e6 is not.
    """

    points: np.ndarray
    mu: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    counts: np.ndarray
    sigma_ref: float = 0.02
    sigma_slack: float = 10.0
    _index: Any = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def normals(self) -> np.ndarray:
        """`(N, 3)` unit surface normals (eigenvector of the smallest eigenvalue)."""
        return self.eigenvectors[:, :, 0]

    @property
    def planarity(self) -> np.ndarray:
        """`(N,)` Weinmann planarity `(l2 - l3) / l1`, 1 for a perfect plane."""
        lam = self.eigenvalues
        largest = np.maximum(lam[:, 2], np.finfo(float).tiny)
        return (lam[:, 1] - lam[:, 0]) / largest

    @property
    def roughness(self) -> np.ndarray:
        """`(N,)` RMS deviation from the local plane, metres.

        The scale-honest test for whether a patch is a surface at all: a facade
        or a road gives a few millimetres, vegetation and other volumetric
        clutter give decimetres. See `ReferenceCloudConstraints.max_roughness`.
        """
        return np.sqrt(self.eigenvalues[:, 0])

    def sigmas(self) -> np.ndarray:
        """`(N, 3)` per-axis sigmas of the constraint, metres, ascending order.

        Interpolates geometrically between the across-surface uncertainty
        `sigma_min = sqrt(sigma_ref^2 + lambda_1)` and `sigma_slack`, using
        `t_i = sqrt((lambda_i - lambda_1) / (lambda_3 - lambda_1))` as the blend.
        See the module docstring for why the eigenvalues are not used as
        variances directly.
        """
        lam = np.maximum(self.eigenvalues, 0.0)
        smallest = lam[:, 0:1]
        spread = lam[:, 2:3] - smallest
        sigma_min = np.sqrt(self.sigma_ref**2 + smallest)
        # A patch with no spread is directionless - a blob or a duplicate point.
        # Going fully slack makes it contribute nothing, which is the safe
        # reading of "this neighbourhood tells us nothing about the surface".
        directionless = spread <= np.finfo(float).tiny
        safe = np.where(directionless, 1.0, spread)
        blend = np.where(directionless, 1.0, (lam - smallest) / safe)
        ratio = np.maximum(self.sigma_slack / sigma_min, 1.0)
        return np.minimum(sigma_min * ratio ** np.sqrt(blend), self.sigma_slack)

    def covariances(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Build the `(M, 3, 3)` GICP covariances, optionally for a subset.

        Args:
            idx: Indices to build. None builds all `N`, which for a large cloud
                is a lot of memory - pass the matched subset instead.

        Returns:
            `(M, 3, 3)` symmetric positive-definite covariances, metres squared.
        """
        sel = slice(None) if idx is None else idx
        vec = self.eigenvectors[sel]
        var = self.sigmas()[sel] ** 2
        return (vec * var[:, None, :]) @ np.swapaxes(vec, -1, -2)

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        *,
        k: int = 20,
        sigma_ref: float = 0.02,
        sigma_slack: float = 10.0,
    ) -> ReferenceCloud:
        """Build the local surface distributions of a reference cloud.

        Args:
            points: `(N, 3)` reference coordinates, metres.
            k: Neighbours per patch. Large enough to be a stable plane fit,
                small enough that a patch stays on one surface; 15-30 is usual.
            sigma_ref: Reference-cloud 1-sigma accuracy, metres.
            sigma_slack: Ceiling on the tangential sigmas, metres.

        Returns:
            A `ReferenceCloud` with its k-NN index already built.

        Raises:
            ValueError: If `points` is not `(N, 3)` or has fewer than `k` points.
        """
        import open3d as o3d

        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {pts.shape}")

        n = pts.shape[0]
        if n <= k:
            raise ValueError(f"need more than k={k} points, got {n}")

        index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(pts))
        # Both: k-NN builds the patches, hybrid serves the radius-gated `query`
        # of every outer pass.
        index.knn_index()
        index.hybrid_index()

        mu = np.empty((n, 3))
        eigenvalues = np.empty((n, 3))
        eigenvectors = np.empty((n, 3, 3))
        counts = np.empty(n, dtype=np.int64)

        # +1 because a point is its own nearest neighbour, and it belongs in its
        # own patch.
        for start in range(0, n, _CHUNK):
            block = pts[start : start + _CHUNK]
            query = o3d.core.Tensor(np.ascontiguousarray(block))
            idx = index.knn_search(query, k + 1)[0].numpy()
            # hybrid_search pads short neighbourhoods with -1; knn_search
            # cannot here, since n > k is already guaranteed above, but the
            # rest of the loop stays general rather than assuming that.
            valid = idx >= 0
            cnt = valid.sum(axis=1)
            neigh = pts[np.where(valid, idx, 0)]  # (m, k+1, 3)
            neigh *= valid[:, :, None]

            safe = np.maximum(cnt, 1)[:, None]
            centre = neigh.sum(axis=1) / safe
            dev = (neigh - centre[:, None, :]) * valid[:, :, None]
            cov = (
                np.einsum("mki,mkj->mij", dev, dev)
                / np.maximum(cnt - 1, 1)[:, None, None]
            )
            # Ascending eigenvalues; roundoff can make the smallest negative.
            val, vec = np.linalg.eigh(cov)

            stop = start + block.shape[0]
            mu[start:stop] = centre
            eigenvalues[start:stop] = np.maximum(val, 0.0)
            eigenvectors[start:stop] = vec
            counts[start:stop] = cnt

        return cls(
            points=pts,
            mu=mu,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            counts=counts,
            sigma_ref=sigma_ref,
            sigma_slack=sigma_slack,
            _index=index,
        )

    def query(
        self, points: np.ndarray, max_dist: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest patch of each query point, gated by distance.

        Args:
            points: `(M, 3)` query coordinates.
            max_dist: Reject correspondences beyond this distance, metres.

        Returns:
            `(idx, dist)`, both `(M,)`. `idx` is -1 where no reference point lies
            within `max_dist`, and `dist` is `inf` there.
        """
        import open3d as o3d

        query = o3d.core.Tensor(
            np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        )
        idx, d2, cnt = self._index.hybrid_search(query, float(max_dist), 1)
        idx = idx.numpy().reshape(-1)
        dist = np.where(idx >= 0, np.sqrt(np.abs(d2.numpy().reshape(-1))), np.inf)
        return idx, dist


@dataclass
class ReferenceCloudConstraints:
    """Which tie points get constrained to the reference cloud, and how hard.

    The gates are load-bearing, not decoration. Tie points sit on texture
    corners, which on an aerial block may be canopy, moving objects or roof
    edges rather than the surface the scanner saw; a constraint pulling those
    onto the reference cloud is worse than no constraint at all.

    Attributes:
        max_dist: Reject a correspondence farther than this, metres. Set it from
            the expected initial misalignment, not from the final accuracy: too
            tight in the first outer pass and the block can never pull itself in.
        min_track_length: Skip points seen by fewer images. Short tracks are
            poorly triangulated and would import their own error into the datum.
        min_neighbors: Skip reference patches built from fewer neighbours - the
            covariance of a sparse patch is meaningless.
        max_roughness: Optional cap on the RMS deviation from the local plane,
            metres. The main clutter filter: a facade or a road is millimetres
            rough, vegetation is decimetres. A few times `sigma_ref` is a good
            starting point. None keeps every patch.
        robust_loss_scale: Cauchy scale, in whitened (sigma) units, so ~2 means
            "start downweighting past 2 sigma". None uses a plain L2 loss, which
            lets a single gross correspondence drag the datum.
    """

    max_dist: float = 0.5
    min_track_length: int = 2
    min_neighbors: int = 6
    max_roughness: float | None = None
    robust_loss_scale: float | None = 2.0

    def __post_init__(self) -> None:
        if self.max_dist <= 0:
            raise ValueError("max_dist must be positive")


@dataclass
class CorrespondenceStats:
    """How the correspondence search went in one outer pass.

    Attributes:
        n_candidates: Tie points offered to the search.
        n_matched: Points with a reference patch within `max_dist`.
        n_used: Points that survived every gate and got a residual block.
        point3D_ids: `(n_used,)` ids of the constrained points, in block order.
        reference_ids: `(n_used,)` matched reference-patch indices, same order.
            Comparing this between passes is the convergence test of the outer
            loop.
        distances: `(n_used,)` distance to the matched reference *sample*,
            metres. Note this has a floor of roughly half the point spacing even
            for a tie point lying exactly on the surface, so it is a diagnostic
            of the search, **not** a measure of fit - read `surface_distances`
            for that.
        surface_distances: `(n_used,)` perpendicular distance to the local plane
            of the matched patch, metres. This is the quantity the residual
            actually penalises, and the one that should shrink between passes.
        rejected: Counts by reason - `no_match`, `track`, `neighbors`,
            `roughness`.
    """

    n_candidates: int = 0
    n_matched: int = 0
    n_used: int = 0
    point3D_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    reference_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    distances: np.ndarray = field(default_factory=lambda: np.empty(0))
    surface_distances: np.ndarray = field(default_factory=lambda: np.empty(0))
    rejected: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable recap."""
        if self.n_used == 0:
            return f"reference cloud: 0 / {self.n_candidates} points constrained"
        dropped = ", ".join(f"{k} {v}" for k, v in self.rejected.items() if v)
        surface = self.surface_distances
        return (
            f"reference cloud: {self.n_used:,} / {self.n_candidates:,} points "
            f"constrained | to plane: median {np.median(surface):.3f} m, "
            f"p95 {np.percentile(surface, 95):.3f} m | to sample: median "
            f"{np.median(self.distances):.3f} m"
            + (f" | dropped: {dropped}" if dropped else "")
        )


@dataclass
class DatumInformation:
    """Observability of the 7-DoF datum under the reference-cloud constraints.

    A flat scene is the trap this exists to catch: every normal points up, so the
    constraint fixes height, roll and pitch and leaves easting, northing and yaw
    free - the block looks converged while floating horizontally. What breaks
    that degeneracy is *normal diversity*: facades, sloped roofs, kerbs. Density
    does not help at all, which is why this has to be measured rather than
    assumed from the number of constrained points.

    Rotation and scale columns are scaled by the RMS distance from the centroid,
    so all seven columns are in metres and directly comparable: `sigmas` reads as
    "how far the block could still move, at its edge, in this mode".

    Attributes:
        information: `(7, 7)` `sum_j J_j^T J_j` with
            `J_j = A_j [I, -[X_j - c]x, (X_j - c)] / L`, `A_j = chol(Sigma_j^-1)`.
        eigenvalues: Ascending eigenvalues of `information`. The smallest is the
            weakest mode.
        eigenvectors: Matching eigenvectors as columns, in the order
            `[tx, ty, tz, rx, ry, rz, scale]`.
        sigmas: `1 / sqrt(eigenvalue)` per mode, metres at the block edge.
        centroid: `(3,)` centroid the modes are taken about.
        scale_length: `L`, the RMS distance from the centroid, metres.
    """

    information: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    sigmas: np.ndarray
    centroid: np.ndarray
    scale_length: float

    LABELS = ("tx", "ty", "tz", "rx", "ry", "rz", "scale")

    def summary(self, threshold: float = 0.05) -> str:
        """Report each mode, flagging any weaker than `threshold` metres.

        Args:
            threshold: Sigma above which a mode is called weak, metres.

        Returns:
            A multi-line report, one line per mode, weakest first.
        """
        lines = [f"datum observability (about centroid, L = {self.scale_length:.1f} m)"]
        for value, vec in zip(self.sigmas, self.eigenvectors.T, strict=True):
            dominant = int(np.argmax(np.abs(vec)))
            flag = "  <-- WEAK" if value > threshold else ""
            lines.append(
                f"  sigma {value:9.4f} m  mostly {self.LABELS[dominant]:>5}"
                f"  ({np.abs(vec[dominant]):.2f}){flag}"
            )
        return "\n".join(lines)


def datum_information(
    points: np.ndarray, covariances: np.ndarray, centroid: np.ndarray | None = None
) -> DatumInformation:
    """Accumulate how much the constraints pin down the 7-DoF similarity.

    Answers "does my reference cloud actually fix the datum, and which degree of
    freedom is weak?" - cheaply, in numpy, without a covariance estimation pass.

    A similarity perturbation of the block moves a point by
    `dX = t + omega x (X - c) + s (X - c)`, so the whitened Jacobian of one
    constraint is `A [I, -[X - c]x, (X - c)]`. Summing `J^T J` gives the
    information the cloud contributes to those seven parameters.

    Args:
        points: `(M, 3)` constrained point coordinates, metres.
        covariances: `(M, 3, 3)` their constraint covariances, metres squared,
            in the same units the residual blocks use.
        centroid: Point the modes are taken about. None uses the centroid of
            `points`.

    Returns:
        A `DatumInformation`.

    Raises:
        ValueError: If the shapes disagree or there are no points.
    """
    pts = np.asarray(points, dtype=float)
    cov = np.asarray(covariances, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (M, 3), got {pts.shape}")
    if cov.shape != (pts.shape[0], 3, 3):
        raise ValueError(f"covariances must be (M, 3, 3), got {cov.shape}")
    if pts.shape[0] == 0:
        raise ValueError("no constrained points to report on")

    centre = pts.mean(axis=0) if centroid is None else np.asarray(centroid, float)
    rel = pts - centre
    length = float(np.sqrt(np.mean(np.sum(rel**2, axis=1))))
    length = max(length, np.finfo(float).eps)

    # A = chol(Sigma^-1)^T, the whitener NormalPrior applies internally.
    whitener = np.linalg.cholesky(np.linalg.inv(cov)).swapaxes(-1, -2)

    m = pts.shape[0]
    jac = np.zeros((m, 3, 7))
    jac[:, :, 0:3] = np.eye(3)
    # -[r]x, the derivative of a small rotation about the centroid.
    jac[:, 0, 4] = rel[:, 2] / length
    jac[:, 0, 5] = -rel[:, 1] / length
    jac[:, 1, 3] = -rel[:, 2] / length
    jac[:, 1, 5] = rel[:, 0] / length
    jac[:, 2, 3] = rel[:, 1] / length
    jac[:, 2, 4] = -rel[:, 0] / length
    jac[:, :, 6] = rel / length
    jac = whitener @ jac

    information = np.einsum("mij,mik->jk", jac, jac)
    values, vectors = np.linalg.eigh(information)
    values = np.maximum(values, 0.0)
    with np.errstate(divide="ignore"):
        sigmas = 1.0 / np.sqrt(values)
    return DatumInformation(
        information=information,
        eigenvalues=values,
        eigenvectors=vectors,
        sigmas=sigmas,
        centroid=centre,
        scale_length=length,
    )


def add_reference_cloud_constraints(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    cloud: ReferenceCloud,
    options: ReferenceCloudConstraints,
    tie_points_sigma_px: float,
    retained: list[Any],
) -> tuple[int, CorrespondenceStats]:
    """Pull the sparse tie points onto the reference surface.

    One `NormalPrior` per surviving tie point, on the `Point3D.xyz` parameter
    block COLMAP already owns - `Point3D.xyz` is a live view into COLMAP's
    memory, so the block Ceres optimises and the block constrained here are the
    same one. Nothing is added to the reconstruction itself.

    Correspondences are resolved once, here. They are *not* refreshed inside the
    solve: Ceres's trust region assumes a fixed cost function, so changing them
    between evaluations makes step quality meaningless. Refresh happens in the
    outer loop instead (`bundle.run_cloud_constrained_bundle_adjustment`).

    Args:
        problem: The Ceres problem from the *single* bundle adjuster.
        reconstruction: Source of the tie points; not modified.
        cloud: The precomputed reference cloud.
        options: Gates and robustification.
        tie_points_sigma_px: Reference sigma for the weighting (see module doc).
        retained: Keep-alive list for the cost functions - pyceres does not take
            ownership, and a collected cost function leaves Ceres reading freed
            memory.

    Returns:
        `(n_blocks_added, stats)`.
    """
    rejected = {
        "no_match": 0,
        "track": 0,
        "neighbors": 0,
        "roughness": 0,
    }

    # Only points the solver actually moves are worth constraining. Asking the
    # problem is more robust than reasoning about which points the config
    # implied: it covers constant points and points left out of the adjustment.
    ids: list[int] = []
    blocks: list[np.ndarray] = []
    n_short = 0
    for pid, point in reconstruction.points3D.items():
        if point.track.length() < options.min_track_length:
            n_short += 1
            continue
        xyz = point.xyz
        if not problem.has_parameter_block(xyz):
            continue
        if problem.is_parameter_block_constant(xyz):
            continue
        ids.append(pid)
        blocks.append(xyz)
    rejected["track"] = n_short

    stats = CorrespondenceStats(n_candidates=len(ids), rejected=rejected)
    if not ids:
        return 0, stats

    xyz = np.array([b for b in blocks], dtype=np.float64)
    ref_idx, dist = cloud.query(xyz, options.max_dist)

    keep = ref_idx >= 0
    rejected["no_match"] = int((~keep).sum())
    stats.n_matched = int(keep.sum())

    safe_idx = np.where(keep, ref_idx, 0)
    too_sparse = cloud.counts[safe_idx] < options.min_neighbors
    rejected["neighbors"] = int((keep & too_sparse).sum())
    keep &= ~too_sparse

    if options.max_roughness is not None:
        smooth = cloud.roughness[safe_idx] <= options.max_roughness
        rejected["roughness"] = int((keep & ~smooth).sum())
        keep &= smooth

    rows = np.flatnonzero(keep)
    if rows.size == 0:
        return 0, stats

    matched = ref_idx[rows]
    # Scale into tie-point sigma units: the whole objective is multiplied by
    # sigma_tie^2, which leaves COLMAP's unweighted tie-point blocks untouched.
    covs = covs_in_tie_units(cloud.covariances(matched), tie_points_sigma_px)
    means = cloud.mu[matched]
    robust = (
        pyceres.CauchyLoss(options.robust_loss_scale)
        if options.robust_loss_scale is not None
        else None
    )

    for row, mean, cov in zip(rows, means, covs, strict=True):
        cost = pyceres.factors.NormalPrior(mean, cov)
        retained.append(cost)
        problem.add_residual_block(cost, robust, [blocks[row]])

    stats.n_used = int(rows.size)
    stats.point3D_ids = np.asarray([ids[r] for r in rows], dtype=np.int64)
    stats.reference_ids = matched.astype(np.int64)
    stats.distances = dist[rows]
    stats.surface_distances = np.abs(
        np.einsum("mi,mi->m", xyz[rows] - means, cloud.normals[matched])
    )
    return stats.n_used, stats
