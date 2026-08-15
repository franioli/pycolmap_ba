"""Helpers: is this block close enough to the reference cloud to be constrained?

The reference-cloud bundle is a *local* refiner. Its residual is the squared
distance to the reference surface, whose quadratic model is only valid inside
the medial axis of that surface (Pottmann & Hofer, *Geometry of the squared
distance function to curves and surfaces*, 2003; Pottmann, Huang, Yang & Hu,
IJCV 67(3):277-296, 2006, sections 3-4). Half the distance to the nearest
*distinct* surface sheet is therefore the half-width of the basin the block has
to start inside: further out than that, the nearest patch belongs to the wrong
sheet and every correspondence is confidently wrong. `surface_separation`
measures that distance from the cloud itself rather than assuming it.

`check_cloud_compatibility` runs that test, plus the two cheaper ones that catch
the failures seen in practice before they reach the solver:

* **Footprint** - the block and the survey rarely cover the same ground. A
  block hanging half outside the surveyed area is not misaligned, it is
  unsupported, and no amount of iterating fixes it.
* **Distance** - within the shared footprint, how far the tie points actually
  sit from the surface, which is what the first correspondence gate has to
  span.

Everything distance-like is measured **inside the footprint intersection only**.
That gating is the point: an ungated nearest-neighbour statistic over the whole
block lets every out-of-footprint point contribute its full (huge) distance, and
under partial overlap that contamination survives even robust trimming, so the
statistic reports a misalignment that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pycolmap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.ref_cloud import ReferenceCloud

__all__ = [
    "CompatibilityOptions",
    "CompatibilityReport",
    "check_cloud_compatibility",
    "sparse_xyz",
    "surface_separation",
]


@dataclass
class CompatibilityOptions:
    """Thresholds deciding whether a block may be constrained to a cloud.

    Attributes:
        max_dist: Distance a tie point may sit from the surface and still count
            as matched, metres. Set it to the **first** correspondence gate the
            bundle will use: this check answers "will that gate find anything",
            so a different value answers a different question.
        min_overlap: Fraction of the in-footprint tie points that must be within
            `max_dist`. Below this the first pass has too little to pull on.
        min_coverage: Fraction of the block that must fall inside the surveyed
            footprint at all. Deliberately low - a cloud covering a corner of
            the block is still a usable datum, an empty intersection is not.
        rms_margin: Multiple of `max_dist` the trimmed RMS may reach before the
            block counts as misaligned rather than merely imperfect.
        cell_m: Grid size of the footprint occupancy masks, metres. A few times
            the tie-point spacing; too fine and a sparse block reads as holes.
        min_track_length: Ignore tie points seen by fewer images - short tracks
            carry their own triangulation error into the statistic.
        percentile: Trim the distances above this percentile before taking the
            RMS. A sparse cloud always holds points with no counterpart at all.
        ambiguity_scale: `Delta_surf`, the distance between distinct surface
            sheets, metres; the block must sit within half of it (see the module
            docstring). None measures it from the cloud with
            `surface_separation`. Pass `float("inf")` to disable the test.
    """

    max_dist: float = 1.0
    min_overlap: float = 0.30
    min_coverage: float = 0.05
    rms_margin: float = 1.0
    cell_m: float = 10.0
    min_track_length: int = 3
    percentile: float = 90.0
    ambiguity_scale: float | None = None


@dataclass
class CompatibilityReport:
    """How the block sits relative to the reference cloud.

    Attributes:
        n_points: Tie points tested.
        n_in_overlap: Of those, how many fall in the footprint intersection.
            Every distance below is measured over these points only.
        coverage: `n_in_overlap / n_points`.
        overlap_fraction: Fraction of the in-footprint points within
            `max_dist` of the cloud - open3d's ICP "fitness", gated.
        trimmed_rms: Trimmed RMS of the nearest-neighbour distances, metres.
            The headline misalignment number.
        inlier_rms: RMS over the within-`max_dist` points only, metres, the
            quantity paired with `overlap_fraction` (open3d's `inlier_rmse`).
            Reads as "how well do the points that do match, match".
        p95_distance: 95th percentile of the distances, metres.
        median_surface_distance: Median point-to-plane distance over the matched
            points, metres. Diagnostic only: it is what the bundle will actually
            penalise, and is far smaller than the point-to-sample distance.
        ambiguity_margin: `Delta_surf / 2`, metres, or None if the cloud shows no
            distinct second sheet (nothing to be ambiguous with).
        failures: Which tests failed, a subset of `coverage`, `overlap`, `rms`,
            `ambiguity`. Empty means the block may be constrained.
        options: The thresholds used, kept so the report is self-explaining.
    """

    n_points: int
    n_in_overlap: int
    coverage: float
    overlap_fraction: float
    trimmed_rms: float
    inlier_rms: float
    p95_distance: float
    median_surface_distance: float
    ambiguity_margin: float | None
    failures: tuple[str, ...] = ()
    options: CompatibilityOptions = field(default_factory=CompatibilityOptions)

    @property
    def passed(self) -> bool:
        """True when no test failed."""
        return not self.failures

    def summary(self) -> str:
        """Two-line recap, in the style of `PrealignResult.summary`."""
        verdict = "ok" if self.passed else "FAILED " + ",".join(self.failures)
        margin = (
            f", ambiguity margin {self.ambiguity_margin:.2f} m"
            if self.ambiguity_margin is not None
            else ""
        )
        return (
            f"compatibility: {verdict} | coverage {self.coverage:.2f} "
            f"({self.n_in_overlap:,} / {self.n_points:,} points in footprint), "
            f"overlap {self.overlap_fraction:.2f} at {self.options.max_dist:.2f} m"
            f"{margin}\n"
            f"  in-footprint distance to cloud: trimmed RMS "
            f"{self.trimmed_rms:.3f} m, inlier RMS {self.inlier_rms:.3f} m, "
            f"p95 {self.p95_distance:.3f} m | to plane "
            f"{self.median_surface_distance:.3f} m"
        )


def sparse_xyz(
    reconstruction: pycolmap.Reconstruction, min_track_length: int = 3
) -> np.ndarray:
    """Well-determined tie points as an `(N, 3)` array."""
    return np.array(
        [
            p.xyz
            for p in reconstruction.points3D.values()
            if p.track.length() >= min_track_length
        ],
        dtype=np.float64,
    )


def surface_separation(
    cloud: ReferenceCloud,
    *,
    cell_m: float = 2.0,
    min_gap: float = 0.5,
    percentile: float = 25.0,
    max_points: int = 1_000_000,
) -> float:
    """Estimate `Delta_surf`, the distance between distinct surface sheets.

    Per XY cell, the largest vertical gap between consecutive points is the
    clear space between the sheets sampled there - ground under a roof, road
    beside a wall top. Gaps smaller than `min_gap` are just the cloud's own
    sampling, not a second surface. The low percentile across cells is taken on
    purpose: the *tightest* separation anywhere in the block is what limits how
    far the block may start out, not the typical one.

    Vertical rather than fully 3D because the sheets that trap an aerial block
    are stacked in height; a facade is a single sheet whichever way it faces.

    Args:
        cloud: The reference cloud.
        cell_m: XY cell size, metres. A few times the point spacing.
        min_gap: Ignore gaps below this, metres - sampling noise, not a sheet.
        percentile: Percentile taken across the cells that show a gap.
        max_points: Subsample above this many points; the estimate is a
            percentile over cells and does not need every sample.

    Returns:
        The separation in metres, or `inf` when no cell shows a second sheet -
        a single-surface scene, where nearest-surface correspondence cannot be
        ambiguous however far out the block starts.
    """
    points = cloud.points
    if len(points) > max_points:
        points = points[:: len(points) // max_points + 1]
    if len(points) < 2:
        return float("inf")

    cells = np.floor(points[:, :2] / cell_m).astype(np.int64)
    _, cell_id = np.unique(cells, axis=0, return_inverse=True)
    cell_id = cell_id.reshape(-1)

    order = np.lexsort((points[:, 2], cell_id))
    z, ids = points[order, 2], cell_id[order]
    same_cell = ids[1:] == ids[:-1]
    # Gap from each point to the next one in its own cell; 0 across cell borders
    # and for the last point of a cell, so neither can win the per-cell maximum.
    gaps = np.append(np.where(same_cell, np.diff(z), 0.0), 0.0)
    starts = np.flatnonzero(np.r_[True, ~same_cell])
    largest = np.maximum.reduceat(gaps, starts)

    separations = largest[largest >= min_gap]
    if separations.size == 0:
        return float("inf")
    return float(np.percentile(separations, percentile))


def _footprint_overlap(
    block_xy: np.ndarray, reference_xy: np.ndarray, cell_m: float
) -> np.ndarray:
    """Per-block-point mask of the footprint intersection.

    Occupancy grids of both clouds on a shared XY lattice, then the block is
    dilated by one cell and the reference eroded by one: generous about where
    the block reaches, conservative about where the survey genuinely supports
    it, so a point one cell inside the reference's ragged edge is not claimed as
    supported.

    Args:
        block_xy: `(N, 2)` tie-point coordinates.
        reference_xy: `(M, 2)` reference coordinates.
        cell_m: Cell size, metres.

    Returns:
        `(N,)` boolean, True where that tie point falls in the intersection.
    """
    lo = np.minimum(block_xy.min(axis=0), reference_xy.min(axis=0)) - cell_m
    hi = np.maximum(block_xy.max(axis=0), reference_xy.max(axis=0)) + cell_m
    shape = tuple(np.ceil((hi - lo) / cell_m).astype(int) + 2)

    def occupancy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((xy - lo) / cell_m).astype(np.int64)
        grid = np.zeros(shape, dtype=bool)
        grid[ij[:, 0], ij[:, 1]] = True
        return grid, ij

    block, block_ij = occupancy(block_xy)
    reference, _ = occupancy(reference_xy)

    # 4-neighbour structuring element. The one-cell margin added to `lo`/`hi`
    # keeps the border false, so np.roll's wraparound can only ever remove
    # cells - which is the safe direction for both operations.
    dilated = block.copy()
    eroded = reference.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            dilated |= np.roll(block, shift, axis=axis)
            eroded &= np.roll(reference, shift, axis=axis)

    overlap = dilated & eroded
    return overlap[block_ij[:, 0], block_ij[:, 1]]


def check_cloud_compatibility(
    reconstruction: pycolmap.Reconstruction,
    cloud: ReferenceCloud,
    options: CompatibilityOptions | None = None,
) -> CompatibilityReport:
    """Decide whether the block may be constrained to the reference cloud.

    Cheap enough to run unconditionally before a bundle, and the only thing
    standing between a badly placed block and a confidently wrong solve. Also
    useful on its own, on real data, as the answer to "is this survey usable as
    control for this block at all".

    Args:
        reconstruction: Source of the tie points; not modified.
        cloud: The reference cloud.
        options: Thresholds. None uses `CompatibilityOptions()`.

    Returns:
        A `CompatibilityReport`; read `passed` for the verdict.

    Raises:
        ValueError: If no tie point survives `min_track_length`.
    """
    opts = options or CompatibilityOptions()
    xyz = sparse_xyz(reconstruction, opts.min_track_length)
    if len(xyz) == 0:
        raise ValueError("no tie points with the requested track length")

    in_overlap = _footprint_overlap(xyz[:, :2], cloud.points[:, :2], opts.cell_m)
    coverage = float(in_overlap.mean())
    inside = xyz[in_overlap]

    separation = (
        surface_separation(cloud)
        if opts.ambiguity_scale is None
        else opts.ambiguity_scale
    )
    margin = float(separation / 2) if np.isfinite(separation) else None

    if len(inside) == 0:
        return CompatibilityReport(
            n_points=len(xyz),
            n_in_overlap=0,
            coverage=0.0,
            overlap_fraction=0.0,
            trimmed_rms=float("inf"),
            inlier_rms=float("inf"),
            p95_distance=float("inf"),
            median_surface_distance=float("nan"),
            ambiguity_margin=margin,
            failures=("coverage", "overlap", "rms"),
            options=opts,
        )

    idx, dist = cloud.nearest(inside)
    within = dist <= opts.max_dist
    trimmed = dist[dist <= np.percentile(dist, opts.percentile)]
    surface = np.abs(
        np.einsum(
            "mi,mi->m", inside[within] - cloud.mu[idx[within]], cloud.normals[idx[within]]
        )
    )

    report = CompatibilityReport(
        n_points=len(xyz),
        n_in_overlap=int(in_overlap.sum()),
        coverage=coverage,
        overlap_fraction=float(within.mean()),
        trimmed_rms=float(np.sqrt(np.mean(trimmed**2))),
        inlier_rms=(
            float(np.sqrt(np.mean(dist[within] ** 2))) if within.any() else float("inf")
        ),
        p95_distance=float(np.percentile(dist, 95)),
        median_surface_distance=(
            float(np.median(surface)) if surface.size else float("nan")
        ),
        ambiguity_margin=margin,
        options=opts,
    )

    failures = []
    if report.coverage < opts.min_coverage:
        failures.append("coverage")
    if report.overlap_fraction < opts.min_overlap:
        failures.append("overlap")
    if report.trimmed_rms > opts.rms_margin * opts.max_dist:
        failures.append("rms")
    if margin is not None and report.trimmed_rms > margin:
        failures.append("ambiguity")
    report.failures = tuple(failures)
    return report
