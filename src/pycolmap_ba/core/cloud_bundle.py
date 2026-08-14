"""Backend: bundle adjustment alternating with reference-cloud correspondence.

Separate from `bundle` because this is an *iteration strategy* wrapped around
the bundle, not a way of assembling one - the same ICP-style alternation that
`prealign` uses to get the block into range, continued at bundle accuracy. It
calls `run_bundle_adjustment` once per pass and never touches a residual block
itself.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pycolmap

from .bundle import BundleResult, run_bundle_adjustment
from .cloud import ReferenceCloud, ReferenceCloudConstraints

__all__ = ["run_cloud_constrained_bundle_adjustment"]


def run_cloud_constrained_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    reference_cloud: ReferenceCloud,
    *,
    reference_options: ReferenceCloudConstraints | None = None,
    max_passes: int = 5,
    max_dist_schedule: Sequence[float] | None = None,
    min_changed_fraction: float = 0.01,
    verbose: bool = True,
    **kwargs: Any,
) -> BundleResult:
    """Bundle adjustment alternating with reference-cloud correspondence search.

    This is ICP's E/M split around the bundle: freeze the correspondences, solve
    to convergence, re-search, repeat. It exists because correspondences must
    *not* be refreshed inside the solve - Ceres's trust region assumes a fixed
    cost function, so mutating the residuals between evaluations makes step
    quality meaningless and the solver oscillates. Refreshing between solves is
    both safe and sufficient: after the first pass the points move by
    centimetres.

    A fresh adjuster is built per pass, which is also what keeps the
    single-adjuster invariant of `run_bundle_adjustment` intact.

    Args:
        reconstruction: Adjusted in place.
        reference_cloud: The precomputed reference cloud.
        reference_options: Gates for the correspondence search. None uses the
            defaults.
        max_dist_schedule: Per-pass override of `reference_options.max_dist`,
            metres, for coarse-to-fine convergence. The last entry is reused for
            any remaining passes. Standard ICP practice, and necessary whenever
            the block starts further out than the final gate: open wide enough to
            capture the initial misalignment, then tighten so the late passes are
            not dragged by the wrong-surface matches the wide gate let in. None
            keeps one gate throughout.
        max_passes: Hard cap on outer passes.
        min_changed_fraction: Stop once fewer than this fraction of the
            correspondences changed target between passes. The natural
            convergence test: the correspondence *set*, not the cost.
        verbose: Print a per-pass line.
        **kwargs: Forwarded to `run_bundle_adjustment` (`solver`, `markers`,
            `camera_poses`, `filters`, `tie_points_sigma_px`, `covariance`, ...).

    Returns:
        The `BundleResult` of the final pass, with `passes` holding one record
        per pass: `pass`, `max_dist`, `n_used`, `changed`, `changed_fraction`,
        `median_surface_distance`, `converged`, `seconds`.

    Raises:
        ValueError: If `max_passes` is below 1 or `max_dist_schedule` is empty.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    if max_dist_schedule is not None and len(max_dist_schedule) == 0:
        raise ValueError("max_dist_schedule must not be empty")

    base = reference_options or ReferenceCloudConstraints()
    records: list[dict[str, Any]] = []
    previous: dict[int, int] = {}
    result: BundleResult | None = None

    for index in range(max_passes):
        started = time.perf_counter()
        options = base
        if max_dist_schedule is not None:
            # Past the end of the schedule, hold the final gate.
            gate = max_dist_schedule[min(index, len(max_dist_schedule) - 1)]
            options = replace(base, max_dist=gate)
        result = run_bundle_adjustment(
            reconstruction,
            reference_cloud=reference_cloud,
            reference_options=options,
            verbose=verbose,
            **kwargs,
        )
        stats = result.correspondences
        current = (
            dict(
                zip(
                    stats.point3D_ids.tolist(),
                    stats.reference_ids.tolist(),
                    strict=True,
                )
            )
            if stats is not None
            else {}
        )
        # A point whose target patch changed - or which entered or left the set -
        # counts as changed. Comparing the assignment is a sharper convergence
        # test than the cost, which keeps drifting long after it has settled.
        # Count *points*, not (point, patch) pairs: the symmetric difference of
        # the pairs double-counts every reassignment and can report over 100 %.
        involved = current.keys() | previous.keys()
        changed = sum(current.get(pid) != previous.get(pid) for pid in involved)
        fraction = changed / max(len(involved), 1)
        elapsed = time.perf_counter() - started

        records.append({
            "pass": index,
            "max_dist": options.max_dist,
            "n_used": len(current),
            "changed": changed,
            "changed_fraction": fraction,
            # To the local plane, not to the nearest sample: the sample
            # distance has a floor of half the point spacing and barely
            # moves, so it would hide whether the block is converging.
            "median_surface_distance": (
                float(np.median(stats.surface_distances))
                if stats is not None and stats.n_used
                else float("nan")
            ),
            "converged": result.converged,
            "seconds": elapsed,
        })
        if verbose:
            print(
                f"  pass {index}: {len(current):,} constrained at "
                f"{options.max_dist:.1f} m gate, "
                f"{fraction:.1%} correspondences changed, {elapsed:.1f} s"
            )

        previous = current
        # Never stop while the gate is still being tightened: correspondences
        # settling at a coarse gate says nothing about the fine one, and
        # stopping there would lock in the coarse solution.
        on_final_gate = max_dist_schedule is None or index >= len(max_dist_schedule) - 1
        if index > 0 and on_final_gate and fraction < min_changed_fraction:
            break

    assert result is not None  # max_passes >= 1 guarantees at least one solve
    result.passes = records
    return result
