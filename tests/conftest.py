"""Shared fixtures: the synthetic truth and a perturbed copy of it."""

from __future__ import annotations

import pycolmap
import pytest
from synthetic import ROT_DEG, SCALE_PPM, SHIFT_M, synthesize

from pycolmap_ba import SolverOptions
from pycolmap_ba.utils.perturb import Perturbation, perturb_reconstruction


@pytest.fixture(scope="session")
def truth() -> pycolmap.Reconstruction:
    """The untouched synthetic block. Never adjust it: copy it first."""
    return synthesize()


@pytest.fixture
def block(truth: pycolmap.Reconstruction) -> pycolmap.Reconstruction:
    """A copy of the truth, moved by a 7-DoF similarity the tie points cannot see."""
    reconstruction = pycolmap.Reconstruction(truth)
    perturb_reconstruction(
        reconstruction,
        Perturbation(shift_m=SHIFT_M, rot_deg=ROT_DEG, scale_ppm=SCALE_PPM, seed=1),
    )
    return reconstruction


@pytest.fixture
def solver() -> SolverOptions:
    """Intrinsics held fixed, so every test isolates the datum."""
    return SolverOptions(refine_focal_length=False)
