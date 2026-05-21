"""Solver configuration shared across backends."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Solver knobs threaded through :func:`pypsa_invopt.calibrate`.

    Attributes:
        solver: Solver label retained for logging. The native back end
            dispatches to HiGHS via :mod:`highspy` regardless; an
            ``"ipopt"`` value triggers the IPOPT-backed susceptance NLP.
        verbose: Print solver-side output.
    """

    solver: str = "highs"
    verbose: bool = False


__all__ = ["SolverConfig"]
