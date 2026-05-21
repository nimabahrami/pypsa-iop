"""Inverse-OPF formulation sub-package.

Three QP-based formulations cover the data regimes encountered in
practice:

* :class:`NoiselessFormulation` — KKT-equality LP (active set fixed,
  observations assumed noiseless). Fast; use when LMPs / dispatch
  come straight from a forward LOPF.
* :class:`NoisyFormulation` — Aswani-style regularised KKT-residual QP
  with explicit observation noise σ. The canonical Liang-Dvorkin
  (2023) single-level reformulation. Use on real market data.
* :class:`ZonalFormulation` — bidding-zone level with NTC shadow
  prices. Use for EUPHEMIA-style European clearings where the nodal
  network is not published.

All three are built and solved via the project's own ``highspy``-direct
sparse QP back end (``solvers/qp.py``).
"""
from pypsa_invopt.formulations.base import InverseFormulation
from pypsa_invopt.formulations.noiseless import NoiselessFormulation
from pypsa_invopt.formulations.noisy import NoisyFormulation
from pypsa_invopt.formulations.zonal import ZonalFormulation

__all__ = [
    "InverseFormulation",
    "NoiselessFormulation",
    "NoisyFormulation",
    "ZonalFormulation",
]
