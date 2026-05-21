"""DC power-transfer-distribution-factor (PTDF) matrix.

PTDF maps nodal injections to line flows under the DC-power-flow
approximation:

    f_l  =  Σ_b  PTDF[l, b] · P_inject[b]

It is built from the network's susceptance matrices

    PTDF  =  B_f · B_θ⁻¹     (with the slack-bus column dropped)

where ``B_θ`` is the bus admittance matrix and ``B_f`` is the
branch-bus incidence weighted by line susceptance. See the PyPSA paper
for the equivalent derivation.

Reference:
    Brown T., Hörsch J., Schlachtberger D. (2018). PyPSA: Python for
    Power System Analysis. *JORS* 6(4). DOI: 10.5334/jors.188.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def compute_ptdf(
    *,
    buses: list[str],
    lines: list[str],
    line_bus0: dict[str, str],
    line_bus1: dict[str, str],
    line_x: dict[str, float],
) -> np.ndarray:
    """Compute the PTDF matrix for a DC-power-flow network.

    Args:
        buses: Ordered list of bus names. The first bus is treated as
            the slack and gets a zero PTDF column by construction.
        lines: Ordered list of line names.
        line_bus0: Line → sending bus.
        line_bus1: Line → receiving bus.
        line_x: Line → reactance (pu).

    Returns:
        PTDF matrix of shape ``(n_lines, n_buses)``.

    Raises:
        ValueError: If any line has non-positive reactance.
    """
    n_buses = len(buses)
    n_lines = len(lines)
    if n_buses == 0 or n_lines == 0:
        return np.zeros((n_lines, n_buses))
    if n_buses == 1:
        return np.zeros((n_lines, n_buses))

    bus_idx = {b: i for i, b in enumerate(buses)}
    b_line = _line_susceptances(lines, line_x)

    b_theta = _bus_admittance_matrix(
        n_buses=n_buses,
        lines=lines,
        line_bus0=line_bus0,
        line_bus1=line_bus1,
        bus_idx=bus_idx,
        b_line=b_line,
    )
    b_f = _branch_bus_incidence(
        n_lines=n_lines,
        n_buses=n_buses,
        lines=lines,
        line_bus0=line_bus0,
        line_bus1=line_bus1,
        bus_idx=bus_idx,
        b_line=b_line,
    )

    # Drop the slack-bus row/column from B_θ and the slack column from
    # B_f, solve B_θ_reduced @ PTDFᵀ = B_f_reducedᵀ, then re-insert the
    # zero slack column.
    non_slack = list(range(1, n_buses))
    b_theta_reduced = b_theta[np.ix_(non_slack, non_slack)].tocsc()
    b_f_reduced = b_f[:, non_slack].toarray()

    ptdf_reduced = np.zeros((n_lines, n_buses - 1))
    for l_idx in range(n_lines):
        ptdf_reduced[l_idx, :] = spsolve(b_theta_reduced, b_f_reduced[l_idx, :])

    ptdf = np.zeros((n_lines, n_buses))
    ptdf[:, 1:] = ptdf_reduced
    return ptdf


def validate_ptdf(ptdf: np.ndarray, tol: float = 1e-6) -> bool:
    """Sanity-check a PTDF matrix for a well-conditioned DC network.

    Returns ``False`` if the slack column is non-zero, if any entry is
    non-finite, or if any entry exceeds ``1 + tol`` in magnitude.
    """
    if ptdf.size == 0:
        return True
    if not np.all(np.abs(ptdf[:, 0]) < tol):
        return False
    if not np.all(np.isfinite(ptdf)):
        return False
    return not np.any(np.abs(ptdf) > 1.0 + tol)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _line_susceptances(
    lines: list[str],
    line_x: dict[str, float],
) -> np.ndarray:
    """Return ``b_l = 1/x_l`` as a dense vector."""
    b_line = np.zeros(len(lines))
    for i, ln in enumerate(lines):
        x = line_x[ln]
        if x <= 0:
            raise ValueError(
                f"Line '{ln}' has non-positive reactance x={x}. "
                "PTDF computation requires x > 0."
            )
        b_line[i] = 1.0 / x
    return b_line


def _bus_admittance_matrix(
    *,
    n_buses: int,
    lines: list[str],
    line_bus0: dict[str, str],
    line_bus1: dict[str, str],
    bus_idx: dict[str, int],
    b_line: np.ndarray,
) -> sparse.csr_matrix:
    """Build the sparse bus admittance matrix ``B_θ``.

    Off-diagonal entry ``B_θ[i, j] = -b_l`` for each line ``(i, j)``;
    diagonal entries accumulate ``+b_l`` for every incident line.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, ln in enumerate(lines):
        b0 = bus_idx[line_bus0[ln]]
        b1 = bus_idx[line_bus1[ln]]
        bl = b_line[i]
        rows.extend([b0, b1, b0, b1])
        cols.extend([b1, b0, b0, b1])
        data.extend([-bl, -bl, bl, bl])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_buses, n_buses))


def _branch_bus_incidence(
    *,
    n_lines: int,
    n_buses: int,
    lines: list[str],
    line_bus0: dict[str, str],
    line_bus1: dict[str, str],
    bus_idx: dict[str, int],
    b_line: np.ndarray,
) -> sparse.csr_matrix:
    """Build the branch-bus incidence matrix weighted by susceptance.

    ``B_f[l, b] = +b_l`` at bus0, ``-b_l`` at bus1, ``0`` elsewhere.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, ln in enumerate(lines):
        b0 = bus_idx[line_bus0[ln]]
        b1 = bus_idx[line_bus1[ln]]
        bl = b_line[i]
        rows.extend([i, i])
        cols.extend([b0, b1])
        data.extend([bl, -bl])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_lines, n_buses))


__all__ = ["compute_ptdf", "validate_ptdf"]
