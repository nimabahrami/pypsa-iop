"""Hand-rolled QP solver wrapper around HiGHS.

Solves the standard convex QP

    min  (1/2) · xᵀ Q x  +  qᵀ x
    s.t. A_eq · x  =  b_eq
         lb  ≤  x  ≤  ub

using :mod:`highspy` directly. This is the substrate every inverse-OPF
formulation targets — the package solves inverse OPF with only
``numpy``, ``scipy.sparse`` and ``highspy`` installed.

The :func:`solve_qp` interface is intentionally narrow (six numpy/scipy
arguments, one result dataclass) so a formulation builder can construct
its matrices, hand them off, and read the answer back without ever
seeing HiGHS's C-API conventions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Sentinel HiGHS uses for an unbounded variable bound.
_HIGHS_INF: float = 1e30


@dataclass(frozen=True)
class QPSolution:
    """Solver output.

    Attributes:
        x: Optimal primal vector, shape ``(n_vars,)``.
        objective: Optimal objective value (HiGHS sign convention,
            i.e. the value of ``(1/2) xᵀ Q x + qᵀ x``).
        status: HiGHS termination status as an upper-case string.
        is_optimal: ``True`` iff ``status == "OPTIMAL"``.
    """

    x: np.ndarray
    objective: float
    status: str
    is_optimal: bool


def solve_qp(
    *,
    Q: sp.csc_matrix | sp.csc_array | np.ndarray,
    q: np.ndarray,
    A_eq: sp.csc_matrix | sp.csc_array | np.ndarray | None,
    b_eq: np.ndarray | None,
    lb: np.ndarray,
    ub: np.ndarray,
    verbose: bool = False,
) -> QPSolution:
    """Solve a convex QP via HiGHS.

    Args:
        Q: Hessian (symmetric, positive-semidefinite). The full
            symmetric matrix; the wrapper converts it to the upper-
            triangular CSC form HiGHS expects.
        q: Linear objective coefficient, shape ``(n_vars,)``.
        A_eq: Equality constraint matrix, shape ``(n_eq, n_vars)``.
            Pass ``None`` for a bound-only problem.
        b_eq: Equality RHS, shape ``(n_eq,)``. Must match ``A_eq``.
        lb, ub: Per-variable lower / upper bounds. Use ``±np.inf``
            for unbounded; the wrapper converts to HiGHS's sentinel.
        verbose: When ``True``, leave HiGHS's solver log on stdout.

    Returns:
        A :class:`QPSolution`.

    Raises:
        ValueError: If shapes are inconsistent or ``Q`` is not square.
    """
    import highspy

    n_vars = q.shape[0]
    _validate_shapes(Q=Q, q=q, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, n_vars=n_vars)

    Q_csc = sp.csc_matrix(Q) if not sp.issparse(Q) else Q.tocsc()
    Q_upper = sp.triu(Q_csc, k=0).tocsc()
    A_csc = _build_constraint_matrix(A_eq, n_vars)

    lp = _build_highs_lp(
        n_vars=n_vars,
        n_eq=A_csc.shape[0],
        q=q,
        lb=lb,
        ub=ub,
        b_eq=b_eq,
        A_csc=A_csc,
    )
    hessian = _build_highs_hessian(Q_upper, n_vars, highspy)

    solver = highspy.Highs()
    if not verbose:
        solver.silent()
    solver.passModel(lp)
    solver.passHessian(hessian)
    solver.run()
    solution = solver.getSolution()
    info = solver.getInfo()

    status = str(solver.getModelStatus()).split(".")[-1].lstrip("k").upper()
    return QPSolution(
        x=np.asarray(solution.col_value, dtype=float),
        objective=float(info.objective_function_value),
        status=status,
        is_optimal=(status == "OPTIMAL"),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_shapes(
    *,
    Q,
    q: np.ndarray,
    A_eq,
    b_eq: np.ndarray | None,
    lb: np.ndarray,
    ub: np.ndarray,
    n_vars: int,
) -> None:
    if Q.shape != (n_vars, n_vars):
        raise ValueError(f"Q must be ({n_vars}, {n_vars}); got {Q.shape}")
    if lb.shape != (n_vars,) or ub.shape != (n_vars,):
        raise ValueError(
            f"lb/ub must have shape ({n_vars},); got {lb.shape}, {ub.shape}"
        )
    if A_eq is not None:
        if b_eq is None:
            raise ValueError("A_eq supplied without b_eq")
        if A_eq.shape[1] != n_vars:
            raise ValueError(
                f"A_eq has {A_eq.shape[1]} columns; need {n_vars}"
            )
        if b_eq.shape != (A_eq.shape[0],):
            raise ValueError(
                f"b_eq shape {b_eq.shape} does not match A_eq rows {A_eq.shape[0]}"
            )


def _build_constraint_matrix(
    A_eq: sp.csc_matrix | sp.csc_array | np.ndarray | None,
    n_vars: int,
) -> sp.csc_matrix:
    if A_eq is None:
        return sp.csc_matrix((0, n_vars))
    if sp.issparse(A_eq):
        return A_eq.tocsc()
    return sp.csc_matrix(A_eq)


def _build_highs_lp(
    *,
    n_vars: int,
    n_eq: int,
    q: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    b_eq: np.ndarray | None,
    A_csc: sp.csc_matrix,
):
    """Build the LP container HiGHS' :func:`passModel` expects."""
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_eq
    lp.col_cost_ = np.asarray(q, dtype=float).tolist()
    lp.col_lower_ = _to_highs_bounds(lb, lower=True)
    lp.col_upper_ = _to_highs_bounds(ub, lower=False)

    if n_eq > 0 and b_eq is not None:
        rhs = np.asarray(b_eq, dtype=float).tolist()
        lp.row_lower_ = rhs
        lp.row_upper_ = rhs
    else:
        lp.row_lower_ = []
        lp.row_upper_ = []

    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A_csc.indptr.astype(int).tolist()
    lp.a_matrix_.index_ = A_csc.indices.astype(int).tolist()
    lp.a_matrix_.value_ = A_csc.data.astype(float).tolist()
    lp.a_matrix_.num_col_ = n_vars
    lp.a_matrix_.num_row_ = n_eq
    return lp


def _build_highs_hessian(Q_upper: sp.csc_matrix, n_vars: int, highspy):
    """Build the upper-triangular Hessian HiGHS' :func:`passHessian` expects."""
    hessian = highspy.HighsHessian()
    hessian.dim_ = n_vars
    hessian.format_ = highspy.HessianFormat.kTriangular
    hessian.start_ = Q_upper.indptr.astype(int).tolist()
    hessian.index_ = Q_upper.indices.astype(int).tolist()
    hessian.value_ = Q_upper.data.astype(float).tolist()
    return hessian


def _to_highs_bounds(bounds: np.ndarray, *, lower: bool) -> list[float]:
    """Map ``±np.inf`` onto HiGHS's :data:`_HIGHS_INF` sentinel."""
    arr = np.asarray(bounds, dtype=float).copy()
    if lower:
        arr[np.isneginf(arr)] = -_HIGHS_INF
        arr[np.isposinf(arr)] = _HIGHS_INF  # rare but safe
    else:
        arr[np.isposinf(arr)] = _HIGHS_INF
        arr[np.isneginf(arr)] = -_HIGHS_INF
    return arr.tolist()


__all__ = ["QPSolution", "solve_qp"]
