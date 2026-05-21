"""Single source of numerical thresholds and penalties.

All tunable tolerance constants used across the package are collected
here. They are intentionally module-level (not config-class fields)
because they are *not* user-facing — they are stable choices grounded
in solver behaviour, KKT theory, or finite-precision arithmetic.

If you find yourself tuning one of these to make a calibration
converge, the right fix is almost always to clean the inputs
(``pio.assess_data_quality``) or relax ``obs_sigma``, not to nudge
these numbers.
"""
from __future__ import annotations

# --- Active-set classification ---------------------------------------
# Per-storage / per-link / per-store snapshot bound: a state is marked
# "at bound" iff |state - bound| <= max(_INTERTEMPORAL_BOUND_TOL_MW,
# _INTERTEMPORAL_BOUND_TOL_REL * p_nom).  Absolute + relative gate so
# the rule scales sensibly across MW magnitudes.
INTERTEMPORAL_BOUND_TOL_MW: float = 0.5
INTERTEMPORAL_BOUND_TOL_REL: float = 0.001    # 0.1 % of p_nom

# Global constraint (e.g. CO2 cap) is "binding" when the realised value
# is within this fraction of the cap.
GLOBAL_CONSTRAINT_BINDING_TOL: float = 0.02

# --- Zonal formulation ----------------------------------------------
# Tiny Tikhonov on the zonal congestion-rent column so the KKT QP is
# strictly convex even when no zonal constraint is active.
ZONAL_MU_REGULARISER: float = 1e-3

# --- BLUE aggregation ------------------------------------------------
# Lower bound on per-batch posterior variance before reciprocal
# weighting in the BLUE aggregator (avoids 1/0 blow-up when a
# parameter is perfectly pinned by a single batch).
BATCH_VARIANCE_FLOOR: float = 1e-6

# --- Susceptance estimator -------------------------------------------
# Penalty on the bus-balance residual when estimating per-line
# susceptance from observed flows + injections. High enough that the
# linear system effectively enforces balance to machine precision.
BALANCE_PENALTY: float = 1e6


__all__ = [
    "BALANCE_PENALTY",
    "BATCH_VARIANCE_FLOOR",
    "GLOBAL_CONSTRAINT_BINDING_TOL",
    "INTERTEMPORAL_BOUND_TOL_MW",
    "INTERTEMPORAL_BOUND_TOL_REL",
    "ZONAL_MU_REGULARISER",
]
