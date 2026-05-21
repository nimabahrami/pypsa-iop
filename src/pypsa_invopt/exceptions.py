"""Custom exceptions for pypsa-invopt."""
from __future__ import annotations


class InvoptInputError(ValueError):
    """Raised when observations / network input fails validation."""


class InvoptConvergenceError(RuntimeError):
    """Raised when the inverse-OPF QP fails to converge."""


__all__ = [
    "InvoptConvergenceError",
    "InvoptInputError",
]
