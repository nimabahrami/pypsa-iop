"""Public API for pypsa-invopt.

Users import from this module; the implementation lives in submodules.
Every entry point lazily imports its target.

Scope: the package's job stops at *recovering bid costs from observed
market data* and reporting recovery uncertainty + identifiability.
Forecasting future markets and operational what-if simulations are
left to the user's own forward-DCOPF / trading pipeline — feed
``result.theta_hat`` (or call ``pio.apply(result, network)``) into
that pipeline as the calibrated parameter vector.

Typical usage::

    import pypsa_invopt as pio

    result    = pio.calibrate(network=n, observations=obs)
    pio.apply(result, n)                            # write θ̂ back
    posterior = pio.posterior(network=n, observations=obs, result=result)
    report    = pio.identifiability(posterior)      # trustworthy params
    flags     = pio.flag_withholding(...)            # market-monitoring
"""
from __future__ import annotations

from typing import Any, Literal

from pypsa_invopt._version import __version__
from pypsa_invopt.exceptions import (
    InvoptConvergenceError,
    InvoptInputError,
)
from pypsa_invopt.reference_costs import ReferenceCost
from pypsa_invopt.results import (
    InverseResult,
    PosteriorResult,
)

FormulationType = Literal["noiseless", "noisy", "zonal"]


def calibrate(*args: Any, **kwargs: Any) -> InverseResult:
    """Calibrate network parameters via inverse OPF.

    See :func:`pypsa_invopt.calibration.calibrate` for the full signature.
    """
    from pypsa_invopt.calibration import calibrate as _impl
    return _impl(*args, **kwargs)


def apply(*args: Any, **kwargs: Any) -> None:
    """Write recovered parameters back to a ``pypsa.Network`` in place.

    See :func:`pypsa_invopt.network.apply_result`.
    """
    from pypsa_invopt.network import apply_result
    apply_result(*args, **kwargs)


def observations_from_pypsa(*args: Any, **kwargs: Any):
    """Build the observation DataFrame ``calibrate`` expects from a
    solved ``pypsa.Network``.

    Convenience wrapper that walks ``buses_t.marginal_price``,
    ``lines_t.p0``, ``generators_t.p`` and the storage / link / store
    time series and emits all the ``price_<bus>``, ``flow_<line>``,
    ``dispatch_<gen>``, ``storage_*``, ``link_*``, ``store_*`` columns.

    See :func:`pypsa_invopt.network.observations_from_pypsa`.
    """
    from pypsa_invopt.network import observations_from_pypsa as _impl
    return _impl(*args, **kwargs)


def posterior(*args: Any, **kwargs: Any) -> PosteriorResult:
    """Compute the Bayesian posterior over recovered parameters.

    See :func:`pypsa_invopt.bayes.posterior`.
    """
    from pypsa_invopt.bayes import posterior as _impl
    return _impl(*args, **kwargs)


def identifiability(*args: Any, **kwargs: Any):
    """Compute per-parameter identifiability metrics from a posterior.

    Standard signals from the inverse-problem literature: posterior
    σ, information-gain ratio, 95 % credible interval, and a binary
    flag suitable for filtering recovered parameters before they are
    used downstream.

    See :func:`pypsa_invopt.identifiability.compute_identifiability`.
    """
    from pypsa_invopt.identifiability import compute_identifiability as _impl
    return _impl(*args, **kwargs)


def flag_withholding(*args: Any, **kwargs: Any):
    """Compare recovered costs against engineering reference costs.

    Implements the Birge-Hortaçsu-Pavlin (2017) MISO market-monitoring
    use case: a generator whose recovered cost lies more than ``z``
    standard deviations above its fuel + CO2 + O&M reference is
    flagged as a strategic-withholding candidate.

    See :func:`pypsa_invopt.reference_costs.flag_withholding`.
    """
    from pypsa_invopt.reference_costs import flag_withholding as _impl
    return _impl(*args, **kwargs)


def compute_reference_cost(*args: Any, **kwargs: Any):
    """Compute the engineering-reference marginal cost of a generator.

    Returns ``fuel_price × heat_rate + co2_price × emission + var_O&M``
    as a :class:`ReferenceCost` dataclass. Use this when you want the
    reference number directly without going through the withholding
    scorer (e.g. for custom flagging rules).

    See :func:`pypsa_invopt.reference_costs.compute_reference_cost`.
    """
    from pypsa_invopt.reference_costs import compute_reference_cost as _impl
    return _impl(*args, **kwargs)


def load_entso_e(*args: Any, **kwargs: Any):
    """Load ENTSO-E market data.

    Requires ``pip install pypsa-invopt[entso_e]``. See
    :func:`pypsa_invopt.data.entso_e.load_entso_e`.
    """
    from pypsa_invopt.data.entso_e import load_entso_e as _impl
    return _impl(*args, **kwargs)


def validate_observations(*args: Any, **kwargs: Any) -> list[str]:
    """Pre-flight check on an observations DataFrame.

    Raises :class:`InvoptInputError` with a clear message if the
    DataFrame is missing required columns / has a non-temporal index /
    contains NaNs in required positions; otherwise returns a list of
    non-fatal warnings.

    Calling this is **optional** — :func:`calibrate` performs the
    minimum validation internally. Use this when you want explicit
    column requirements (e.g. "every bus must have a price column").

    See :func:`pypsa_invopt.utils.validate.validate_observations`.
    """
    from pypsa_invopt.utils.validate import validate_observations as _impl
    return _impl(*args, **kwargs)


def assess_data_quality(*args: Any, **kwargs: Any):
    """Quality report on a market-data DataFrame.

    Returns a :class:`pypsa_invopt.data.quality.DataQualityReport` with
    counts of missing values, negative prices, price spikes and the
    overall coverage percentage. Useful before feeding ENTSO-E /
    market-data downloads into :func:`calibrate` — gappy or spiky data
    silently degrades recovery quality.

    See :func:`pypsa_invopt.data.quality.assess_quality`.
    """
    from pypsa_invopt.data.quality import assess_quality as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "FormulationType",
    "InverseResult",
    "InvoptConvergenceError",
    "InvoptInputError",
    "PosteriorResult",
    "ReferenceCost",
    "__version__",
    "apply",
    "assess_data_quality",
    "calibrate",
    "compute_reference_cost",
    "flag_withholding",
    "identifiability",
    "load_entso_e",
    "observations_from_pypsa",
    "posterior",
    "validate_observations",
]
