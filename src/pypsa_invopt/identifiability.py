"""Per-parameter identifiability metrics from the Laplace posterior.

Three signals per parameter:

1. **Posterior σ** — ``√Σ_ii``.
2. **Information gain** — ``1 − σ_post / σ_prior`` ∈ ``[0, 1]``.
3. **95% credible interval** — ``θ̂ ± 1.96·σ_post`` (Laplace Gaussian).

A binary ``identifiable`` flag is ``True`` iff ``σ_post`` is below
``sigma_threshold`` AND information gain exceeds
``min_information_gain``. Both checks together because either alone
admits pathologies (very wide prior → high "gain" but still wide
posterior; flat prior → low gain even when posterior is tight).

References: Stuart (2010) *Acta Numerica* 19 §2.4 (posterior-σ as
identifiability signal); Brewer & Donovan (2018) *Statistical
Modelling* 18(2) (D-optimality / information gain); Liang-Dvorkin
(2023) §5.2 (marginal-only identifiability in inverse OPF).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pypsa_invopt.results import PosteriorResult


@dataclass(frozen=True)
class ParameterIdentifiability:
    """Per-parameter identifiability bundle.

    Attributes:
        sigma_post: Posterior standard deviation (the diagonal of the
            Laplace covariance).
        sigma_prior: Prior standard deviation used during calibration.
        information_gain: ``1 − σ_post / σ_prior`` ∈ [0, 1].
        ci_low / ci_high: 95 % credible interval ``θ̂ ± 1.96·σ_post``.
        identifiable: ``True`` iff the parameter is sufficiently
            pinned-down by the data (see :func:`compute_identifiability`).
        reason: Short explanation when ``identifiable`` is ``False``
            (empty string when identifiable).
    """

    sigma_post: float
    sigma_prior: float
    information_gain: float
    ci_low: float
    ci_high: float
    identifiable: bool
    reason: str = ""


def compute_identifiability(
    posterior: PosteriorResult,
    *,
    sigma_prior: float | dict[str, float] = 5.0,
    sigma_threshold: float = 2.0,
    min_information_gain: float = 0.25,
    z_score: float = 1.96,
) -> dict[str, ParameterIdentifiability]:
    """Compute the per-parameter identifiability bundle.

    Args:
        posterior: A :class:`PosteriorResult` from the Laplace path
            (i.e. ``cov`` is populated).
        sigma_prior: The prior std used during calibration. Pass a
            single float for a uniform prior, or a dict keyed by
            parameter name for per-parameter priors. Default 5.0 —
            matches the package's standard Laplace prior.
        sigma_threshold: A parameter is flagged identifiable only when
            ``σ_post ≤ sigma_threshold`` (absolute units; EUR/MWh for
            generator costs). Default 2.0 EUR/MWh — chosen as a
            conservative "actionable" level for European day-ahead
            markets where typical LMP variation across hours is
            10–50 EUR/MWh; a parameter with σ_post above 2 EUR/MWh has
            uncertainty comparable to a sizeable fraction of the
            actionable signal. Tune for your use case.
        min_information_gain: A parameter is flagged identifiable only
            when ``information_gain ≥ this``. Default 0.25 — at least
            25 % of the prior uncertainty must be resolved by data.
            Below this threshold the data has barely shifted the
            posterior off the prior, so the "recovery" is essentially
            the prior. Brewer-Donovan 2018 use information gain as a
            D-optimality signal but don't prescribe a threshold; 0.25
            is a project-specific convention. Tune for your use case.
        z_score: Standard-normal multiplier for the credible interval.
            Default 1.96 (95 % CI).

    Returns:
        Dict mapping parameter name → :class:`ParameterIdentifiability`.
        Parameter names match :attr:`PosteriorResult.parameter_order`
        and the keys of :attr:`InverseResult.theta_hat`.
    """
    if posterior.cov is None:
        # MCMC posterior: derive σ from the empirical samples directly.
        if posterior.samples is None:
            raise ValueError(
                "PosteriorResult has neither covariance nor samples — "
                "cannot compute identifiability."
            )
        return _from_mcmc(
            posterior=posterior,
            sigma_prior=sigma_prior,
            sigma_threshold=sigma_threshold,
            min_information_gain=min_information_gain,
            z_score=z_score,
        )

    return _from_laplace(
        posterior=posterior,
        sigma_prior=sigma_prior,
        sigma_threshold=sigma_threshold,
        min_information_gain=min_information_gain,
        z_score=z_score,
    )


def _from_laplace(
    *,
    posterior: PosteriorResult,
    sigma_prior: float | dict[str, float],
    sigma_threshold: float,
    min_information_gain: float,
    z_score: float,
) -> dict[str, ParameterIdentifiability]:
    cov = posterior.cov
    assert cov is not None
    sigmas = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out: dict[str, ParameterIdentifiability] = {}
    for i, key in enumerate(posterior.parameter_order):
        s_post = float(sigmas[i])
        s_prior = _prior_lookup(key, sigma_prior)
        mean = float(posterior.mean.get(key, 0.0))
        info_gain = _safe_information_gain(s_post, s_prior)
        identifiable, reason = _flag(
            sigma_post=s_post,
            information_gain=info_gain,
            sigma_threshold=sigma_threshold,
            min_information_gain=min_information_gain,
        )
        out[key] = ParameterIdentifiability(
            sigma_post=s_post,
            sigma_prior=s_prior,
            information_gain=info_gain,
            ci_low=mean - z_score * s_post,
            ci_high=mean + z_score * s_post,
            identifiable=identifiable,
            reason=reason,
        )
    return out


def _from_mcmc(
    *,
    posterior: PosteriorResult,
    sigma_prior: float | dict[str, float],
    sigma_threshold: float,
    min_information_gain: float,
    z_score: float,
) -> dict[str, ParameterIdentifiability]:
    samples = posterior.samples
    assert samples is not None
    out: dict[str, ParameterIdentifiability] = {}
    for key in posterior.parameter_order:
        arr = samples.get(key)
        if arr is None or arr.size == 0:
            continue
        s_post = float(np.std(arr))
        s_prior = _prior_lookup(key, sigma_prior)
        info_gain = _safe_information_gain(s_post, s_prior)
        # Empirical CI from samples — more honest than ±1.96σ when
        # the posterior is non-Gaussian.
        ci_low, ci_high = (
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        )
        identifiable, reason = _flag(
            sigma_post=s_post,
            information_gain=info_gain,
            sigma_threshold=sigma_threshold,
            min_information_gain=min_information_gain,
        )
        out[key] = ParameterIdentifiability(
            sigma_post=s_post,
            sigma_prior=s_prior,
            information_gain=info_gain,
            ci_low=ci_low,
            ci_high=ci_high,
            identifiable=identifiable,
            reason=reason,
        )
    return out


def _prior_lookup(key: str, sigma_prior: float | dict[str, float]) -> float:
    if isinstance(sigma_prior, dict):
        return float(sigma_prior.get(key, max(sigma_prior.values()) if sigma_prior else 5.0))
    return float(sigma_prior)


def _safe_information_gain(sigma_post: float, sigma_prior: float) -> float:
    if sigma_prior <= 0:
        return 0.0
    gain = 1.0 - sigma_post / sigma_prior
    # Clamp to [0, 1] — numerical artefacts in σ_post can push slightly negative.
    return float(max(0.0, min(1.0, gain)))


def _flag(
    *,
    sigma_post: float,
    information_gain: float,
    sigma_threshold: float,
    min_information_gain: float,
) -> tuple[bool, str]:
    if sigma_post > sigma_threshold:
        return False, f"σ_post={sigma_post:.3f} > threshold {sigma_threshold:.3f}"
    if information_gain < min_information_gain:
        return False, (
            f"information_gain={information_gain:.2f} < "
            f"min {min_information_gain:.2f} (data carries little signal)"
        )
    return True, ""


__all__ = ["ParameterIdentifiability", "compute_identifiability"]
