"""Identifiability metrics from the Laplace posterior — Gap 2.

Verifies the three signals from Stuart (2010) §2.4 / Brewer-Donovan
(2018): posterior σ, information-gain ratio, 95 % CI, and the binary
identifiability flag. The flag is the key trading-decision artefact —
"trust this recovered cost / don't trust this one."
"""
from __future__ import annotations

import numpy as np
import pytest

from pypsa_invopt import identifiability
from pypsa_invopt.results import PosteriorResult


def _make_posterior_with_known_cov(
    sigmas: list[float], means: list[float] | None = None,
) -> PosteriorResult:
    """Build a synthetic ``PosteriorResult`` with a diagonal covariance."""
    n = len(sigmas)
    cov = np.diag(np.array(sigmas) ** 2)
    keys = tuple(f"gen:g{i}:marginal_cost" for i in range(n))
    mu = means or [10.0 * (i + 1) for i in range(n)]
    return PosteriorResult(
        method="laplace",
        mean={k: float(v) for k, v in zip(keys, mu, strict=True)},
        cov=cov,
        parameter_order=keys,
    )


def test_information_gain_uniform_prior():
    """With a uniform prior of σ=5, posterior σ=0.5 gives gain = 0.9."""
    post = _make_posterior_with_known_cov(sigmas=[0.5, 5.0])
    report = identifiability(post, sigma_prior=5.0)
    g0 = report["gen:g0:marginal_cost"]
    g1 = report["gen:g1:marginal_cost"]
    assert g0.information_gain == pytest.approx(0.9, abs=1e-6)
    assert g1.information_gain == pytest.approx(0.0, abs=1e-6)


def test_ci_uses_z_score_1_96():
    """95 % CI = mean ± 1.96·σ_post for a Laplace posterior."""
    post = _make_posterior_with_known_cov(sigmas=[2.0], means=[20.0])
    report = identifiability(post, sigma_prior=5.0)
    g0 = report["gen:g0:marginal_cost"]
    assert g0.ci_low == pytest.approx(20.0 - 1.96 * 2.0, abs=1e-6)
    assert g0.ci_high == pytest.approx(20.0 + 1.96 * 2.0, abs=1e-6)


def test_identifiable_flag_requires_both_sigma_and_information_gain():
    """A parameter is flagged identifiable only when σ_post is below
    threshold AND information gain exceeds the minimum.
    """
    # Tight σ, high gain → identifiable
    tight = _make_posterior_with_known_cov(sigmas=[0.5])
    assert identifiability(
        tight, sigma_prior=5.0,
        sigma_threshold=2.0, min_information_gain=0.25,
    )["gen:g0:marginal_cost"].identifiable is True

    # Wide σ → not identifiable, even with high "gain" (vacuous because
    # the prior was wider still)
    wide = _make_posterior_with_known_cov(sigmas=[10.0])
    rep = identifiability(
        wide, sigma_prior=100.0,
        sigma_threshold=2.0, min_information_gain=0.25,
    )
    g0 = rep["gen:g0:marginal_cost"]
    assert g0.identifiable is False
    assert "σ_post" in g0.reason

    # Tight σ but tiny information gain → not identifiable (data is
    # confirming the prior, not learning from observations)
    tight_low_gain = _make_posterior_with_known_cov(sigmas=[0.45])
    rep = identifiability(
        tight_low_gain, sigma_prior=0.5,
        sigma_threshold=2.0, min_information_gain=0.5,
    )
    g0 = rep["gen:g0:marginal_cost"]
    assert g0.identifiable is False
    assert "information_gain" in g0.reason


def test_per_parameter_priors_via_dict():
    """The function accepts a dict of per-parameter prior σs."""
    post = _make_posterior_with_known_cov(sigmas=[0.5, 0.5])
    rep = identifiability(
        post,
        sigma_prior={
            "gen:g0:marginal_cost": 5.0,
            "gen:g1:marginal_cost": 0.6,
        },
    )
    # g0: σ_post 0.5 vs prior 5.0 → gain 0.9
    # g1: σ_post 0.5 vs prior 0.6 → gain 0.167
    assert rep["gen:g0:marginal_cost"].information_gain == pytest.approx(
        0.9, abs=1e-6,
    )
    assert rep["gen:g1:marginal_cost"].information_gain == pytest.approx(
        1.0 - 0.5 / 0.6, abs=1e-6,
    )


def test_mcmc_posterior_uses_empirical_quantiles():
    """When ``cov`` is ``None`` but ``samples`` is populated, the CI
    is computed from empirical 2.5 / 97.5 quantiles rather than ±1.96σ."""
    rng = np.random.default_rng(42)
    samples = {"gen:g0:marginal_cost": rng.normal(20.0, 2.0, size=10_000)}
    post = PosteriorResult(
        method="mcmc",
        mean={"gen:g0:marginal_cost": 20.0},
        cov=None,
        samples=samples,
        parameter_order=("gen:g0:marginal_cost",),
    )
    rep = identifiability(post, sigma_prior=5.0)
    g0 = rep["gen:g0:marginal_cost"]
    # CI should be approximately 20 ± 1.96·2 = (16.08, 23.92)
    assert g0.ci_low == pytest.approx(16.0, abs=0.2)
    assert g0.ci_high == pytest.approx(24.0, abs=0.2)
    assert g0.sigma_post == pytest.approx(2.0, abs=0.1)
