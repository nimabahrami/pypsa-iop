"""Tests for utils.susceptance — flow-limit and reactance recovery."""
from __future__ import annotations

import numpy as np

from pypsa_invopt.network import read_network
from pypsa_invopt.utils.susceptance import (
    recover_flow_limits,
    recover_line_susceptances,
)


def test_flow_limit_recovery_picks_max_observed(three_bus_network):
    """recover_flow_limits returns max |f*| × safety_margin on congested lines.

    Lines that were never flagged congested do not appear in the result —
    the caller is expected to leave their s_nom untouched in that case.
    """
    net_data = read_network(three_bus_network)
    T = 24
    flows = np.zeros((T, len(net_data.lines)))
    ln = net_data.lines[0]
    l_idx = net_data.line_index[ln]
    flows[:, l_idx] = 95.0  # near s_nom=100
    flows[5, l_idx] = 98.0  # peak

    congested = {ln: list(range(T)), net_data.lines[1]: [], net_data.lines[2]: []}

    s_nom_hat = recover_flow_limits(
        network_data=net_data,
        flows=flows,
        congested_per_line=congested,
        safety_margin=1.02,
    )

    # Only the congested line is recovered.
    assert set(s_nom_hat.keys()) == {ln}
    # Recovered value is the peak observed × safety margin.
    assert abs(s_nom_hat[ln] - 98.0 * 1.02) < 1e-6


def test_flow_limit_recovery_empty_when_no_congestion(three_bus_network):
    """No congested timesteps for any line → empty result dict."""
    net_data = read_network(three_bus_network)
    flows = np.zeros((4, len(net_data.lines)))
    congested = {ln: [] for ln in net_data.lines}
    assert recover_flow_limits(
        network_data=net_data,
        flows=flows,
        congested_per_line=congested,
    ) == {}


def _build_dc_consistent_flows(three_bus_network):
    """Generate DC-PF-consistent flow observations for the 3-bus net."""
    from pypsa_invopt.utils.ptdf import compute_ptdf
    net = three_bus_network.copy()
    net_data = read_network(net)
    true_x = {ln: float(net.lines.at[ln, "x"]) for ln in net.lines.index}
    ptdf_true = compute_ptdf(
        buses=net_data.buses,
        lines=net_data.lines,
        line_bus0=net_data.line_bus0,
        line_bus1=net_data.line_bus1,
        line_x=true_x,
    )
    T = 8
    rng = np.random.default_rng(0)
    inject = np.zeros((T, len(net_data.buses)))
    inject[:, 0] = 120.0
    inject[:, 1] = -80.0 + rng.normal(0, 5, T)
    inject[:, 2] = -40.0 + rng.normal(0, 5, T)
    flows = inject @ ptdf_true.T
    dispatch = np.zeros((T, len(net_data.generators)))
    dispatch[:, 0] = 120.0
    loads_per_bus = np.zeros((T, len(net_data.buses)))
    loads_per_bus[:, 1] = 80.0
    loads_per_bus[:, 2] = 40.0
    return net_data, true_x, flows, dispatch, loads_per_bus


def test_susceptance_recovery_correct_prior_returns_near_truth(three_bus_network):
    """With prior = truth and DC-PF-consistent data, recovered x ≈ truth.

    The DC-PF model has a gauge symmetry (b, θ) → (α b, θ/α) that the
    prior anchor breaks; with prior = truth the anchored optimum *is*
    the truth, so the NLP should converge there.
    """
    net_data, true_x, flows, dispatch, loads = _build_dc_consistent_flows(
        three_bus_network
    )
    # prior = truth (no perturbation of net_data.line_x)

    x_hat = recover_line_susceptances(
        network_data=net_data,
        flows=flows,
        dispatch=dispatch,
        loads_per_bus=loads,
        sample_size=None,
        prior_weight=0.01,
    )

    # Recovery may exclude lines that hit a bound; lines that come back
    # must be within 10% of truth.
    assert x_hat, "expected at least one line to be recovered"
    for ln, x_val in x_hat.items():
        rel_err = abs(x_val - true_x[ln]) / true_x[ln]
        assert rel_err < 0.1, (
            f"Line {ln}: recovered x={x_val:.4f}, truth={true_x[ln]:.4f} "
            f"(relative error {rel_err:.1%})"
        )


def test_susceptance_recovery_biased_prior_stays_in_bounds(three_bus_network):
    """With a 1.5×-off prior, recovered x respects bounds.

    The gauge ambiguity means the NLP cannot identify the absolute
    susceptance scale without help. The prior anchor selects *some*
    consistent gauge — we verify only that the result is well-defined
    (positive, finite, inside the configured bounds), not that it
    recovers the true value, which would require an information source
    we do not have here.
    """
    net_data, true_x, flows, dispatch, loads = _build_dc_consistent_flows(
        three_bus_network
    )
    for ln in net_data.lines:
        net_data.line_x[ln] = 1.5 * true_x[ln]

    x_hat = recover_line_susceptances(
        network_data=net_data,
        flows=flows,
        dispatch=dispatch,
        loads_per_bus=loads,
        sample_size=None,
        prior_weight=0.1,
    )

    for ln, x_val in x_hat.items():
        assert np.isfinite(x_val)
        assert 0.25 * (1.5 * true_x[ln]) <= x_val <= 4.0 * (1.5 * true_x[ln])
