"""Active-set detection and ASTB clustering tests."""
import numpy as np

from pypsa_invopt.utils.active_set import (
    ActiveSet,
    cluster_active_sets,
    detect_active_set,
    detect_active_sets_temporal,
    jaccard_similarity,
)


def test_mu_based_active_set_detection_overrides_tolerance():
    """When PyPSA-reported shadow prices are supplied, the detector
    uses them directly instead of the ``|flow| ≥ s_nom − ε`` heuristic."""
    flows = np.array([[50.0, 10.0]])                     # 1 timestep × 2 lines
    flow_limits = np.array([100.0, 100.0])               # neither flow near limit
    dispatch = np.array([[40.0, 40.0]])                  # 1 timestep × 2 gens
    gen_max = np.array([100.0, 100.0])
    gen_min = np.array([0.0, 0.0])
    mu_lines = np.array([[12.5, 0.0]])                   # line 0 binding
    mu_gens_upper = np.array([[0.0, 5.0]])               # gen 1 at upper bound
    mu_gens_lower = np.array([[3.0, 0.0]])               # gen 0 at lower bound

    [as_t0] = detect_active_sets_temporal(
        flows=flows, flow_limits=flow_limits,
        dispatch=dispatch, gen_max=gen_max, gen_min=gen_min,
        line_names=["l0", "l1"], gen_names=["g0", "g1"],
        mu_lines=mu_lines, mu_gens_upper=mu_gens_upper,
        mu_gens_lower=mu_gens_lower,
    )
    assert as_t0.congested_lines == frozenset({"l0"})
    assert as_t0.maxed_generators == frozenset({"g1"})
    assert as_t0.min_bound_generators == frozenset({"g0"})


def test_offline_generator_filtered_from_active_set():
    """A generator with ``p_max ≤ eps`` at a snapshot is offline
    (outage or commitment OFF). The KKT row for that (g, t) carries
    no information — it must be excluded from both the maxed and the
    min-bound active sets, otherwise ν or ξ absorbs arbitrary value.
    """
    flows = np.array([[10.0], [10.0]])
    flow_limits = np.array([100.0])
    dispatch = np.array([
        [50.0, 0.0],     # snapshot 0: gen0 dispatching, gen1 OFFLINE
        [60.0, 30.0],    # snapshot 1: both dispatching
    ])
    # Time-varying gen_max: gen1 has p_max=0 in snapshot 0 (offline)
    gen_max = np.array([
        [100.0, 0.0],
        [100.0, 50.0],
    ])
    gen_min = np.zeros_like(gen_max)
    as_list = detect_active_sets_temporal(
        flows=flows, flow_limits=flow_limits,
        dispatch=dispatch, gen_max=gen_max, gen_min=gen_min,
        line_names=["l0"], gen_names=["g0", "g1"],
        eps=0.5,
    )
    # Snapshot 0: gen1 is offline (p_max ≈ 0). Even though dispatch
    # (0) ≤ p_max (0) + eps, it should NOT appear in maxed or min.
    assert "g1" not in as_list[0].maxed_generators
    assert "g1" not in as_list[0].min_bound_generators
    # Snapshot 1: gen1 dispatches 30 of 50 → interior, neither bound.
    assert "g1" not in as_list[1].maxed_generators
    assert "g1" not in as_list[1].min_bound_generators


def test_tolerance_path_still_works_without_mu():
    """Without ``mu_*`` arrays, the detector falls back to the
    tolerance-based heuristic — keeping legacy callers unchanged."""
    flows = np.array([[99.9]])
    flow_limits = np.array([100.0])
    dispatch = np.array([[99.95]])
    gen_max = np.array([100.0])
    gen_min = np.array([0.0])
    [as_t0] = detect_active_sets_temporal(
        flows=flows, flow_limits=flow_limits,
        dispatch=dispatch, gen_max=gen_max, gen_min=gen_min,
        line_names=["l0"], gen_names=["g0"],
        eps=0.5,   # |flow| ≥ 99.5 → congested
    )
    assert as_t0.congested_lines == frozenset({"l0"})
    assert as_t0.maxed_generators == frozenset({"g0"})


def test_detect_active_set_no_congestion():
    """No constraints binding → empty active set."""
    aset = detect_active_set(
        flows=np.array([10.0, 20.0]),
        flow_limits=np.array([100.0, 100.0]),
        dispatch=np.array([50.0, 30.0]),
        gen_max=np.array([100.0, 100.0]),
        gen_min=np.array([0.0, 0.0]),
        line_names=["L1", "L2"],
        gen_names=["G1", "G2"],
    )
    assert len(aset.congested_lines) == 0
    assert len(aset.maxed_generators) == 0


def test_detect_active_set_with_congestion():
    """Line at limit → congested."""
    aset = detect_active_set(
        flows=np.array([99.5, 20.0]),
        flow_limits=np.array([100.0, 100.0]),
        dispatch=np.array([50.0, 99.8]),
        gen_max=np.array([100.0, 100.0]),
        gen_min=np.array([0.0, 0.0]),
        line_names=["L1", "L2"],
        gen_names=["G1", "G2"],
        eps=1.0,
    )
    assert "L1" in aset.congested_lines
    assert "L2" not in aset.congested_lines
    assert "G2" in aset.maxed_generators


def test_cluster_identical_patterns():
    """Identical patterns cluster into one batch."""
    as1 = ActiveSet(congested_lines=frozenset({"L1"}))
    as2 = ActiveSet(congested_lines=frozenset({"L1"}))
    as3 = ActiveSet(congested_lines=frozenset())

    batches = cluster_active_sets([as1, as2, as3])
    assert len(batches) == 2
    # Largest batch first
    assert len(batches[0].timestep_indices) == 2
    assert len(batches[1].timestep_indices) == 1


def test_cluster_all_unique():
    """All different → K == T."""
    active_sets = [
        ActiveSet(congested_lines=frozenset({f"L{i}"}))
        for i in range(10)
    ]
    batches = cluster_active_sets(active_sets)
    assert len(batches) == 10


def test_astb_compression_synthetic():
    """Verify K << T for realistic synthetic data."""
    rng = np.random.default_rng(42)
    T = 8760
    n_lines = 10
    n_gens = 5

    # Generate flows where only a few lines congest
    flows = rng.normal(30, 10, (T, n_lines))
    flow_limits = np.full(n_lines, 100.0)
    flow_limits[:2] = 40.0  # These two congest sometimes

    dispatch = rng.uniform(20, 80, (T, n_gens))
    gen_max = np.full(n_gens, 100.0)
    gen_min = np.zeros(n_gens)

    line_names = [f"L{i}" for i in range(n_lines)]
    gen_names = [f"G{i}" for i in range(n_gens)]

    active_sets = detect_active_sets_temporal(
        flows=flows, flow_limits=flow_limits,
        dispatch=dispatch, gen_max=gen_max, gen_min=gen_min,
        line_names=line_names, gen_names=gen_names,
    )
    batches = cluster_active_sets(active_sets)

    # K should be much less than T
    assert len(batches) < T
    # Total timesteps across batches = T
    total = sum(len(b.timestep_indices) for b in batches)
    assert total == T


def test_jaccard_similarity():
    """Jaccard similarity between active sets."""
    a = ActiveSet(congested_lines=frozenset({"L1", "L2"}))
    b = ActiveSet(congested_lines=frozenset({"L1", "L3"}))
    c = ActiveSet(congested_lines=frozenset({"L1", "L2"}))

    assert jaccard_similarity(a, c) == 1.0
    assert 0.0 < jaccard_similarity(a, b) < 1.0
    assert jaccard_similarity(ActiveSet(), ActiveSet()) == 1.0


def test_astb_multibatch_calibrate(two_bus_network):
    """Calibration with multiple distinct active-set patterns runs
    end-to-end through the multi-batch ASTB path.

    The two-bus fixture has two lines / two generators, and we
    construct observations whose patterns vary across timesteps. The
    calibrator should cluster them into multiple batches, solve each,
    and BLUE-aggregate the per-batch θ estimates.
    """
    import pandas as pd

    from pypsa_invopt.calibration import calibrate

    T = 12
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    rng = np.random.default_rng(3)

    # Construct active-set patterns by alternating regimes. The active-
    # set detector marks a line as congested when |flow| >= s_nom - eps,
    # so we put some steps *at* the limit and others well inside it.
    flows = np.full((T, 1), 30.0)
    flows[::3, 0] = 100.0  # at s_nom=100 → congested
    dispatch = np.column_stack([
        np.full(T, 60.0), np.full(T, 10.0),
    ])
    dispatch[1::3, 1] = 100.0  # gB at p_nom=100 → maxed

    obs = pd.DataFrame({
        "price_A": 20.0 + rng.normal(0, 1.0, T),
        "price_B": 50.0 + rng.normal(0, 1.0, T),
        "flow_A-B": flows[:, 0],
        "dispatch_gA": dispatch[:, 0],
        "dispatch_gB": dispatch[:, 1],
    }, index=idx)

    result = calibrate(
        network=two_bus_network,
        observations=obs,
        formulation="noisy",
        solver="highs",
        active_set_tol=0.1,  # 0.1 MW slack for "binding"
        lambda_reg=0.01,
        obs_sigma=1.0,
    )

    # Multiple unique patterns ⇒ ASTB had something to do.
    assert result.n_active_sets >= 2
    # Calibration produced cost recoveries for both generators.
    assert "gen:gA:marginal_cost" in result.theta_hat
    assert "gen:gB:marginal_cost" in result.theta_hat
    assert all(v > 0 for v in result.theta_hat.values())
