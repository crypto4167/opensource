import itertools
import math
import statistics

import numpy as np
import pytest

from backtest_skeptic import stats as st
from backtest_skeptic.stats import (
    adjust_pvalues,
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    haircut_sharpe,
    lo_annualized_sharpe,
    lo_scale_factor,
    min_backtest_length,
    min_track_record_length,
    moments,
    newey_west,
    pbo,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_standard_error,
)
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars


def ar1(n, phi, mu, seed):
    """Stationary AR(1) with unit innovations, shifted by ``mu``."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1.0 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + mu


def naive_t(x):
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


# ---------------------------------------------------------------------------
# Sharpe ratio, moments, conversions
# ---------------------------------------------------------------------------


def test_sharpe_ratio_hand_computed():
    x = [0.01, -0.02, 0.03, 0.00, 0.015]
    expected = statistics.mean(x) / statistics.stdev(x)
    assert sharpe_ratio(x) == pytest.approx(expected, rel=1e-12)
    assert sharpe_ratio(x, periods_per_year=252) == pytest.approx(expected * math.sqrt(252))


def test_sharpe_ratio_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 2"):
        sharpe_ratio([0.1])
    with pytest.raises(ValueError, match="zero dispersion"):
        sharpe_ratio([0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="finite"):
        sharpe_ratio([0.1, math.nan, 0.2])
    with pytest.raises(ValueError, match="one-dimensional"):
        sharpe_ratio([[0.1, 0.2], [0.3, 0.4]])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_scale_free_statistics_survive_extreme_magnitudes(scale):
    # Squaring 1e200 overflows and squaring 1e-200 underflows, which used to
    # return a Sharpe ratio of 0.0 (huge) or raise "zero dispersion" (tiny).
    rng = np.random.default_rng(7)
    x = rng.normal(0.1, 1.0, 300)
    sx = x * scale
    assert sharpe_ratio(sx) == pytest.approx(sharpe_ratio(x), rel=1e-12)
    np.testing.assert_allclose(moments(sx), moments(x), rtol=1e-12)
    nw, nw_s = newey_west(x), newey_west(sx)
    assert nw_s.t == pytest.approx(nw.t, rel=1e-12)
    assert nw_s.se == pytest.approx(nw.se * scale, rel=1e-12)
    assert nw_s.mean == pytest.approx(nw.mean * scale, rel=1e-12)
    lo, lo_s = lo_annualized_sharpe(x, 4), lo_annualized_sharpe(sx, 4)
    assert lo_s.annualized == pytest.approx(lo.annualized, rel=1e-12)
    m = rng.normal(size=(200, 6))
    m[:, 0] += 0.3
    assert pbo(m * scale, n_splits=4).pbo == pbo(m, n_splits=4).pbo
    with pytest.raises(ValueError, match="zero dispersion"):
        sharpe_ratio([scale] * 5)


def test_annualisation_round_trip():
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))
    assert deannualize_sharpe(annualize_sharpe(0.07, 12), 12) == pytest.approx(0.07)
    with pytest.raises(ValueError):
        annualize_sharpe(0.1, 0)
    with pytest.raises(ValueError):
        deannualize_sharpe(1.0, -12)


def test_moments_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    d = x - 4.0
    m2, m3, m4 = (d**2).mean(), (d**3).mean(), (d**4).mean()
    skew, kurt = moments(x)
    assert skew == pytest.approx(m3 / m2**1.5)
    assert kurt == pytest.approx(m4 / m2**2)
    # symmetric two-point data: skew 0, raw kurtosis exactly 1
    assert moments([1.0, -1.0, 1.0, -1.0]) == pytest.approx((0.0, 1.0))


def test_moments_report_raw_kurtosis_on_normal_data():
    x = np.random.default_rng(0).standard_normal(200_000)
    skew, kurt = moments(x)
    assert abs(skew) < 0.02
    assert kurt == pytest.approx(3.0, abs=0.05)  # raw, not excess (which would be ~0)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe ratio and MinTRL (Bailey & Lopez de Prado, 2012)
# ---------------------------------------------------------------------------


def test_psr_reproduces_2012_paper_hedge_fund_example():
    # Figure 6 and section 3: 24 monthly returns, annual SR 1.585, so the monthly
    # SR is 1.585 / sqrt(12) = 0.4575 (shown rounded as 0.458 in the paper).
    sr = 1.585 / math.sqrt(12)
    assert round(probabilistic_sharpe_ratio(sr=sr, n=24, skew=0, kurtosis=3), 3) == 0.982
    hedge = {"sr": sr, "skew": -2.448, "kurtosis": 10.164}
    assert round(probabilistic_sharpe_ratio(n=24, **hedge), 3) == 0.913
    # "should we have 3 years instead of 2"
    assert round(probabilistic_sharpe_ratio(n=36, **hedge), 3) == 0.953


def test_psr_hand_computed_formula():
    sr, n, g3, g4, b = 0.1, 500, -0.5, 6.0, 0.02
    z = (sr - b) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    expected = statistics.NormalDist().cdf(z)
    got = probabilistic_sharpe_ratio(sr=sr, n=n, skew=g3, kurtosis=g4, benchmark=b)
    assert got == pytest.approx(expected, rel=1e-14)


def test_psr_from_returns_matches_explicit_moments():
    x = np.random.default_rng(1).standard_t(5, size=400) * 0.01 + 0.001
    g3, g4 = moments(x)
    explicit = probabilistic_sharpe_ratio(sr=sharpe_ratio(x), n=len(x), skew=g3, kurtosis=g4)
    assert probabilistic_sharpe_ratio(x) == pytest.approx(explicit, rel=1e-14)


def test_psr_input_forms_cannot_be_mixed_or_left_incomplete():
    x = np.random.default_rng(2).normal(0, 1, 50)
    with pytest.raises(ValueError, match="not both"):
        probabilistic_sharpe_ratio(x, sr=0.1)
    with pytest.raises(ValueError, match="missing"):
        probabilistic_sharpe_ratio(sr=0.1, n=100)
    with pytest.raises(ValueError, match="> 1"):
        probabilistic_sharpe_ratio(sr=0.1, n=1, skew=0, kurtosis=3)


def test_excess_kurtosis_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="RAW kurtosis"):
        probabilistic_sharpe_ratio(sr=0.1, n=100, skew=0.0, kurtosis=0.0)
    with pytest.raises(ValueError, match="RAW kurtosis"):
        sharpe_standard_error(0.1, 100, skew=-2.0, kurtosis=4.0)  # needs >= 1 + 4
    with pytest.raises(ValueError, match="RAW kurtosis"):
        min_track_record_length(0.1, kurtosis=0.5)


def test_sharpe_standard_error_hand_computed():
    assert sharpe_standard_error(0.1, 101) == pytest.approx(math.sqrt(1.005) / 10)
    expected = math.sqrt((1 + 0.5 * 0.2 + (7 - 1) / 4 * 0.04) / 249)
    assert sharpe_standard_error(0.2, 250, skew=-0.5, kurtosis=7) == pytest.approx(expected)


def test_psr_negative_control_is_calibrated_on_null_returns():
    # 300 driftless samples: PSR > 0.95 should happen about 5% of the time.
    hits = sum(
        probabilistic_sharpe_ratio(np.random.default_rng(1000 + s).normal(0, 1, 250)) > 0.95
        for s in range(300)
    )
    assert 0.02 <= hits / 300 <= 0.09


def test_psr_negative_control_on_random_walk_bars():
    bars = random_walk_bars(5000, seed=11)
    r = np.diff(np.log(bars.close))
    assert probabilistic_sharpe_ratio(r) < 0.95


def test_psr_positive_control_on_planted_signal():
    bars, signal = planted_signal_bars(20000, seed=3)
    r = np.diff(np.log(bars.close))
    fired = signal[:-1] != 0
    strategy = (signal[:-1] * r)[fired]
    assert probabilistic_sharpe_ratio(strategy) > 0.999
    assert probabilistic_sharpe_ratio(strategy, benchmark=0.1) > 0.95


@pytest.mark.parametrize(
    ("ppy", "observed", "true", "years"),
    [
        # Figure 8 (daily, IID normal)
        (252, 2.0, 1.0, 2.73),
        (252, 0.5, 0.0, 10.83),
        (252, 5.0, 0.0, 0.12),
        (252, 5.0, 4.5, 11.36),
        # Figure 9 (weekly)
        (52, 2.0, 1.0, 2.83),
        (52, 5.0, 4.5, 13.44),
        # Figure 10 (monthly)
        (12, 2.0, 1.0, 3.24),
        (12, 4.5, 4.0, 20.04),
    ],
)
def test_min_track_record_reproduces_2012_paper_normal_tables(ppy, observed, true, years):
    obs = min_track_record_length(observed / math.sqrt(ppy), true / math.sqrt(ppy))
    assert obs / ppy == pytest.approx(years, abs=0.006)


def test_min_track_record_reproduces_2012_paper_non_normal_example():
    # Figure 11 / appendix A.3: monthly, skew -0.72, kurtosis 5.78, SR 2 vs 1 annual.
    sr, ref = 2 / math.sqrt(12), 1 / math.sqrt(12)
    n = min_track_record_length(sr, ref, skew=-0.72, kurtosis=5.78)
    assert n == pytest.approx(59.895, abs=5e-4)
    assert n / 12 == pytest.approx(4.99, abs=0.005)
    psr = probabilistic_sharpe_ratio(sr=sr, n=59.895, skew=-0.72, kurtosis=5.78, benchmark=ref)
    assert psr == pytest.approx(0.95, abs=1e-6)


def test_min_track_record_is_infinite_without_an_edge_and_inverts_psr():
    assert min_track_record_length(0.05, benchmark=0.05) == math.inf
    assert min_track_record_length(-0.1) == math.inf
    rng = np.random.default_rng(4)
    for _ in range(20):
        sr = rng.uniform(0.05, 0.5)
        g3 = rng.uniform(-1.5, 0.5)
        g4 = 1 + g3**2 + rng.uniform(0.5, 8)
        prob = rng.uniform(0.6, 0.99)
        n = min_track_record_length(sr, 0.01, skew=g3, kurtosis=g4, prob=prob)
        back = probabilistic_sharpe_ratio(sr=sr, n=n, skew=g3, kurtosis=g4, benchmark=0.01)
        assert back == pytest.approx(prob, abs=1e-9)
    with pytest.raises(ValueError):
        min_track_record_length(0.1, prob=1.0)


# ---------------------------------------------------------------------------
# Expected maximum, deflated Sharpe ratio, MinBTL
# ---------------------------------------------------------------------------


def test_dsr_reproduces_2014_paper_example():
    # Bailey & Lopez de Prado (2014): annual SR 2.5 over T=1250 daily returns
    # (250 a year), annual trial variance 0.5, skew -3, raw kurtosis 10.
    # Converted to per-period units: SR = 2.5/sqrt(250), V = 0.5/250.
    sr, v = 2.5 / math.sqrt(250), 0.5 / 250
    assert expected_max_sharpe(100, v) == pytest.approx(0.1132, abs=5e-5)
    common = {"sr": sr, "n": 1250, "skew": -3, "kurtosis": 10, "var_trials_sharpe": v}
    assert deflated_sharpe_ratio(n_trials=100, **common) == pytest.approx(0.9004, abs=5e-5)
    assert deflated_sharpe_ratio(n_trials=46, **common) == pytest.approx(0.9505, abs=5e-5)
    normal = {**common, "skew": 0, "kurtosis": 3}
    assert deflated_sharpe_ratio(n_trials=88, **normal) == pytest.approx(0.9505, abs=5e-5)


def test_dsr_falls_as_the_number_of_trials_rises():
    common = {"sr": 0.12, "n": 1000, "skew": -0.5, "kurtosis": 5, "var_trials_sharpe": 0.002}
    values = [deflated_sharpe_ratio(n_trials=k, **common) for k in (1, 2, 5, 10, 50, 100, 1000)]
    assert all(a > b for a, b in itertools.pairwise(values))


def test_dsr_with_one_trial_is_psr_against_zero():
    x = np.random.default_rng(5).normal(0.05, 1, 300)
    assert deflated_sharpe_ratio(x, n_trials=1, var_trials_sharpe=0.01) == pytest.approx(
        probabilistic_sharpe_ratio(x)
    )


def test_dsr_trial_inputs_are_required_no_silent_psr_fallback():
    x = np.random.default_rng(6).normal(0.05, 1, 300)
    with pytest.raises(TypeError):
        deflated_sharpe_ratio(x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        deflated_sharpe_ratio(x, n_trials=10)  # type: ignore[call-arg]


def test_dsr_negative_control_best_of_noise_is_not_significant():
    r = np.random.default_rng(7).normal(0, 0.01, (1000, 50))
    srs = r.mean(axis=0) / r.std(axis=0, ddof=1)
    best = int(np.argmax(srs))
    # The selected winner looks significant on its own...
    assert probabilistic_sharpe_ratio(r[:, best]) > 0.95
    # ...but not once the 50 trials are accounted for.
    dsr = deflated_sharpe_ratio(
        r[:, best], n_trials=50, var_trials_sharpe=float(np.var(srs, ddof=1))
    )
    assert dsr < 0.9


def test_dsr_positive_control_real_edge_survives_deflation():
    r = np.random.default_rng(7).normal(0, 0.01, (1000, 50))
    r[:, 3] += 0.0015  # Sharpe ~0.15 per period
    srs = r.mean(axis=0) / r.std(axis=0, ddof=1)
    best = int(np.argmax(srs))
    assert best == 3
    dsr = deflated_sharpe_ratio(
        r[:, best], n_trials=50, var_trials_sharpe=float(np.var(srs, ddof=1))
    )
    assert dsr > 0.95


def test_expected_max_sharpe_edge_cases_and_monte_carlo():
    assert expected_max_sharpe(1, 0.3) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(0.5, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.1)
    sims = np.random.default_rng(0).standard_normal((4000, 100)).max(axis=1).mean()
    assert expected_max_sharpe(100, 1.0) == pytest.approx(sims, rel=0.02)
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


def test_expected_max_sharpe_for_very_many_trials():
    # 1 - 1/N rounds to 1.0 in double precision from N ~ 1e16 on, and loses
    # digits well before that; the upper quantile must come from 1/N directly.
    g, Z = 0.57721566490153286, statistics.NormalDist()
    for n in (1e12, 1e16, 1e20, 1e300):
        exact = (1 - g) * -Z.inv_cdf(1 / n) + g * -Z.inv_cdf(1 / (n * math.e))
        assert expected_max_sharpe(n, 1.0) == pytest.approx(exact, rel=1e-12)
        assert min_backtest_length(n, 1.0) == pytest.approx(exact**2, rel=1e-12)
    assert expected_max_sharpe(1e16, 1.0) < expected_max_sharpe(1e20, 1.0)


def test_min_backtest_length_reproduces_notices_paper():
    # Bailey et al. (2014): with 5 years of data, no more than 45 configurations
    # before an annual in-sample Sharpe of 1 is expected from noise.
    assert min_backtest_length(45, 1.0) == pytest.approx(5.0, abs=0.01)
    assert min_backtest_length(46, 1.0) > 5.0
    # "After trying only 7 ... the expected maximum SR IS is 1 for a 2-year long backtest"
    assert min_backtest_length(7, 1.0) == pytest.approx(2.0, abs=0.1)
    # N = 10 gives an expected maximum of 1.57 after one year
    assert expected_max_sharpe(10, 1.0) == pytest.approx(1.57, abs=0.005)
    for n in (2, 5, 10, 100, 10_000):
        assert min_backtest_length(n, 1.0) < 2 * math.log(n)  # the paper's upper bound
    assert min_backtest_length(100, 2.0) == pytest.approx(min_backtest_length(100, 1.0) / 4)
    with pytest.raises(ValueError):
        min_backtest_length(10, 0.0)


# ---------------------------------------------------------------------------
# PBO (CSCV)
# ---------------------------------------------------------------------------


def reference_pbo(m, n_splits, metric):
    """Straightforward loop implementation of CSCV, used to cross-check pbo()."""
    t, n = m.shape
    base, extra = divmod(t, n_splits)
    sizes = [base + (1 if b < extra else 0) for b in range(n_splits)]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [np.arange(starts[b], starts[b + 1]) for b in range(n_splits)]
    logits = []
    for c in itertools.combinations(range(n_splits), n_splits // 2):
        rest = [b for b in range(n_splits) if b not in c]
        ins = metric(m[np.concatenate([blocks[b] for b in c])])
        oos = metric(m[np.concatenate([blocks[b] for b in rest])])
        best = max(range(n), key=lambda j: (ins[j], -j))
        w = oos[best]
        rank = sum(v < w for v in oos) + (sum(v == w for v in oos) + 1) / 2
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return float(np.mean(logits < 0) + 0.5 * np.mean(logits == 0)), logits


def column_sharpe(block):
    return block.mean(axis=0) / block.std(axis=0, ddof=1)


def test_pbo_hand_computed_counts_the_exact_median_as_half():
    # Two blocks (S=2), three configurations, metric = mean.
    # Block A means (3, 1, 2), block B means (2, 4, 1).
    # IS=A: winner col 0, OOS rank 2 of 3 -> w = 0.5, logit 0 (exact median).
    # IS=B: winner col 1, OOS rank 1 of 3 -> w = 0.25, logit ln(1/3).
    m = np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0], [2.0, 4.0, 1.0], [2.0, 4.0, 1.0]])
    res = pbo(m, n_splits=2, metric=lambda b: b.mean(axis=0))
    assert res.n_combinations == 2
    assert res.logits.tolist() == pytest.approx([0.0, math.log(1 / 3)])
    # The median split counts half: (0.5 + 1) / 2. Ignoring it would give 0.5.
    assert res.pbo == pytest.approx(0.75)
    assert res.is_metric.tolist() == [3.0, 4.0]
    assert res.oos_metric.tolist() == [2.0, 1.0]
    assert res.degradation_slope == pytest.approx(-1.0)
    assert res.degradation_intercept == pytest.approx(5.0)
    assert res.prob_oos_loss == 0.0


def test_pbo_identical_configurations_use_average_ranks():
    # Columns 0 and 1 are identical and better than column 2 in every block.
    # Average rank of the winner is 2.5 of 3 -> w = 0.625 every split; PBO = 0.
    # Counting the tie as "worse" would give w = 0.5 and make it look overfit.
    rng = np.random.default_rng(8)
    good = rng.normal(0.5, 1.0, 400)
    bad = rng.normal(-0.5, 1.0, 400)
    m = np.column_stack([good, good, bad])
    res = pbo(m, n_splits=8)
    assert res.pbo == 0.0
    assert np.allclose(res.logits, math.log(0.625 / 0.375))


def test_pbo_matches_loop_reference_with_ties_and_uneven_blocks():
    rng = np.random.default_rng(9)
    m = rng.normal(0.0, 1.0, (203, 7))
    m[:, 4] = m[:, 1]  # an exact duplicate configuration
    m[:, 0] += 0.05
    ref_pbo, ref_logits = reference_pbo(m, 6, column_sharpe)
    res = pbo(m, n_splits=6)
    assert res.n_combinations == math.comb(6, 3)
    assert res.pbo == pytest.approx(ref_pbo)
    assert res.logits == pytest.approx(ref_logits, rel=1e-9, abs=1e-12)
    custom = pbo(m, n_splits=6, metric=column_sharpe)
    assert custom.pbo == pytest.approx(ref_pbo)
    assert custom.logits == pytest.approx(ref_logits, rel=1e-9, abs=1e-12)


def test_pbo_negative_control_pure_noise_is_near_half():
    values = [
        pbo(np.random.default_rng(s).normal(0, 1, (600, 20)), n_splits=10).pbo for s in range(20)
    ]
    assert 0.4 <= float(np.mean(values)) <= 0.6


def test_pbo_positive_control_one_superior_configuration():
    m = np.random.default_rng(10).normal(0, 1, (1000, 20))
    m[:, 7] += 0.3
    res = pbo(m, n_splits=10)
    assert res.pbo < 0.01
    assert res.prob_oos_loss < 0.01
    assert np.median(res.logits) > 2.0


def test_pbo_positive_control_selection_that_reverses_out_of_sample():
    # Each configuration is +0.5 on half the blocks and -0.5 on the other half,
    # so the in-sample winner is the configuration with the worst remaining blocks.
    rng = np.random.default_rng(11)
    s, rows = 8, 50
    cols = []
    for _ in range(20):
        signs = rng.permutation([1.0] * (s // 2) + [-1.0] * (s // 2))
        cols.append(np.repeat(0.5 * signs, rows) + rng.normal(0, 1, s * rows))
    res = pbo(np.column_stack(cols), n_splits=s)
    assert res.pbo > 0.9
    assert res.prob_oos_loss > 0.9
    assert res.degradation_slope is not None and res.degradation_slope < 0


def test_pbo_default_sixteen_splits_and_result_formatting():
    m = np.random.default_rng(12).normal(0, 1, (320, 6))
    res = pbo(m)
    assert res.n_splits == 16 and res.n_combinations == 12870 and len(res.logits) == 12870
    assert res.n_configurations == 6 and res.n_periods == 320
    d = res.as_dict()
    assert d["pbo"] == res.pbo and len(d["logits"]) == 12870
    assert "PBO" in str(res) and "12870" in str(res)
    with pytest.raises(ValueError):
        res.logits[0] = 1.0  # read-only


def test_pbo_constant_columns_score_zero_sharpe():
    rng = np.random.default_rng(13)
    m = np.column_stack([np.full(200, 0.1), rng.normal(0, 1, 200), rng.normal(0, 1, 200)])
    res = pbo(m, n_splits=4)
    assert np.isfinite(res.logits).all()


def test_pbo_rejects_bad_input():
    m = np.random.default_rng(14).normal(0, 1, (100, 4))
    with pytest.raises(ValueError, match="at least 2 configurations"):
        pbo(m[:, :1])
    with pytest.raises(ValueError, match="even"):
        pbo(m, n_splits=5)
    with pytest.raises(ValueError, match="too few"):
        pbo(m, n_splits=52)
    with pytest.raises(ValueError, match="identical"):
        pbo(np.column_stack([m[:, 0]] * 3), n_splits=4)
    with pytest.raises(ValueError, match="2-D"):
        pbo(m[:, 0])
    bad = m.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pbo(bad, n_splits=4)
    with pytest.raises(ValueError, match="one per column"):
        pbo(m, n_splits=4, metric=lambda b: b.mean())


# ---------------------------------------------------------------------------
# Newey-West / Hansen-Hodrick
# ---------------------------------------------------------------------------


def test_newey_west_hand_computed():
    # x = 1..5: mean 3, gamma0 = 10/5 = 2, gamma1 = (2 + 0 + 0 + 2)/5 = 0.8.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = newey_west(x, lags=1)
    assert res.mean == 3.0
    assert res.se == pytest.approx(math.sqrt((2 + 2 * 0.5 * 0.8) / 5))
    assert res.t == pytest.approx(3.0 / math.sqrt(0.56))
    hh = newey_west(x, lags=1, kernel="uniform")
    assert hh.se == pytest.approx(math.sqrt((2 + 2 * 0.8) / 5))
    zero = newey_west(x, lags=0)
    assert zero.se == pytest.approx(math.sqrt(2 / 5))
    mean, se, t = res
    assert (mean, se, t) == (res.mean, res.se, res.t)


def test_newey_west_default_lag_rule():
    x = np.random.default_rng(15).normal(0, 1, 500)
    assert newey_west(x).lags == math.floor(4 * (500 / 100) ** (2 / 9)) == 5


def test_newey_west_positive_control_deflates_t_on_ar1():
    x = ar1(5000, 0.6, 0.1, seed=1)
    res = newey_west(x)
    assert abs(res.t) < 0.75 * abs(naive_t(x))


def test_newey_west_negative_control_is_calibrated_on_driftless_ar1():
    naive_hits = nw_hits = 0
    for s in range(300):
        x = ar1(400, 0.5, 0.0, seed=s)
        naive_hits += abs(naive_t(x)) > 1.96
        nw_hits += abs(newey_west(x).t) > 1.96
    assert naive_hits / 300 > 0.18  # the naive test is badly over-sized
    assert nw_hits / 300 < 0.11


def test_uniform_kernel_handles_overlapping_returns():
    # Overlapping 5-period sums of IID N(0,1): the long-run sd is exactly 5.
    e = np.random.default_rng(5).normal(0, 1, 20004)
    x = np.convolve(e, np.ones(5), "valid")[:20000]
    hh = newey_west(x, lags=4, kernel="uniform")
    bart = newey_west(x, lags=4)
    assert not hh.fallback and hh.kernel == "uniform"
    assert hh.se * math.sqrt(len(x)) == pytest.approx(5.0, rel=0.05)
    assert bart.se < 0.9 * hh.se  # Bartlett with h-1 lags under-corrects


def test_uniform_kernel_falls_back_to_bartlett_when_variance_not_positive():
    x = np.tile([1.0, -1.0], 50) + np.random.default_rng(16).normal(0, 0.01, 100)
    res = newey_west(x, lags=1, kernel="uniform")
    assert res.fallback and res.kernel == "bartlett" and res.requested_kernel == "uniform"
    assert res.se == pytest.approx(newey_west(x, lags=1).se)
    assert "fell back" in str(res)
    assert res.as_dict()["fallback"] is True


def test_newey_west_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        newey_west([1.0])
    with pytest.raises(ValueError, match="zero dispersion"):
        newey_west([2.0] * 10)
    with pytest.raises(ValueError, match="lags"):
        newey_west([1.0, 2.0, 3.0], lags=3)
    with pytest.raises(ValueError, match="lags"):
        newey_west([1.0, 2.0, 3.0], lags=-1)
    with pytest.raises(ValueError, match="kernel"):
        newey_west([1.0, 2.0, 3.0], kernel="parzen")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lo (2002)
# ---------------------------------------------------------------------------


def test_lo_scale_factor_hand_computed():
    assert lo_scale_factor([0.0] * 11, 12) == pytest.approx(math.sqrt(12))
    assert lo_scale_factor([], 1) == 1.0
    phi = 0.3
    rho = [phi**k for k in range(1, 12)]
    denom = 12 + 2 * sum((12 - k) * phi**k for k in range(1, 12))
    assert lo_scale_factor(rho, 12) == pytest.approx(12 / math.sqrt(denom))
    with pytest.raises(ValueError, match="not positive"):
        lo_scale_factor([-0.9] * 11, 12)
    with pytest.raises(ValueError, match="need 11"):
        lo_scale_factor([0.1] * 5, 12)


def test_lo_positive_control_autocorrelated_returns_are_scaled_down():
    res = lo_annualized_sharpe(ar1(6000, 0.5, 0.1, seed=2), 12)
    assert res.annualized < 0.75 * res.naive_annualized
    assert res.autocorrelations[0] == pytest.approx(0.5, abs=0.05)
    assert "naive" in str(res) and res.as_dict()["periods"] == 12


def test_lo_negative_control_iid_returns_scale_by_sqrt_q():
    x = np.random.default_rng(3).normal(0.1, 1, 6000)
    res = lo_annualized_sharpe(x, 12)
    assert res.eta == pytest.approx(math.sqrt(12), rel=0.1)
    assert res.sharpe == pytest.approx(sharpe_ratio(x))
    assert res.naive_annualized == pytest.approx(sharpe_ratio(x, periods_per_year=12))


def test_lo_rejects_bad_input():
    with pytest.raises(ValueError, match="periods"):
        lo_annualized_sharpe([0.1, 0.2, 0.3], 0)
    with pytest.raises(ValueError, match="observations"):
        lo_annualized_sharpe([0.1, 0.2, 0.3], 12)


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def test_adjust_pvalues_hand_computed_sorted():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert adjust_pvalues(p, "bonferroni") == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.25])
    assert adjust_pvalues(p, "holm") == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.05] * 5)
    c5 = 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5
    assert adjust_pvalues(p, "by") == pytest.approx([0.05 * c5] * 5)


def test_adjust_pvalues_hand_computed_unsorted():
    p = [0.04, 0.001, 0.03, 0.2]
    assert adjust_pvalues(p, "bonferroni") == pytest.approx([0.16, 0.004, 0.12, 0.8])
    assert adjust_pvalues(p, "holm") == pytest.approx([0.09, 0.004, 0.09, 0.2])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.16 / 3, 0.004, 0.16 / 3, 0.2])
    c4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert adjust_pvalues(p, "by") == pytest.approx(
        [0.16 / 3 * c4, 0.004 * c4, 0.16 / 3 * c4, min(1.0, 0.2 * c4)]
    )
    assert adjust_pvalues([0.6, 0.9], "bonferroni").tolist() == [1.0, 1.0]


def test_adjust_pvalues_properties():
    p = np.random.default_rng(17).uniform(0, 0.2, 40)
    order = np.argsort(p)
    for method in ("holm", "bonferroni", "bh", "by"):
        adj = adjust_pvalues(p, method)
        assert (adj >= p - 1e-15).all() and (adj <= 1).all()
        assert (np.diff(adj[order]) >= -1e-15).all()  # monotone in p
    assert (adjust_pvalues(p, "holm") <= adjust_pvalues(p, "bonferroni") + 1e-15).all()
    assert (adjust_pvalues(p, "bh") <= adjust_pvalues(p, "by") + 1e-15).all()
    assert adjust_pvalues(p.reshape(5, 8)).shape == (5, 8)
    assert adjust_pvalues([]).shape == (0,)


def test_adjust_pvalues_controls_null_and_finds_real_effects():
    rng = np.random.default_rng(18)
    null = rng.uniform(0, 1, 990)
    assert (adjust_pvalues(null, "holm") <= 0.05).sum() == 0
    assert (adjust_pvalues(null, "bh") <= 0.05).sum() == 0
    mixed = np.concatenate([np.full(10, 1e-6), null])
    assert (adjust_pvalues(mixed, "holm")[:10] <= 0.05).all()
    assert (adjust_pvalues(mixed, "by")[:10] <= 0.05).all()


def test_adjust_pvalues_rejects_bad_input():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, math.nan])
    with pytest.raises(ValueError, match="method"):
        adjust_pvalues([0.1], "fdr")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Harvey & Liu (2015) haircut
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "df", "t"),
    [
        (0.05, 10, 2.228138852),
        (0.01, 5, 4.032142984),
        (0.05, 1, 12.70620474),
        (0.05, 30, 2.042272456),
    ],
)
def test_student_t_quantiles_match_tables(p, df, t):
    assert st._two_sided_quantile(p, df) == pytest.approx(t, rel=1e-8)
    assert st._two_sided_p(t, df) == pytest.approx(p, rel=1e-7)


def test_haircut_reproduces_harvey_liu_text_example():
    # 20 years of monthly returns (T = 240), annual SR 0.75, N = 200 independent
    # tests: p_S = 0.0008, p_M = 0.15, haircut Sharpe 0.32 (paper, eq. 4-5).
    h = haircut_sharpe(0.75, 240, 200, periods_per_year=12, method="sidak", distribution="normal")
    assert h.t_stat == pytest.approx(0.75 * math.sqrt(20))
    assert h.p_single == pytest.approx(0.0008, abs=5e-5)
    assert h.p_multiple == pytest.approx(0.15, abs=0.005)
    assert h.haircut_sharpe == pytest.approx(0.32, abs=0.005)
    assert h.haircut == pytest.approx((0.75 - 0.32) / 0.75, abs=0.01)


@pytest.mark.parametrize(
    ("t_stat", "p_single", "haircuts"),
    [
        (2.99, 2.88e-3, (0.266, 0.500, 0.616)),  # E/P
        (4.70, 3.20e-6, (0.109, 0.192, 0.230)),  # MOM
    ],
)
def test_haircut_reproduces_harvey_liu_exhibit_1(t_stat, p_single, haircuts):
    # Exhibit 1 (N = 10, 50, 100). July 1963 - December 2012 is 594 months.
    # The column of p^M values equals 1 - (1 - p^S)^N with Student-t p-values.
    sr = t_stat / math.sqrt(594 / 12)
    for n_trials, expected in zip((10, 50, 100), haircuts, strict=True):
        h = haircut_sharpe(sr, 594, n_trials, periods_per_year=12, method="sidak")
        assert h.t_stat == pytest.approx(t_stat)
        assert h.p_single == pytest.approx(p_single, rel=0.02)
        assert h.haircut == pytest.approx(expected, abs=0.0015)


def test_haircut_monotonicity_and_edge_cases():
    single = haircut_sharpe(1.0, 120, 1, periods_per_year=12)
    assert single.haircut == pytest.approx(0.0, abs=1e-9)
    cuts = [haircut_sharpe(1.0, 120, n, periods_per_year=12).haircut for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in itertools.pairwise(cuts))
    bonf = haircut_sharpe(0.6, 240, 50, periods_per_year=12, method="bonferroni")
    sidak = haircut_sharpe(0.6, 240, 50, periods_per_year=12, method="sidak")
    assert sidak.haircut <= bonf.haircut
    neg = haircut_sharpe(-0.6, 240, 50, periods_per_year=12, method="bonferroni")
    assert neg.haircut_sharpe == pytest.approx(-bonf.haircut_sharpe)
    assert neg.haircut == pytest.approx(bonf.haircut)
    weak = haircut_sharpe(0.1, 60, 100, periods_per_year=12)
    assert weak.p_multiple == 1.0 and weak.haircut_sharpe == 0.0 and weak.haircut == 1.0
    # Student t is more conservative than the normal approximation in short samples
    t_cut = haircut_sharpe(1.5, 36, 5, periods_per_year=12, distribution="t").haircut
    n_cut = haircut_sharpe(1.5, 36, 5, periods_per_year=12, distribution="normal").haircut
    assert t_cut > n_cut
    assert "haircut" in str(bonf) and bonf.as_dict()["n_trials"] == 50


def test_haircut_negative_control_best_of_noise_is_not_significant():
    r = np.random.default_rng(19).normal(0, 0.01, (1260, 500))
    srs = r.mean(axis=0) / r.std(axis=0, ddof=1)
    best = annualize_sharpe(float(srs.max()), 252)
    h = haircut_sharpe(best, 1260, 500, periods_per_year=252)
    assert h.p_single < 0.01  # the best of 500 looks significant alone
    assert h.p_multiple > 0.05
    assert h.haircut > 0.5


def test_haircut_positive_control_real_edge_survives():
    bars, signal = planted_signal_bars(20000, seed=3)
    r = np.diff(np.log(bars.close))
    strategy = (signal[:-1] * r)[signal[:-1] != 0]
    sr_annual = sharpe_ratio(strategy, periods_per_year=252)
    h = haircut_sharpe(sr_annual, len(strategy), 1000, periods_per_year=252)
    assert h.p_multiple < 0.01
    assert h.haircut < 0.3


def test_haircut_rejects_bad_input():
    with pytest.raises(ValueError, match="zero Sharpe"):
        haircut_sharpe(0.0, 100, 10, periods_per_year=12)
    with pytest.raises(ValueError, match="n_obs"):
        haircut_sharpe(1.0, 1, 10, periods_per_year=12)
    with pytest.raises(ValueError, match="n_trials"):
        haircut_sharpe(1.0, 100, 0, periods_per_year=12)
    with pytest.raises(ValueError, match="integer"):
        haircut_sharpe(1.0, 100.5, 10, periods_per_year=12)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        haircut_sharpe(1.0, 100, 10, periods_per_year=12, method="holm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="periods_per_year"):
        haircut_sharpe(1.0, 100, 10, periods_per_year=0)
    with pytest.raises(ValueError, match="underflows"):
        haircut_sharpe(50.0, 252 * 40, 10, periods_per_year=252, distribution="normal")
