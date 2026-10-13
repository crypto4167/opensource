import math
import time

import numpy as np
import pytest

from backtest_skeptic import Bars, Costs, Setup, Trade, run, run_signals
from backtest_skeptic.engine import _resolve
from backtest_skeptic.random_entry import (
    NullTestResult,
    _Config,
    _resolve_batch,
    positive_control,
    random_entry_test,
    side_test,
)
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars

COSTS = Costs(half_spread=0.005, fee_rate=0.0001)


def bracket_trades(bars, side, stop_frac, target_frac, costs=None, **kwargs):
    """Run a fixed-fraction bracket on a side array (+1/-1/0 per bar)."""
    side = np.asarray(side, dtype=np.float64)
    close = bars.close
    stop = close * (1 - side * stop_frac)
    target = close * (1 + side * target_frac)
    return run_signals(bars, side, stop, target, costs=costs, **kwargs)


def random_sides(n, prob, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < prob, rng.choice([-1, 1], n), 0)


def ramp_bars(n=120, growth=1.01):
    """Deterministic bars: every bar opens at the previous close, which rises
    1% per bar (so prices triple over 110 bars), and trades from 0.5% below
    to 3% above that open. A long with a 1% stop and 2% target therefore wins
    exactly 2R on its entry bar from any signal bar; a short with the mirrored
    bracket loses exactly 1R."""
    close = 100.0 * growth ** np.arange(n)
    prev = np.concatenate([[100.0], close[:-1]])
    return Bars(prev, 1.03 * prev, 0.995 * prev, close)


def ramp_strategy(bars, longs, shorts):
    side = np.zeros(len(bars))
    side[longs] = 1
    side[shorts] = -1
    return bracket_trades(bars, side, 0.01, 0.02)


# ---------------------------------------------------------------- resolver


@pytest.mark.parametrize(
    "cfg",
    [
        _Config(Costs(), None, "stop", 0),
        _Config(Costs(half_spread=0.02, fee_rate=0.0005, slippage=0.01), 7, "target", 0),
        _Config(Costs(half_spread=0.05, fee_rate=0.001), 3, "stop", 2),
    ],
)
def test_vectorised_resolver_is_bit_identical_to_engine(cfg):
    bars = random_walk_bars(1500, seed=3, gap_vol=0.004)
    rng = np.random.default_rng(1)
    k = 3000
    sig = rng.integers(0, len(bars) - 2 - cfg.entry_delay, k)
    side = rng.choice([-1.0, 1.0], k)
    c = bars.close[sig]
    stop = c * (1 - side * rng.uniform(0.0005, 0.03, k))
    target = c * (1 + side * rng.uniform(0.0005, 0.05, k))
    r, exit_index = _resolve_batch(bars, sig, side, stop, target, cfg)
    skipped = 0
    for i in range(k):
        setup = Setup(int(side[i]), float(stop[i]), float(target[i]))
        out = _resolve(
            bars, int(sig[i]), setup, cfg.costs, cfg.max_bars, cfg.entry_delay, cfg.same_bar, None
        )
        if isinstance(out, Trade):
            assert r[i] == out.r
            assert exit_index[i] == out.exit_index
        else:
            skipped += 1
            assert math.isnan(r[i])
            assert exit_index[i] == -1
    assert skipped > 0  # the gaps produced unfillable setups, so that path was exercised


# ---------------------------------------------------------- hand-computed


def test_matched_sides_on_deterministic_bars_give_a_constant_null():
    bars = ramp_bars()
    res = ramp_strategy(bars, longs=[5, 15, 25, 35, 45, 55], shorts=[65, 75, 85, 95])
    assert [round(t.r, 9) for t in res.trades] == [2.0] * 6 + [-1.0] * 4
    out = random_entry_test(bars, res, n_sims=200, seed=1)
    # Every simulation has exactly 6 longs (+2R) and 4 shorts (-1R): (12 - 4) / 10.
    assert out.observed == pytest.approx(0.8)
    np.testing.assert_allclose(out.null, 0.8, atol=1e-12)
    # Ties count against the strategy, and a null with no spread has no z-score.
    assert out.p_value == 1.0
    assert out.percentile == pytest.approx(50.0)
    assert out.z_score is None
    win = random_entry_test(bars, res, n_sims=50, seed=1, metric="win_rate")
    np.testing.assert_allclose(win.null, 0.6)


def test_matched_sides_keep_each_side_with_its_own_bracket():
    # Longs use a 1% stop and 2% target (+2R on the ramp); shorts use a 4% stop
    # and 0.4% target (+0.1R). Matching the side mix but pairing sides with
    # brackets at random gives a long a short's bracket (+0.1R) or a short a
    # long's (-1R), so the null would no longer describe this strategy.
    bars = ramp_bars()
    side = np.zeros(len(bars))
    side[[5, 15, 25, 35, 45, 55]] = 1
    side[[65, 75, 85, 95]] = -1
    close = bars.close
    stop_frac = np.where(side > 0, 0.01, 0.04)
    target_frac = np.where(side > 0, 0.02, 0.004)
    stop = close * (1 - side * stop_frac)
    target = close * (1 + side * target_frac)
    res = run_signals(bars, side, stop, target)
    assert [round(t.r, 9) for t in res.trades] == [2.0] * 6 + [0.1] * 4
    out = random_entry_test(bars, res, n_sims=200, seed=1)
    assert out.observed == pytest.approx(1.24)
    np.testing.assert_allclose(out.null, 1.24, atol=1e-9)


def test_geometry_is_copied_as_fractions_of_the_signal_close():
    # Prices triple over the series. As fractions (1% stop, 2% target) every
    # random long wins 2R; absolute distances copied from the first trade
    # (about 1.05 and 2.1 price units) would be stopped out on the same bar
    # at the higher prices.
    bars = ramp_bars()
    res = ramp_strategy(bars, longs=[5, 6, 7, 8, 9, 10], shorts=[])
    out = random_entry_test(bars, res, n_sims=100, seed=2, sides="long")
    np.testing.assert_allclose(out.null, 2.0, atol=1e-12)
    short = random_entry_test(bars, res, n_sims=100, seed=2, sides="short")
    np.testing.assert_allclose(short.null, -1.0, atol=1e-12)
    assert short.p_value == pytest.approx(1 / 101)
    assert short.percentile == 100.0


def test_coin_sides_p_value_matches_binomial_tail():
    # With fair-coin sides each trade is +2R or -1R with probability 1/2, so
    # expectancy >= 0.8 needs at least 6 longs of 10: P = 386 / 1024.
    bars = ramp_bars()
    res = ramp_strategy(bars, longs=[5, 15, 25, 35, 45, 55], shorts=[65, 75, 85, 95])
    out = random_entry_test(bars, res, n_sims=4000, seed=3, sides="random")
    assert out.p_value == pytest.approx(386 / 1024, abs=0.025)
    assert out.null_mean == pytest.approx(0.5, abs=0.03)
    side = side_test(bars, res, n_sims=4000, seed=3)
    assert side.p_value == pytest.approx(386 / 1024, abs=0.025)


def test_side_flip_on_deterministic_bars():
    bars = ramp_bars()
    res = ramp_strategy(bars, longs=[5, 15, 25, 35, 45, 55], shorts=[65, 75, 85, 95])
    out = side_test(bars, res, mode="flip")
    # Flipped: 4 longs (+2R) and 6 shorts (-1R) -> (8 - 6) / 10.
    assert out.test == "side_flip"
    assert out.null.tolist() == pytest.approx([0.2])
    assert out.excess == pytest.approx(0.6)
    assert out.p_value is None and out.percentile is None and out.z_score is None
    assert out.n_sims == 1 and out.seed is None
    assert "flipped +0.2000" in str(out)


def test_p_value_formula_and_total_r():
    bars = ramp_bars()
    res = ramp_strategy(bars, longs=[5, 15, 25, 35, 45, 55], shorts=[65, 75, 85, 95])
    out = random_entry_test(bars, res, n_sims=999, seed=4, sides="random", metric="total_r")
    tol = 1e-12 * max(1.0, abs(out.observed))
    expected = (1 + int((out.null >= out.observed - tol).sum())) / 1000
    assert out.p_value == expected
    assert out.observed == pytest.approx(8.0)


# ------------------------------------------------- exact trade-count guard


def test_every_simulation_has_exactly_the_strategy_trade_count():
    # Regression: an earlier implementation sampled 3n candidate bars and
    # produced about 2.5x the strategy's trade count per simulation.
    bars = random_walk_bars(3000, seed=11)
    res = bracket_trades(bars, random_sides(3000, 0.05, 1), 0.01, 0.015, costs=COSTS)
    n = len(res.trades)
    counts = random_entry_test(bars, res, n_sims=300, seed=0, metric=len)
    assert counts.n_trades == n
    assert counts.null.tolist() == [float(n)] * 300
    win = random_entry_test(bars, res, n_sims=300, seed=0, metric="win_rate")
    np.testing.assert_allclose(win.null * n, np.round(win.null * n), atol=1e-9)


def test_null_width_matches_the_strategy_trade_count():
    # A 1:1 bracket with no costs on a martingale gives R = +-1 with
    # probability 1/2, so the null's sd is about sqrt((1 - n/m) / n). With
    # 2.5n trades per simulation it would be about 1.55x narrower.
    bars = random_walk_bars(4000, seed=12)
    res = bracket_trades(bars, random_sides(4000, 0.03, 2), 0.01, 0.01)
    n = len(res.trades)
    out = random_entry_test(bars, res, n_sims=1000, seed=0)
    expected_sd = math.sqrt((1 - n / out.n_eligible) / n)
    assert 0.85 < out.null_std / expected_sd < 1.15


def test_unfillable_setups_are_redrawn_not_dropped():
    # Large overnight-style gaps and a tight stop: many random setups open
    # beyond their stop. Each gets a fresh bar, so the count stays exact.
    bars = random_walk_bars(2500, seed=13, gap_vol=0.006)
    side = random_sides(2500, 0.05, 3)
    res = bracket_trades(bars, side, 0.004, 0.008)
    n = len(res.trades)
    out = random_entry_test(bars, res, n_sims=200, seed=0, metric=len)
    assert out.unfilled > 0
    assert out.null.tolist() == [float(n)] * 200
    assert "unfilled random setups" in str(out)


# ------------------------------------------------------- negative controls


def test_random_strategy_on_random_walk_is_not_flagged():
    p_values = []
    for seed in range(40):
        bars = random_walk_bars(1500, seed=1000 + seed)
        res = bracket_trades(bars, random_sides(1500, 0.04, seed), 0.008, 0.012, costs=COSTS)
        p_values.append(random_entry_test(bars, res, n_sims=200, seed=seed).p_value)
    p = np.array(p_values)
    # Roughly uniform: centred near 0.5 and not piling up near zero.
    assert 0.38 < p.mean() < 0.62
    assert (p < 0.05).mean() <= 0.125
    assert 0.3 < (p < 0.5).mean() < 0.7


def test_coin_direction_at_good_timing_has_no_side_skill():
    for seed in range(4):
        bars, signal = planted_signal_bars(4000, edge=0.5, signal_prob=0.1, seed=seed)
        coin = np.random.default_rng(100 + seed).choice([-1, 1], len(signal)) * (signal != 0)
        res = bracket_trades(bars, coin, 0.005, 0.005, costs=COSTS)
        out = side_test(bars, res, n_sims=300, seed=seed)
        assert out.p_value > 0.05


# ------------------------------------------------------- positive controls


def test_planted_signal_is_significant():
    bars, signal = planted_signal_bars(4000, edge=0.5, signal_prob=0.1, seed=0)
    res = bracket_trades(bars, signal, 0.005, 0.005, costs=COSTS)
    out = random_entry_test(bars, res, n_sims=300, seed=0)
    assert out.p_value < 0.01
    assert out.observed > out.null_mean
    assert out.z_score is not None and out.z_score > 3


def test_side_test_finds_real_direction_skill():
    bars, signal = planted_signal_bars(4000, edge=0.5, signal_prob=0.1, seed=1)
    res = bracket_trades(bars, signal, 0.005, 0.005, costs=COSTS)
    out = side_test(bars, res, n_sims=300, seed=1)
    assert out.p_value < 0.01
    flip = side_test(bars, res, mode="flip")
    assert flip.null_mean < 0 < flip.observed
    assert flip.excess > 0.3


def test_positive_control_catches_the_oracle():
    out = positive_control(random_walk_bars(1500, seed=21), n_sims=200, seed=0)
    assert out.p_value < 0.01
    assert out.settings["costs"]["fee_rate"] == 0.0


def test_positive_control_refuses_fees_but_accepts_spread():
    bars = random_walk_bars(800, seed=22)
    with pytest.raises(ValueError, match="fee_rate=0"):
        positive_control(bars, costs=Costs(fee_rate=0.001))
    out = positive_control(bars, n_sims=150, costs=Costs(half_spread=0.01))
    assert out.p_value < 0.01


# ----------------------------------------------------------- eligible mask


def test_eligible_mask_and_warmup_are_respected():
    bars = random_walk_bars(1200, seed=31)
    res = bracket_trades(bars, random_sides(1200, 0.05, 4), 0.01, 0.01)
    mask = np.zeros(len(bars), dtype=bool)
    mask[::3] = True
    seen = []

    def geometry(b, j, side):
        seen.append(j)
        c = b.close[j]
        return c * (1 - side * 0.01), c * (1 + side * 0.01)

    out = random_entry_test(
        bars, res, n_sims=30, seed=0, eligible=mask, warmup=200, geometry=geometry
    )
    seen = np.array(seen)
    assert mask[seen].all()
    assert seen.min() >= 200 and seen.max() <= len(bars) - 2
    assert out.n_eligible == int(mask[200 : len(bars) - 1].sum())
    assert out.settings["geometry"] == "geometry"
    assert any("outside the random-entry pool" in note for note in out.notes)


def test_eligible_mask_separates_timing_from_regime_selection():
    # An up-trend followed by a down-trend; the strategy buys at random bars
    # inside the up-trend. Against random entries from all bars it looks
    # skilled; against random entries from its own regime it does not.
    up = random_walk_bars(2000, drift=0.004, seed=7)
    down = random_walk_bars(2000, drift=-0.004, seed=8)
    scale = up.close[-1] / down.open[0]
    bars = Bars(
        np.concatenate([up.open, down.open * scale]),
        np.concatenate([up.high, down.high * scale]),
        np.concatenate([up.low, down.low * scale]),
        np.concatenate([up.close, down.close * scale]),
    )
    side = np.zeros(len(bars))
    side[:2000] = np.random.default_rng(3).random(2000) < 0.15
    res = bracket_trades(bars, side, 0.01, 0.01, costs=COSTS)
    regime = np.zeros(len(bars), dtype=bool)
    regime[:2000] = True
    everywhere = random_entry_test(bars, res, n_sims=300, seed=0)
    in_regime = random_entry_test(bars, res, n_sims=300, seed=0, eligible=regime)
    assert everywhere.p_value < 0.01
    assert in_regime.p_value > 0.1
    assert not in_regime.notes


# ------------------------------------------------------------- matching


def test_sides_options_reach_the_geometry():
    bars = random_walk_bars(1500, seed=41)
    side = random_sides(1500, 0.05, 5)
    res = bracket_trades(bars, side, 0.01, 0.01)
    n = len(res.trades)
    n_long = sum(t.side == 1 for t in res.trades)
    assert 0 < n_long < n
    calls = []

    def geometry(b, j, s):
        calls.append(s)
        c = b.close[j]
        return c * (1 - s * 0.01), c * (1 + s * 0.01)

    random_entry_test(bars, res, n_sims=20, seed=0, geometry=geometry)
    first = np.array(calls[: 20 * n]).reshape(20, n)
    assert ((first == 1).sum(axis=1) == n_long).all()
    for sides, value in (("long", 1), ("short", -1)):
        calls.clear()
        random_entry_test(bars, res, n_sims=5, seed=0, geometry=geometry, sides=sides)
        assert set(calls) == {value}
    calls.clear()
    random_entry_test(bars, res, n_sims=20, seed=0, geometry=geometry, sides="random")
    first = np.array(calls[: 20 * n]).reshape(20, n)
    assert len({int(row.sum()) for row in first}) > 1


def test_backtest_result_supplies_engine_settings():
    bars = random_walk_bars(2000, seed=42)
    costs = Costs(half_spread=0.01, fee_rate=0.0002, slippage=0.005)
    res = bracket_trades(
        bars, random_sides(2000, 0.05, 6), 0.015, 0.03, costs=costs, max_bars=3, entry_delay=1
    )
    assert any(t.reason == "time" for t in res.trades)
    out = random_entry_test(bars, res, n_sims=50, seed=0)
    assert out.settings["max_bars"] == 3
    assert out.settings["entry_delay"] == 1
    assert out.settings["costs"]["slippage"] == 0.005
    # The same trades as a plain list need the settings spelled out ...
    with pytest.raises(ValueError, match="does not reproduce"):
        random_entry_test(bars, list(res.trades), n_sims=50, seed=0)
    # ... and give the same answer once they are.
    explicit = random_entry_test(bars, list(res.trades), n_sims=50, seed=0, costs=costs, max_bars=3)
    np.testing.assert_array_equal(explicit.null, out.null)


def test_wrong_costs_are_caught_instead_of_compared():
    bars = random_walk_bars(1500, seed=43)
    res = bracket_trades(bars, random_sides(1500, 0.05, 7), 0.01, 0.01, costs=COSTS)
    with pytest.raises(ValueError, match="does not reproduce"):
        random_entry_test(bars, res.trades, n_sims=20)
    with pytest.raises(ValueError, match="does not reproduce"):
        side_test(bars, res.trades, n_sims=20, costs=COSTS.scaled(2))
    unchecked = random_entry_test(bars, res.trades, n_sims=20, verify=False)
    assert unchecked.n_trades == len(res.trades)


def test_works_with_trades_from_run():
    bars = random_walk_bars(1500, seed=44)
    side = random_sides(1500, 0.05, 8)

    def decide(i, history):
        if side[i] == 0:
            return None
        c = history.close[-1]
        return (int(side[i]), c * (1 - side[i] * 0.01), c * (1 + side[i] * 0.02))

    res = run(bars, decide, costs=COSTS, max_bars=10)
    out = random_entry_test(bars, res, n_sims=100, seed=0)
    assert out.n_trades == len(res.trades)
    assert 0 < out.p_value <= 1


def test_seeds_are_reproducible():
    bars = random_walk_bars(1500, seed=45)
    res = bracket_trades(bars, random_sides(1500, 0.05, 9), 0.01, 0.015, costs=COSTS)
    a = random_entry_test(bars, res, n_sims=200, seed=7)
    b = random_entry_test(bars, res, n_sims=200, seed=7)
    c = random_entry_test(bars, res, n_sims=200, seed=8)
    np.testing.assert_array_equal(a.null, b.null)
    assert a.p_value == b.p_value
    assert not np.array_equal(a.null, c.null)
    s1 = side_test(bars, res, n_sims=200, seed=7)
    s2 = side_test(bars, res, n_sims=200, seed=7)
    np.testing.assert_array_equal(s1.null, s2.null)


def test_callable_metric():
    bars = random_walk_bars(1500, seed=46)
    res = bracket_trades(bars, random_sides(1500, 0.05, 10), 0.01, 0.015, costs=COSTS)

    def median_r(r):
        return float(np.median(r))

    out = random_entry_test(bars, res, n_sims=50, seed=0, metric=median_r)
    assert out.metric == "median_r"
    assert out.observed == median_r(res.r)
    with pytest.raises(ValueError, match="non-finite"):
        random_entry_test(bars, res, n_sims=5, metric=lambda r: math.nan)


# ------------------------------------------------------------- results


def test_result_as_dict_and_str():
    bars = random_walk_bars(1500, seed=47)
    res = bracket_trades(bars, random_sides(1500, 0.05, 11), 0.01, 0.015, costs=COSTS)
    out = random_entry_test(bars, res, n_sims=100, seed=0)
    assert isinstance(out, NullTestResult)
    d = out.as_dict()
    assert d["test"] == "random_entry"
    assert d["n_sims"] == 100 and len(d["null"]) == 100
    assert d["excess"] == pytest.approx(out.observed - out.null_mean)
    assert d["settings"]["sides"] == "match"
    text = str(out)
    assert text.startswith("Random-entry test (expectancy")
    assert f"{out.n_trades} trades" in text and "p = " in text
    with pytest.raises(ValueError):
        out.null[0] = 1.0  # read-only
    with pytest.raises(AttributeError):
        out.p_value = 0.0  # frozen


# ------------------------------------------------------------ bad input


def test_bad_inputs_raise():
    bars = random_walk_bars(600, seed=48)
    res = bracket_trades(bars, random_sides(600, 0.05, 12), 0.01, 0.01)
    with pytest.raises(ValueError, match="at least 5 trades"):
        random_entry_test(bars, res.trades[:4])
    with pytest.raises(ValueError, match="sides"):
        random_entry_test(bars, res, sides="both")
    with pytest.raises(ValueError, match="geometry"):
        random_entry_test(bars, res, geometry="atr")
    with pytest.raises(ValueError, match="metric"):
        random_entry_test(bars, res, metric="sharpe")
    with pytest.raises(ValueError, match="n_sims"):
        random_entry_test(bars, res, n_sims=0)
    with pytest.raises(ValueError, match="boolean"):
        random_entry_test(bars, res, eligible=np.ones(len(bars)))
    with pytest.raises(ValueError, match="one value per bar"):
        random_entry_test(bars, res, eligible=np.ones(10, dtype=bool))
    few = np.zeros(len(bars), dtype=bool)
    few[:3] = True
    with pytest.raises(ValueError, match="eligible signal bars"):
        random_entry_test(bars, res, eligible=few)
    with pytest.raises(ValueError, match="mode"):
        side_test(bars, res, mode="reverse")
    with pytest.raises(ValueError, match="Bars"):
        random_entry_test(bars.close, res)
    with pytest.raises(ValueError, match="Trade objects"):
        random_entry_test(bars, [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="non-finite"):
        random_entry_test(bars, res, n_sims=5, geometry=lambda b, j, s: (math.nan, 1.0))
    with pytest.raises(ValueError, match="right bars"):
        random_entry_test(bars[:100], res)


def test_mixed_entry_delays_need_an_explicit_delay():
    bars = random_walk_bars(1500, seed=49)
    side = random_sides(1500, 0.05, 13)
    side_a, side_b = side.copy(), side.copy()
    side_a[750:] = 0
    side_b[:750] = 0
    a = bracket_trades(bars, side_a, 0.01, 0.01)
    b = bracket_trades(bars, side_b, 0.01, 0.01, entry_delay=1)
    with pytest.raises(ValueError, match="entry delays"):
        random_entry_test(bars, a.trades + b.trades)


def test_wrong_side_geometry_cannot_be_copied():
    bars = ramp_bars()
    trades = list(ramp_strategy(bars, longs=[5, 15, 25, 35, 45, 55], shorts=[]).trades)
    t = trades[0]
    # A long whose stop sits above its signal bar's close (it filled only
    # because the next bar opened higher): its geometry is not a distance.
    trades[0] = Trade(**{**t.as_dict(), "stop": bars.close[t.signal_index] * 1.001})
    with pytest.raises(ValueError, match="wrong side"):
        random_entry_test(bars, trades, verify=False)


# ---------------------------------------------------------------- speed


def test_thousand_simulations_of_three_hundred_trades_is_fast():
    bars = random_walk_bars(20000, seed=50)
    side = np.zeros(len(bars))
    picks = np.random.default_rng(4).choice(len(bars) - 2, 300, replace=False)
    side[picks] = np.random.default_rng(5).choice([-1, 1], 300)
    res = bracket_trades(bars, side, 0.01, 0.015, costs=COSTS)
    assert len(res.trades) >= 290
    start = time.perf_counter()
    out = random_entry_test(bars, res, n_sims=1000, seed=0)
    elapsed = time.perf_counter() - start
    assert out.null.shape == (1000,)
    assert elapsed < 5.0, f"1000 x {len(res.trades)} random-entry simulations took {elapsed:.2f}s"
