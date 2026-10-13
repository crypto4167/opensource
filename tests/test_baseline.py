import dataclasses
import itertools
import json
import math

import numpy as np
import pytest

from backtest_skeptic import Costs, Trade, resolve_bracket, run_signals
from backtest_skeptic.baseline import (
    BaselineResult,
    bracket_odds,
    breakeven_win_rate,
    expectancy_decomposition,
    no_skill_test,
    poisson_binomial_pmf,
)
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars
from helpers import flat_then

VOL = 0.002
COSTS = Costs(half_spread=0.1, fee_rate=0.001, slippage=0.05)


def random_sides(n, seed, prob=0.3):
    """Skill-less entries: a random side on a random 30% of bars."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n) * (rng.random(n) < prob)


def brackets(bars, side, stop_vols, target_vols):
    """Stop and target placed ``stop_vols``/``target_vols`` bar volatilities from the close."""
    c = bars.close
    return c - side * stop_vols * VOL * c, c + side * target_vols * VOL * c


@pytest.fixture(scope="module")
def walk():
    return random_walk_bars(40_000, vol=VOL, seed=11)


@pytest.fixture(scope="module")
def planted():
    return planted_signal_bars(40_000, edge=0.3, signal_prob=0.1, vol=VOL, seed=7)


def long_target_trade(costs=COSTS):
    # Signal at bar 0 (close 100), fill at bar 1's open (100), target hit on bar 2.
    bars = flat_then([(100.0, 101.0, 99.5, 100.5), (100.5, 103.5, 100.2, 103.0)])
    t = resolve_bracket(bars, 0, (1, 98.0, 103.0), costs=costs)
    assert t is not None and t.reason == "target"
    return t


def long_stop_trade(costs=COSTS):
    bars = flat_then([(100.0, 100.5, 97.5, 98.0)])
    t = resolve_bracket(bars, 0, (1, 98.0, 103.0), costs=costs)
    assert t is not None and t.reason == "stop"
    return t


# ---------------------------------------------------------------------------
# hand-computed arithmetic
# ---------------------------------------------------------------------------


def test_hand_long_target_with_spread_fees_and_slippage():
    t = long_target_trade()
    # m = 100, a = 2, b = 3; entry = 100.15, risk = 2.15.
    assert t.entry_price == pytest.approx(100.15)
    r_win = (102.9 - 100.15 - 0.001 * (100.15 + 102.9)) / 2.15
    r_loss = (97.85 - 100.15 - 0.001 * (100.15 + 97.85)) / 2.15
    p_star = -r_loss / (r_win - r_loss)
    assert t.r == pytest.approx(r_win)

    res = no_skill_test([t], costs=COSTS)
    assert res.n_resolved == 1 and res.wins == 1
    assert res.no_skill_rate == pytest.approx(0.4)
    assert res.breakeven_rate == pytest.approx(p_star)
    assert res.breakeven_rate == pytest.approx(2.498 / (2.54695 + 2.498))
    assert res.skill_edge == pytest.approx(0.6)
    assert res.cost_drag == pytest.approx(p_star - 0.4)
    assert res.margin == pytest.approx(1.0 - p_star)
    assert res.expected_wins == pytest.approx(0.4)
    assert res.z == pytest.approx(0.6 / math.sqrt(0.24))
    assert res.p_value == pytest.approx(0.4)
    assert res.p_value_lower == pytest.approx(1.0)
    assert res.method == "exact"
    assert res.no_skill_expectancy_r == pytest.approx(0.4 * r_win + 0.6 * r_loss)

    odds = bracket_odds(t, costs=COSTS)
    assert (odds.stop_distance, odds.target_distance) == pytest.approx((2.0, 3.0))
    assert odds.r_win == pytest.approx(r_win)
    assert odds.r_loss == pytest.approx(r_loss)
    assert odds.breakeven_prob == pytest.approx(p_star)


def test_hand_short_stop_is_the_mirror_image():
    bars = flat_then([(100.0, 100.5, 99.5, 100.2), (100.2, 102.4, 99.8, 102.1)])
    t = resolve_bracket(bars, 0, ("SELL", 102.0, 97.0), costs=COSTS)
    assert t is not None and t.reason == "stop" and t.side == -1
    r_win = (2.75 - 0.001 * (99.85 + 97.1)) / 2.15
    r_loss = (-2.3 - 0.001 * (99.85 + 102.15)) / 2.15
    assert t.r == pytest.approx(r_loss)

    res = no_skill_test([t], costs=COSTS)
    assert res.wins == 0 and res.gap_fills == 0
    assert res.no_skill_rate == pytest.approx(0.4)
    assert res.breakeven_rate == pytest.approx(-r_loss / (r_win - r_loss))
    assert res.p_value == pytest.approx(1.0)
    assert res.p_value_lower == pytest.approx(0.6)


def test_breakeven_win_rate_helper():
    assert breakeven_win_rate(1.0, 9.0) == pytest.approx(0.1)
    # A target one ninth the size of the stop needs a 90% win rate to break even.
    assert breakeven_win_rate(9.0, 1.0) == pytest.approx(0.9)
    assert breakeven_win_rate(1.0, 1.0, cost_r=0.1) == pytest.approx(0.55)
    assert breakeven_win_rate(1.0, 0.05, cost_r=0.1) > 1.0  # costs eat the reward
    for bad in [(0.0, 1.0), (1.0, -1.0), (math.nan, 1.0), ("1", 1.0)]:
        with pytest.raises(ValueError):
            breakeven_win_rate(*bad)
    with pytest.raises(ValueError):
        breakeven_win_rate(1.0, 1.0, cost_r=-0.1)


def test_breakeven_helper_equals_no_skill_rate_without_costs():
    t = long_target_trade(costs=Costs())
    res = no_skill_test([t])
    assert res.breakeven_rate == pytest.approx(res.no_skill_rate)
    assert res.breakeven_rate == pytest.approx(breakeven_win_rate(2.0, 3.0))
    assert res.cost_drag == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# regression tests for the three mistakes
# ---------------------------------------------------------------------------


def test_bug1_baseline_is_anchored_at_the_fill_open_not_the_signal_close():
    # Signal close 100, next open gaps up to 100.5. From the signal close the
    # bracket looks like a = 1, b = 2 (1/3); the engine brackets from 100.5,
    # where a = b = 1.5 (1/2).
    bars = flat_then([(100.5, 101.0, 100.2, 100.8), (100.8, 102.3, 100.6, 102.0)])
    t = resolve_bracket(bars, 0, (1, 99.0, 102.0))
    assert t is not None and t.reason == "target" and t.mid_entry == 100.5
    res = no_skill_test([t])
    assert res.no_skill_rate == pytest.approx(0.5)
    odds = bracket_odds(t)
    assert (odds.stop_distance, odds.target_distance) == pytest.approx((1.5, 1.5))


def test_bug1_selection_on_the_fill_open_does_not_create_fake_skill(walk):
    # Skill-less entries filled a bar late, keeping only fills that moved away
    # from the stop (risk_band). The surviving brackets genuinely have better
    # odds; a baseline taken from the signal close calls that skill.
    side = random_sides(len(walk), 200)
    stop, target = brackets(walk, side, 3, 3)
    bt = run_signals(walk, side, stop, target, entry_delay=1, risk_band=(1.0, 10.0))
    res = no_skill_test(bt)
    assert res.n_resolved > 2000
    assert abs(res.z) < 3
    assert res.p_value > 0.01

    resolved = [t for t in bt.trades if t.reason in ("stop", "target")]
    close = walk.close
    a_sig = np.array([t.side * (close[t.signal_index] - t.stop) for t in resolved])
    b_sig = np.array([t.side * (t.target - close[t.signal_index]) for t in resolved])
    wrong_rate = float(np.mean(a_sig / (a_sig + b_sig)))
    se = math.sqrt(res.no_skill_rate * (1 - res.no_skill_rate) / res.n_resolved)
    assert (res.win_rate - wrong_rate) / se > 8  # what the old baseline would report
    assert res.no_skill_rate > wrong_rate + 0.1


def test_bug2_no_skill_trader_loses_the_round_trip_cost():
    h, s = 0.1, 0.05
    costs = Costs(half_spread=h, slippage=s)
    t = long_target_trade(costs=costs)
    res = no_skill_test([t], costs=costs)
    a, b, risk = 2.0, 3.0, 2.15
    expected = -(2 * h + s * (a + 2 * b) / (a + b)) / risk
    assert res.no_skill_expectancy_r == pytest.approx(expected)
    assert bracket_odds(t, costs=costs).no_skill_expectancy_r == pytest.approx(expected)
    assert res.breakeven_rate > res.no_skill_rate
    assert res.cost_drag > 0


def test_bug2_random_entries_with_costs_beat_no_skill_by_zero_and_lose_money(walk):
    costs = Costs(half_spread=0.02, fee_rate=0.0001, slippage=0.01)
    side = random_sides(len(walk), 101)
    stop, target = brackets(walk, side, 3, 1)
    res = no_skill_test(run_signals(walk, side, stop, target, costs=costs))
    assert abs(res.z) < 3 and res.p_value > 0.01
    assert res.cost_drag > 0.05
    assert res.margin < 0  # edge over no-skill is not profitability
    assert res.no_skill_expectancy_r < 0
    se_r = 1.5 / math.sqrt(res.n_resolved)
    assert res.mean_r == pytest.approx(res.no_skill_expectancy_r, abs=3 * se_r)


def test_bug3_time_exit_winners_are_not_counted_as_wins():
    # Drifts up without touching either level; max_bars closes it in profit.
    up = flat_then([(100.0, 100.8, 99.6, 100.6), (100.6, 101.6, 100.4, 101.5)])
    timed = resolve_bracket(up, 0, (1, 98.0, 103.0), max_bars=2)
    assert timed is not None and timed.reason == "time" and timed.r > 0
    stopped = long_stop_trade(costs=Costs())
    ended = resolve_bracket(flat_then([(100.0, 100.8, 99.6, 100.6)]), 0, (1, 98.0, 103.0))
    assert ended is not None and ended.reason == "end"

    res = no_skill_test([timed, stopped, ended])
    assert res.n_trades == 3 and res.n_resolved == 1
    assert res.excluded == {"time": 1, "end": 1, "invalid_geometry": 0}
    assert res.wins == 0 and res.win_rate == 0.0
    assert res.unresolved_share == pytest.approx(2 / 3)
    assert any("random_entry_test" in w for w in res.warnings)
    # Exclusion-free score: time exit closed 70% of the way to the target,
    # end exit 52%; every bracket has p0 = 0.4.
    assert res.all_exits_n == 3
    assert res.all_exits_edge == pytest.approx(((0.7 - 0.4) + (0 - 0.4) + (0.52 - 0.4)) / 3)
    assert res.exclusion_bias == pytest.approx((0.4 - 0.7) + (0.4 - 0.52))


def test_bug3_wide_targets_with_time_exits_do_not_look_skilled(walk):
    side = random_sides(len(walk), 400)
    stop, target = brackets(walk, side, 1, 5)
    bt = run_signals(walk, side, stop, target, max_bars=6)
    res = no_skill_test(bt)
    assert res.excluded["time"] > 1000
    # The old mistake: profitable time exits counted as wins against a/(a+b).
    p0 = np.array([t.side * (t.mid_entry - t.stop) for t in bt.trades]) / np.array(
        [abs(t.target - t.stop) for t in bt.trades]
    )
    naive_edge = float(np.mean(bt.r > 0) - p0.mean())
    assert naive_edge > 0.08
    assert res.p_value > 0.5  # no fake skill from the time exits
    assert any("random_entry_test" in w for w in res.warnings)
    assert abs(res.all_exits_z) < 3


# ---------------------------------------------------------------------------
# excluding time exits is itself a selection: it must be flagged
# ---------------------------------------------------------------------------


def test_tight_targets_with_many_time_exits_warn_and_all_exits_stays_null(walk):
    side = random_sides(len(walk), 500)
    stop, target = brackets(walk, side, 3, 1)
    res = no_skill_test(run_signals(walk, side, stop, target, max_bars=4))
    assert res.unresolved_share > 0.2
    # Resolved-only rate is biased toward the nearer level (the target here):
    # this is why the warning exists.
    assert res.z > 5
    se = math.sqrt(res.no_skill_rate * (1 - res.no_skill_rate) / res.n_resolved)
    assert res.exclusion_bias == pytest.approx(res.skill_edge, abs=3 * se)
    assert any("all_exits_edge" in w for w in res.warnings)
    assert abs(res.all_exits_z) < 3
    assert res.all_exits_p_value > 0.01


def test_time_exits_below_ten_percent_still_warn_when_the_bias_is_material(walk):
    side = random_sides(len(walk), 601)
    stop, target = brackets(walk, side, 3, 1)
    res = no_skill_test(run_signals(walk, side, stop, target, max_bars=8))
    assert 0 < res.unresolved_share < 0.10
    assert any("standard errors" in w for w in res.warnings)
    assert abs(res.all_exits_z) < 3


# ---------------------------------------------------------------------------
# negative and positive controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("stop_vols", "target_vols"), [(3, 1), (1, 3), (9, 1)])
def test_random_entries_on_a_random_walk_show_no_skill(walk, stop_vols, target_vols):
    costs = Costs(half_spread=0.005, slippage=0.005)
    side = random_sides(len(walk), 100)
    stop, target = brackets(walk, side, stop_vols, target_vols)
    res = no_skill_test(run_signals(walk, side, stop, target, costs=costs))
    assert res.n_resolved > 2000
    assert abs(res.z) < 3
    assert res.p_value > 0.01
    assert res.margin < 0
    assert res.cost_drag > 0
    assert res.no_skill_rate == pytest.approx(stop_vols / (stop_vols + target_vols), abs=0.01)
    assert res.warnings == ()


def test_a_ninety_percent_win_rate_is_just_geometry(walk):
    side = random_sides(len(walk), 102)
    stop, target = brackets(walk, side, 9, 1)
    res = no_skill_test(run_signals(walk, side, stop, target, costs=COSTS.scaled(0.1)))
    assert res.win_rate > 0.87
    assert res.p_value > 0.01
    assert res.margin < 0


def test_planted_signal_is_detected(planted):
    bars, signal = planted
    stop, target = brackets(bars, signal, 1.5, 1.5)
    res = no_skill_test(run_signals(bars, signal, stop, target))
    assert res.skill_edge > 0.05
    assert res.p_value < 0.01
    assert res.z > 5
    assert res.p_value_lower > 0.99


def test_inverted_planted_signal_shows_up_in_the_lower_tail(planted):
    bars, signal = planted
    stop, target = brackets(bars, -signal, 1.5, 1.5)
    res = no_skill_test(run_signals(bars, -signal, stop, target))
    assert res.p_value_lower < 0.01
    assert res.p_value > 0.99


def test_planted_signal_is_detected_by_all_exits_score_despite_time_exits(planted):
    bars, signal = planted
    stop, target = brackets(bars, signal, 3, 3)
    res = no_skill_test(run_signals(bars, signal, stop, target, max_bars=3))
    assert res.unresolved_share > 0.5
    assert res.all_exits_edge > 0.02
    assert res.all_exits_p_value < 0.01


def test_gap_fills_are_counted_and_warned():
    bars = flat_then([(100, 100.5, 99.5, 100.0), (98.0, 98.5, 97.5, 98.2)])
    gapped = resolve_bracket(bars, 0, (1, 99.0, 102.0))
    assert gapped is not None and gapped.r == pytest.approx(-2.0)
    clean = long_target_trade(costs=Costs())
    res = no_skill_test([gapped, clean])
    assert res.gap_fills == 1
    assert any("gaps" in w for w in res.warnings)


def test_gap_fills_on_a_gapping_random_walk(walk):
    bars = random_walk_bars(20_000, vol=VOL, gap_vol=VOL, seed=12)
    side = random_sides(len(bars), 300)
    stop, target = brackets(bars, side, 3, 3)
    res = no_skill_test(run_signals(bars, side, stop, target))
    assert res.gap_fills > 50
    assert abs(res.z) < 3.5


def test_ambiguous_wins_are_flagged():
    bars = flat_then([(100, 102.5, 98.5, 100.0)])
    t = resolve_bracket(bars, 0, (1, 99.0, 102.0), same_bar="target")
    assert t is not None and t.reason == "target" and t.ambiguous
    res = no_skill_test([t])
    assert res.ambiguous == 1 and res.ambiguous_wins == 1
    assert any("touched both" in w for w in res.warnings)
    t_stop = resolve_bracket(bars, 0, (1, 99.0, 102.0))
    res_stop = no_skill_test([t_stop])
    assert res_stop.ambiguous == 1 and res_stop.ambiguous_wins == 0
    assert res_stop.warnings == ()


def test_invalid_geometry_is_excluded_with_a_warning():
    costs = Costs(half_spread=0.03, slippage=0.02)
    # The fill-bar open (99.98) is already below the stop (99.99), yet the
    # costs push the entry fill above it. This engine skips such setups, but
    # trade logs from other backtesters can contain them.
    bars = flat_then([(99.98, 100.2, 99.9, 100.1)])
    assert resolve_bracket(bars, 0, (1, 99.99, 101.0), costs=costs) is None
    bad = Trade(
        side=1,
        signal_index=0,
        entry_index=1,
        exit_index=1,
        mid_entry=99.98,
        entry_price=99.98 + 0.05,
        exit_price=99.98 - 0.05,
        stop=99.99,
        target=101.0,
        reason="stop",
        r=(99.93 - 100.03) / (100.03 - 99.99),
        risk=100.03 - 99.99,
        reward=101.0 - 100.03,
        fees=0.0,
    )
    assert bad.mid_entry < bad.stop
    good = long_stop_trade(costs=costs)
    res = no_skill_test([bad, good], costs=costs)
    assert res.excluded["invalid_geometry"] == 1
    assert res.n_resolved == 1
    assert any("beyond the stop and were excluded" in w for w in res.warnings)
    with pytest.raises(ValueError, match="undefined"):
        bracket_odds(bad, costs=costs)


# ---------------------------------------------------------------------------
# Poisson-binomial distribution
# ---------------------------------------------------------------------------


def test_poisson_binomial_equals_binomial_when_probabilities_are_equal():
    n, p = 40, 0.3
    pmf = poisson_binomial_pmf(np.full(n, p))
    binom = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    np.testing.assert_allclose(pmf, binom, rtol=1e-10, atol=1e-15)


def test_poisson_binomial_matches_brute_force_enumeration():
    p = np.random.default_rng(3).uniform(0.05, 0.95, size=10)
    brute = np.zeros(11)
    for outcome in itertools.product((0, 1), repeat=10):
        o = np.array(outcome)
        brute[o.sum()] += float(np.prod(np.where(o == 1, p, 1 - p)))
    np.testing.assert_allclose(poisson_binomial_pmf(p), brute, rtol=1e-12, atol=1e-16)


def test_poisson_binomial_edge_cases():
    np.testing.assert_array_equal(poisson_binomial_pmf([]), [1.0])
    np.testing.assert_allclose(poisson_binomial_pmf([0.0, 1.0, 1.0]), [0, 0, 1, 0], atol=0)
    assert poisson_binomial_pmf(np.full(500, 0.5)).sum() == pytest.approx(1.0)
    for bad in ([0.5, 1.2], [-0.1], [math.nan], [[0.5, 0.5]]):
        with pytest.raises(ValueError):
            poisson_binomial_pmf(bad)


def test_no_skill_p_value_matches_brute_force_tail():
    p = np.random.default_rng(5).uniform(0.2, 0.8, size=8)
    pmf = poisson_binomial_pmf(p)
    for wins in range(9):
        brute = sum(
            float(np.prod(np.where(np.array(o) == 1, p, 1 - p)))
            for o in itertools.product((0, 1), repeat=8)
            if sum(o) >= wins
        )
        assert pmf[wins:].sum() == pytest.approx(brute, rel=1e-12)


def test_normal_approximation_is_flagged_and_close_to_exact(walk):
    side = random_sides(len(walk), 103)
    stop, target = brackets(walk, side, 2, 2)
    bt = run_signals(walk, side, stop, target)
    exact = no_skill_test(bt)
    approx = no_skill_test(bt, exact_limit=100)
    assert exact.method == "exact" and approx.method == "normal"
    assert any("normal approximation" in w for w in approx.warnings)
    assert approx.p_value == pytest.approx(exact.p_value, abs=0.01)
    assert approx.p_value_lower == pytest.approx(exact.p_value_lower, abs=0.01)
    assert approx.z == exact.z


# ---------------------------------------------------------------------------
# expectancy decomposition
# ---------------------------------------------------------------------------


def test_decomposition_identity_holds_on_engine_trades_without_gaps(walk, planted):
    costs = Costs(half_spread=0.01, fee_rate=0.0002, slippage=0.005)
    side = random_sides(len(walk), 104)
    stop, target = brackets(walk, side, 3, 2)
    bt_null = run_signals(walk, side, stop, target, costs=costs)
    bars, signal = planted
    stop_p, target_p = brackets(bars, signal, 1.5, 1.5)
    bt_skill = run_signals(bars, signal, stop_p, target_p, costs=costs)
    for bt in (bt_null, bt_skill):
        dec = expectancy_decomposition(bt)
        assert dec.excluded["gap_fill"] == 0
        assert abs(dec.residual_r) < 1e-9
        assert dec.mean_r == pytest.approx(dec.skill_r + dec.cost_r, abs=1e-9)
        resolved = [t for t in bt.trades if t.reason in ("stop", "target")]
        assert dec.n_used == len(resolved)
        assert dec.mean_r == pytest.approx(np.mean([t.r for t in resolved]), abs=1e-12)
        for t in resolved[:300]:  # the identity holds trade by trade
            o = bracket_odds(t, costs=costs)
            w = 1.0 if t.reason == "target" else 0.0
            assert t.r == pytest.approx((w - o.breakeven_prob) * (o.r_win - o.r_loss), abs=1e-9)

    null = expectancy_decomposition(bt_null)
    skill = expectancy_decomposition(bt_skill)
    se = 2.5 / math.sqrt(null.n_used)
    assert abs(null.skill_r) < 3 * se
    assert null.cost_r < 0
    assert null.cost_r == pytest.approx(no_skill_test(bt_null).no_skill_expectancy_r)
    assert skill.skill_r > 0.1
    assert skill.cost_r < 0 and skill.mean_r < skill.skill_r  # costs take a bite of real skill


def test_decomposition_hand_values():
    t = long_target_trade()
    r_win = (102.9 - 100.15 - 0.001 * (100.15 + 102.9)) / 2.15
    r_loss = (97.85 - 100.15 - 0.001 * (100.15 + 97.85)) / 2.15
    p_star = -r_loss / (r_win - r_loss)
    dec = expectancy_decomposition([t], costs=COSTS)
    assert dec.skill_r == pytest.approx((1 - 0.4) * (r_win - r_loss))
    assert dec.cost_r == pytest.approx((0.4 - p_star) * (r_win - r_loss))
    assert dec.mean_r == pytest.approx(r_win)
    assert dec.residual_r == pytest.approx(0.0, abs=1e-12)


def test_decomposition_leaves_out_gap_fills_and_time_exits():
    bars = flat_then([(100, 100.5, 99.5, 100.0), (98.0, 98.5, 97.5, 98.2)])
    gapped = resolve_bracket(bars, 0, (1, 99.0, 102.0))
    up = flat_then([(100.0, 100.8, 99.6, 100.6), (100.6, 101.6, 100.4, 101.5)])
    timed = resolve_bracket(up, 0, (1, 98.0, 103.0), max_bars=2)
    clean = long_target_trade(costs=Costs())
    dec = expectancy_decomposition([gapped, timed, clean])
    assert dec.n_used == 1
    assert dec.excluded == {"time": 1, "end": 0, "invalid_geometry": 0, "gap_fill": 1}
    assert dec.mean_r == pytest.approx(clean.r)
    assert len(dec.warnings) == 2
    with pytest.raises(ValueError, match="gap fill"):
        expectancy_decomposition([gapped])
    with pytest.raises(ValueError, match="no trade exited"):
        expectancy_decomposition([timed])


# ---------------------------------------------------------------------------
# costs and input validation
# ---------------------------------------------------------------------------


def test_forgetting_the_costs_raises_instead_of_giving_wrong_breakeven(walk):
    costs = Costs(half_spread=0.02)
    side = random_sides(len(walk), 105)
    stop, target = brackets(walk, side, 2, 2)
    bt = run_signals(walk, side, stop, target, costs=costs)
    with pytest.raises(ValueError, match="entry cost"):
        no_skill_test(bt.trades)
    with pytest.raises(ValueError, match="differ"):
        no_skill_test(bt, costs=Costs())
    from_result = no_skill_test(bt)
    assert from_result.costs == costs
    assert no_skill_test(bt.trades, costs=costs).breakeven_rate == from_result.breakeven_rate
    assert no_skill_test(bt, costs=costs).breakeven_rate == from_result.breakeven_rate


def test_swapped_spread_and_slippage_are_caught_by_target_exits():
    t = long_target_trade()
    swapped = Costs(half_spread=0.05, fee_rate=0.001, slippage=0.1)  # same entry cost
    with pytest.raises(ValueError, match="target exit"):
        no_skill_test([t], costs=swapped)


def test_wrong_fee_rate_is_caught():
    with pytest.raises(ValueError, match="fees"):
        no_skill_test([long_target_trade()], costs=dataclasses.replace(COSTS, fee_rate=0.002))


def test_inconsistent_trade_records_are_rejected():
    t = long_stop_trade(costs=Costs())
    with pytest.raises(ValueError, match="does not match"):
        no_skill_test([dataclasses.replace(t, r=t.r + 0.1)])
    better = dataclasses.replace(t, exit_price=98.5, r=(98.5 - 100.0) / 2.0)
    with pytest.raises(ValueError, match="better than the stop"):
        no_skill_test([better])
    with pytest.raises(ValueError, match="exit reason"):
        no_skill_test([dataclasses.replace(t, reason="trail")])
    with pytest.raises(ValueError, match="risk"):
        no_skill_test([dataclasses.replace(t, risk=-2.0)])


def test_bad_inputs():
    t = long_target_trade(costs=Costs())
    with pytest.raises(ValueError, match="no trades"):
        no_skill_test([])
    with pytest.raises(TypeError):
        no_skill_test([t, 0.5])
    with pytest.raises(TypeError):
        no_skill_test(t)
    with pytest.raises(TypeError):
        no_skill_test([t], costs=0.01)
    with pytest.raises(ValueError):
        no_skill_test([t], tolerance=-1.0)
    with pytest.raises(ValueError):
        no_skill_test([t], exact_limit=-1)
    up = flat_then([(100.0, 100.8, 99.6, 100.6), (100.6, 101.6, 100.4, 101.5)])
    timed = resolve_bracket(up, 0, (1, 98.0, 103.0), max_bars=2)
    with pytest.raises(ValueError, match="random_entry_test"):
        no_skill_test([timed])


def test_results_are_frozen_serialisable_and_readable():
    up = flat_then([(100.0, 100.8, 99.6, 100.6), (100.6, 101.6, 100.4, 101.5)])
    timed = resolve_bracket(up, 0, (1, 98.0, 103.0), max_bars=2, costs=COSTS)
    trades = [long_target_trade(), long_stop_trade(), timed]
    res = no_skill_test(trades, costs=COSTS)
    assert isinstance(res, BaselineResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.wins = 5  # type: ignore[misc]
    d = res.as_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["excluded"] == {"time": 1, "end": 0, "invalid_geometry": 0}
    text = str(res)
    assert "skill edge" in text and "break-even" in text and "all exits" in text
    assert "warning:" in text

    dec = expectancy_decomposition(trades, costs=COSTS)
    assert json.loads(json.dumps(dec.as_dict())) == dec.as_dict()
    assert "skill" in str(dec) and "costs" in str(dec)
    odds = bracket_odds(trades[0], costs=COSTS)
    assert json.loads(json.dumps(odds.as_dict())) == odds.as_dict()
    assert "no-skill win rate 40.0%" in str(odds)
