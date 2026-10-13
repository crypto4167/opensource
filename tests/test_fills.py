import datetime
import json

import numpy as np
import pytest

from backtest_skeptic import Bars, Costs, Trade, run_signals
from backtest_skeptic.fills import CODES, FillAudit, Issue, TradeLog, TradeRow, audit_fills
from backtest_skeptic.synthetic import oracle_signal, planted_signal_bars, random_walk_bars
from helpers import bars_from_rows

COSTS = Costs(half_spread=0.01, fee_rate=0.0002, slippage=0.005)

# Bar 0 is the signal bar (close 100). A long enters at bar 1's open (100.0)
# with stop 99 and target 102 unless a test says otherwise.
SIGNAL = (100.0, 100.5, 99.5, 100.0)
ENTRY = (100.0, 100.6, 99.6, 100.2)
QUIET = (100.2, 100.8, 99.8, 100.4)


def long_trade(exit_index, exit_price, reason, **kw):
    rec = dict(
        side=1,
        signal_index=0,
        entry_index=1,
        exit_index=exit_index,
        entry_price=100.0,
        exit_price=exit_price,
        stop=99.0,
        target=102.0,
        reason=reason,
    )
    rec.update(kw)
    return rec


def codes(audit):
    return [i.code for i in audit.issues]


def engine_trades(seed, *, n=6_000, gap_vol=0.0, max_bars=None, same_bar="stop", costs=COSTS):
    bars = random_walk_bars(n, vol=0.004, gap_vol=gap_vol, seed=seed)
    rng = np.random.default_rng(seed + 100)
    side = rng.choice([-1, 0, 0, 0, 1], size=n)
    d = 0.004 * bars.close
    res = run_signals(
        bars,
        side,
        bars.close - side * 0.6 * d,
        bars.close + side * 0.8 * d,
        costs=costs,
        max_bars=max_bars,
        same_bar=same_bar,
    )
    return bars, res


def naive_engine(
    bars,
    signal,
    dist,
    *,
    signal_bar_fill=None,
    gap_stop_at_stop=False,
    target_first=False,
    step=1,
    check_last=True,
    max_bars=12,
):
    """A small third-party style engine with switchable fill bugs.

    With every switch off it fills at the next open, fills gapped stops at the
    open, books the stop on a bar that touches both levels and checks every
    bar, so its trades must pass the audit.
    """
    o, h, lo, c = bars.open, bars.high, bars.low, bars.close
    n = len(bars)
    out = []
    i = 0
    while i < n - 2:
        side = int(signal[i])
        if side == 0:
            i += 1
            continue
        if signal_bar_fill == "close":
            e, entry = i, float(c[i])
        elif signal_bar_fill == "mid":
            e, entry = i, float((h[i] + lo[i]) / 2)
        else:
            e, entry = i + 1, float(o[i + 1])
        first = i + 1
        stop, target = entry - side * dist[i], entry + side * 1.2 * dist[i]
        last = min(n - 1, first + max_bars - 1)
        exit_j, reason, price = last, "time", float(c[last])
        for j in range(first, last + 1):
            if (j - first) % step and j != last:
                continue
            if j == last and not check_last:
                break
            if side > 0:
                hit_s, hit_t = lo[j] <= stop, h[j] >= target
            else:
                hit_s, hit_t = h[j] >= stop, lo[j] <= target
            if hit_s and not (hit_t and target_first):
                gapped = side * (o[j] - stop) < 0
                price = stop if (gap_stop_at_stop or not gapped) else float(o[j])
                exit_j, reason = j, "stop"
                break
            if hit_t:
                exit_j, reason, price = j, "target", target
                break
        out.append(
            dict(
                side=side,
                signal_index=i,
                entry_index=e,
                exit_index=exit_j,
                entry_price=entry,
                exit_price=price,
                stop=stop,
                target=target,
                reason=reason,
            )
        )
        i = exit_j + 1
    return out


@pytest.fixture(scope="module")
def gappy():
    bars = random_walk_bars(4_000, vol=0.004, gap_vol=0.003, seed=21)
    rng = np.random.default_rng(5)
    signal = rng.choice([-1, 0, 0, 1], size=len(bars))
    return bars, signal, 0.6 * 0.004 * bars.close


# ------------------------------------------------------------ negative controls


@pytest.mark.parametrize(
    "seed,gap_vol,max_bars",
    [(1, 0.0, None), (2, 0.003, None), (3, 0.003, 15), (4, 0.0, 5)],
)
def test_engine_trades_on_a_random_walk_have_no_fails(seed, gap_vol, max_bars):
    bars, res = engine_trades(seed, gap_vol=gap_vol, max_bars=max_bars)
    assert len(res.trades) > 500
    audit = audit_fills(bars, res)
    assert audit.passed, str(audit)
    assert audit.trades_with_fail == 0
    assert audit.fails == ()
    # The engine books the stop on ambiguous bars, so nothing is optimistic.
    assert "ambiguous_resolved_optimistically" not in audit.counts
    assert audit.ambiguous_exits > 0
    assert audit.ambiguous_target == 0
    assert audit.n_rescored == len(res.trades)
    assert audit.checked["signal_bar"] == len(res.trades)
    # Booked R reproduces the engine's R, and the engine is the worst case
    # except where it books a stop on a bar that opened beyond the target.
    np.testing.assert_allclose(audit.booked_r, res.r, rtol=0, atol=1e-12)
    booked, worst, best = (np.array(x) for x in (audit.booked_r, audit.worst_r, audit.best_r))
    assert np.all(worst >= booked - 1e-12)
    assert np.all(best >= worst - 1e-12)
    assert audit.best_expectancy_r > audit.worst_expectancy_r
    if gap_vol == 0.0:
        np.testing.assert_allclose(worst, booked, rtol=0, atol=1e-12)


def test_engine_trades_with_entry_delay_pass():
    bars = random_walk_bars(4_000, vol=0.004, gap_vol=0.002, seed=8)
    rng = np.random.default_rng(8)
    side = rng.choice([-1, 0, 0, 0, 1], size=len(bars))
    d = 0.004 * bars.close
    res = run_signals(
        bars, side, bars.close - side * d, bars.close + side * d, costs=COSTS, entry_delay=2
    )
    audit = audit_fills(bars, res)
    assert audit.passed
    assert len(res.trades) > 200


def test_honest_third_party_log_passes(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist)
    assert len(records) > 200
    audit = audit_fills(bars, records)
    assert audit.passed, str(audit)
    assert audit.issues == ()


def test_wrong_costs_are_not_silently_accepted():
    # Fills that include a spread audited as if they had none: entries land
    # above some entry bars' highs. The audit must not quietly pass them.
    bars, res = engine_trades(5, costs=Costs(half_spread=0.5))
    audit = audit_fills(bars, res, costs=Costs())
    assert not audit.passed
    assert "entry_outside_bar" in audit.counts
    assert audit_fills(bars, res).passed


def test_a_real_edge_with_honest_fills_passes():
    # The audit judges fills, not edges: planted-signal trades filled by the
    # engine pass, and the edge survives the worst case.
    bars, signal = planted_signal_bars(20_000, edge=0.5, signal_prob=0.1, vol=0.004, seed=3)
    d = 0.004 * bars.close
    res = run_signals(bars, signal, bars.close - signal * d, bars.close + signal * d, max_bars=1)
    audit = audit_fills(bars, res)
    assert audit.passed
    assert audit.worst_expectancy_r > 0.05


def test_lookahead_in_the_signal_is_not_a_fill_problem():
    # A cheating oracle with honest fills passes this audit: lookahead in the
    # signal is the lookahead module's job, which is why this audit says
    # nothing about the edge itself.
    bars = random_walk_bars(5_000, vol=0.004, seed=4)
    signal = oracle_signal(bars, horizon=2)
    d = 0.004 * bars.close
    res = run_signals(bars, signal, bars.close - signal * d, bars.close + signal * d, costs=COSTS)
    audit = audit_fills(bars, res)
    assert audit.passed
    assert audit.booked_expectancy_r > 0.2


def test_empty_log():
    bars = bars_from_rows([SIGNAL, ENTRY])
    audit = audit_fills(bars, [])
    assert audit.passed and audit.n_trades == 0
    assert audit.fail_share is None
    assert audit.booked_expectancy_r is None
    assert "no trade could be rescored" in str(audit)


# ------------------------------------------------- positive controls, synthetic


def test_buggy_engine_filling_inside_the_signal_bar_is_caught(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, signal_bar_fill="mid")
    audit = audit_fills(bars, records)
    assert audit.counts["phantom_signal_bar_fill"] == len(records)
    assert audit.fail_share == 1.0


def test_engine_filling_at_the_signal_close_is_warned_not_failed(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, signal_bar_fill="close")
    audit = audit_fills(bars, records)
    assert audit.counts == {"filled_at_signal_close": len(records)}
    assert audit.passed


def test_buggy_engine_filling_gapped_stops_at_the_stop_is_caught(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, gap_stop_at_stop=True)
    audit = audit_fills(bars, records)
    assert audit.counts["gap_stop_fill_too_good"] > 10
    assert set(audit.counts) <= {"gap_stop_fill_too_good", "phantom_exit_fill"}
    # The engine books the stop on ambiguous bars, like the worst case, yet its
    # booked expectancy beats the worst case: the gap losses went missing.
    assert audit.booked_expectancy_r > audit.worst_expectancy_r


def test_buggy_engine_booking_targets_on_ambiguous_bars_is_caught(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, target_first=True)
    audit = audit_fills(bars, records)
    assert audit.counts["ambiguous_resolved_optimistically"] > 10
    assert audit.counts["ambiguous_resolved_optimistically"] == audit.ambiguous_target
    assert audit.booked_expectancy_r > audit.worst_expectancy_r
    assert audit.booked_expectancy_r >= audit.best_expectancy_r - 1e-12


def test_target_first_without_gaps_is_only_a_warning():
    bars = random_walk_bars(4_000, vol=0.004, seed=22)
    signal = np.random.default_rng(6).choice([-1, 0, 0, 1], size=len(bars))
    records = naive_engine(bars, signal, 0.6 * 0.004 * bars.close, target_first=True)
    audit = audit_fills(bars, records)
    assert audit.passed
    assert audit.counts["ambiguous_resolved_optimistically"] == audit.ambiguous_target > 10
    assert audit.best_expectancy_r == pytest.approx(audit.booked_expectancy_r, abs=1e-12)


def test_engine_that_skips_bars_is_caught(gappy):
    # Regression: an audit that only inspected the exit bar passed this.
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, step=2)
    audit = audit_fills(bars, records)
    assert audit.counts["missed_earlier_touch"] > 10
    for issue in audit.issues:
        rec = records[issue.trade]
        assert rec["entry_index"] <= issue.bar < rec["exit_index"]


def test_engine_that_never_checks_the_last_bar_is_caught(gappy):
    # Wide brackets and a short holding limit, so many trades time out.
    bars, signal, dist = gappy
    assert audit_fills(bars, naive_engine(bars, signal, 4 * dist, max_bars=4)).passed
    records = naive_engine(bars, signal, 4 * dist, check_last=False, max_bars=4)
    audit = audit_fills(bars, records)
    assert audit.counts["missed_earlier_touch"] > 10
    assert all(
        i.bar == records[i.trade]["exit_index"] and "close" in i.message for i in audit.issues
    )


def test_engine_modes_land_on_the_audit_bounds():
    # The engine settles a both-touched bar by its open when it can, so its two
    # same_bar modes differ only on truly ambiguous bars: stop-first booking is
    # the audit's worst case and target-first booking its best case.
    bars, optimistic = engine_trades(2, gap_vol=0.006, same_bar="target", costs=Costs())
    audit = audit_fills(bars, optimistic)
    assert audit.counts["ambiguous_resolved_optimistically"] > 10
    assert audit.passed
    assert audit.booked_expectancy_r == pytest.approx(audit.best_expectancy_r, abs=1e-12)
    _, pessimistic = engine_trades(2, gap_vol=0.006, same_bar="stop", costs=Costs())
    audit_p = audit_fills(bars, pessimistic)
    assert audit_p.issues == ()
    assert audit_p.booked_expectancy_r == pytest.approx(audit.worst_expectancy_r, abs=1e-12)


# ------------------------------------------------------ hand-built positive cases


def test_signal_bar_fill_at_an_intrabar_price():
    bars = bars_from_rows([(100.0, 101.0, 99.0, 100.5), QUIET, QUIET])
    rec = long_trade(2, 100.4, "time", entry_index=0, entry_price=99.5)
    audit = audit_fills(bars, [rec])
    assert codes(audit) == ["phantom_signal_bar_fill"]
    assert audit.issues[0].severity == "fail" and audit.issues[0].bar == 0
    assert not audit.passed and audit.fail_share == 1.0


def test_signal_bar_fill_at_the_close_after_costs():
    bars = bars_from_rows([(100.0, 101.0, 99.0, 100.5), QUIET, QUIET])
    costs = Costs(half_spread=0.1, slippage=0.02)
    rec = long_trade(2, 100.4 - 0.12, "time", entry_index=0, entry_price=100.62)
    audit = audit_fills(bars, [rec], costs=costs)
    assert codes(audit) == ["filled_at_signal_close"]
    assert audit.passed


def test_entry_before_signal():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET])
    audit = audit_fills(bars, [long_trade(2, 100.4, "time", signal_index=1, entry_index=0)])
    assert codes(audit) == ["entry_before_signal"]


def test_exit_before_entry():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET])
    audit = audit_fills(bars, [long_trade(1, 100.2, "time", entry_index=2, entry_price=100.2)])
    assert codes(audit) == ["exit_before_entry"]
    assert audit.unscored["exit_before_entry"] == 1
    assert audit.n_rescored == 0


@pytest.mark.parametrize("price,better", [(99.4, True), (100.9, False)])
def test_entry_outside_the_bar(price, better):
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET, QUIET])
    audit = audit_fills(bars, [long_trade(3, 100.4, "time", entry_price=price)])
    assert codes(audit) == ["entry_outside_bar"]
    assert ("better" if better else "worse") in audit.issues[0].message
    assert audit.issues[0].bar == 1


def test_short_entry_outside_the_bar():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET, QUIET])
    rec = long_trade(3, 100.4, "time", side=-1, entry_price=100.7, stop=101.5, target=98.0)
    assert codes(audit_fills(bars, [rec])) == ["entry_outside_bar"]


def test_target_that_was_never_reached_is_a_phantom_exit():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET, (100.4, 101.5, 100.2, 101.2)])
    audit = audit_fills(bars, [long_trade(3, 102.0, "target")])
    assert codes(audit) == ["phantom_exit_fill"]
    # The bars do not confirm the target: both replays close at bar 3's close.
    assert audit.worst_r == audit.best_r == (pytest.approx(1.2),)
    assert audit.booked_r == (pytest.approx(2.0),)


def test_exit_worse_than_the_bar_is_only_a_warning():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET, QUIET])
    audit = audit_fills(bars, [long_trade(3, 99.5, "time")])
    assert codes(audit) == ["exit_outside_bar"]
    assert audit.passed


def test_target_filled_at_target_after_a_favourable_gap_is_fine():
    bars = bars_from_rows([SIGNAL, ENTRY, (103.0, 103.5, 102.5, 103.2)])
    assert audit_fills(bars, [long_trade(2, 102.0, "target")]).issues == ()
    # Without a reason, an exit at the target level is read as a target exit.
    assert audit_fills(bars, [long_trade(2, 102.0, None)]).issues == ()


def test_stop_filled_at_the_stop_on_a_gap_bar():
    # Bar 2 opens at 98.5, through the stop at 99, then trades back up to
    # 99.4. The stop price traded, but only after the stop had filled at 98.5.
    bars = bars_from_rows([SIGNAL, ENTRY, (98.5, 99.4, 98.2, 99.2)])
    audit = audit_fills(bars, [long_trade(2, 99.0, "stop")])
    assert codes(audit) == ["gap_stop_fill_too_good"]
    assert audit.issues[0].bar == 2
    assert audit.booked_r == (pytest.approx(-1.0),)
    assert audit.worst_r == audit.best_r == (pytest.approx(-1.5),)
    # An exit at the stop level counts as a stop exit when the reason is unknown.
    assert codes(audit_fills(bars, [long_trade(2, 99.0, None)])) == ["gap_stop_fill_too_good"]
    # Filled at the open, it is fine.
    assert audit_fills(bars, [long_trade(2, 98.5, "stop")]).issues == ()


def test_stop_at_the_stop_when_the_whole_bar_gapped_past_it():
    bars = bars_from_rows([SIGNAL, ENTRY, (98.0, 98.5, 97.5, 98.2)])
    audit = audit_fills(bars, [long_trade(2, 99.0, "stop")])
    assert sorted(codes(audit)) == ["gap_stop_fill_too_good", "phantom_exit_fill"]
    assert audit.trades_with_fail == 1


def test_short_stop_filled_at_the_stop_on_a_gap_bar():
    bars = bars_from_rows([SIGNAL, ENTRY, (101.5, 101.8, 100.7, 101.0)])
    rec = long_trade(2, 101.0, "stop", side=-1, stop=101.0, target=98.0)
    audit = audit_fills(bars, [rec])
    assert codes(audit) == ["gap_stop_fill_too_good"]
    assert audit.worst_r == (pytest.approx(-1.5),)


def test_target_booked_on_an_ambiguous_bar():
    bars = bars_from_rows([SIGNAL, ENTRY, (100.2, 102.5, 98.5, 100.0)])
    audit = audit_fills(bars, [long_trade(2, 102.0, "target")])
    assert codes(audit) == ["ambiguous_resolved_optimistically"]
    assert audit.passed
    assert (audit.ambiguous_exits, audit.ambiguous_target) == (1, 1)
    assert audit.booked_r == (pytest.approx(2.0),)
    assert audit.worst_r == (pytest.approx(-1.0),)
    assert audit.best_r == (pytest.approx(2.0),)
    # Booked as the stop: fine, but still counted.
    stop_audit = audit_fills(bars, [long_trade(2, 99.0, "stop")])
    assert stop_audit.issues == ()
    assert (stop_audit.ambiguous_exits, stop_audit.ambiguous_target) == (1, 0)


def test_short_target_booked_on_an_ambiguous_bar():
    bars = bars_from_rows([SIGNAL, ENTRY, (100.2, 101.5, 97.5, 100.0)])
    rec = long_trade(2, 98.0, "target", side=-1, stop=101.0, target=98.0)
    audit = audit_fills(bars, [rec])
    assert codes(audit) == ["ambiguous_resolved_optimistically"]
    assert audit.worst_r == (pytest.approx(-1.0),)
    assert audit.best_r == (pytest.approx(2.0),)


def test_short_exit_logged_after_the_stop_was_hit():
    rows = [SIGNAL, ENTRY, (100.2, 101.3, 100.0, 100.5), (100.5, 100.6, 97.8, 98.0)]
    rec = long_trade(3, 98.0, "target", side=-1, stop=101.0, target=98.0)
    audit = audit_fills(bars_from_rows(rows), [rec])
    assert codes(audit) == ["missed_earlier_touch"]
    assert "high 101.3" in audit.issues[0].message


def test_unknown_reason_exit_at_the_target_pays_no_slippage():
    costs = Costs(half_spread=0.1, slippage=0.5)
    bars = bars_from_rows([SIGNAL, ENTRY, (100.2, 102.1, 100.0, 101.0)])
    rec = long_trade(2, 101.9, None, entry_price=100.6)
    audit = audit_fills(bars, [rec], costs=costs)
    assert audit.issues == ()
    assert audit.booked_r == audit.worst_r == audit.best_r


def test_target_booked_on_a_bar_that_opened_through_the_stop():
    bars = bars_from_rows([SIGNAL, ENTRY, (98.5, 102.5, 98.2, 102.2)])
    audit = audit_fills(bars, [long_trade(2, 102.0, "target")])
    assert codes(audit) == ["missed_earlier_touch"]
    assert "opened" in audit.issues[0].message
    assert audit.ambiguous_exits == 0
    assert audit.worst_r == audit.best_r == (pytest.approx(-1.5),)


def test_exit_logged_after_a_bar_that_already_hit_the_stop():
    rows = [SIGNAL, ENTRY, (100.2, 100.4, 98.7, 99.5), (99.5, 102.5, 99.4, 102.3)]
    audit = audit_fills(bars_from_rows(rows), [long_trade(3, 102.0, "target")])
    assert codes(audit) == ["missed_earlier_touch"]
    assert audit.issues[0].bar == 2
    assert "stop 99" in audit.issues[0].message
    assert audit.booked_r == (pytest.approx(2.0),)
    assert audit.worst_r == audit.best_r == (pytest.approx(-1.0),)


def test_missed_target_touch_is_also_caught():
    rows = [SIGNAL, ENTRY, (100.2, 102.4, 100.0, 101.0), (101.0, 101.2, 98.5, 98.8)]
    audit = audit_fills(bars_from_rows(rows), [long_trade(3, 99.0, "stop")])
    assert codes(audit) == ["missed_earlier_touch"]
    assert "target" in audit.issues[0].message
    assert audit.worst_r == (pytest.approx(2.0),)


def test_touch_on_the_entry_bar_is_checked():
    # Regression: the entry bar (entered at its open) must be replayed too.
    rows = [SIGNAL, (100.0, 100.5, 98.7, 99.8), QUIET, (100.4, 102.5, 100.3, 102.3)]
    audit = audit_fills(bars_from_rows(rows), [long_trade(3, 102.0, "target")])
    assert codes(audit) == ["missed_earlier_touch"]
    assert audit.issues[0].bar == 1


@pytest.mark.parametrize("reason", ["time", "end"])
def test_close_exit_on_a_bar_that_hit_the_stop(reason):
    rows = [SIGNAL, ENTRY, (100.2, 100.4, 98.7, 99.5)]
    audit = audit_fills(bars_from_rows(rows), [long_trade(2, 99.5, reason)])
    assert codes(audit) == ["missed_earlier_touch"]
    assert audit.issues[0].bar == 2
    assert audit.worst_r == (pytest.approx(-1.0),)


def test_exit_at_the_open_ignores_the_rest_of_the_exit_bar():
    rows = [SIGNAL, ENTRY, (100.2, 100.4, 98.7, 99.5)]
    audit = audit_fills(bars_from_rows(rows), [long_trade(2, 100.2, "other")])
    assert audit.issues == ()
    assert audit.worst_r == audit.best_r == (pytest.approx(0.2),)


def test_level_touched_exactly_is_not_flagged():
    rows = [SIGNAL, (100.0, 100.5, 99.0, 99.8), QUIET, (100.4, 102.5, 100.3, 102.3)]
    assert audit_fills(bars_from_rows(rows), [long_trade(3, 102.0, "target")]).issues == ()


def test_intrabar_entry_touch_that_may_precede_the_entry_is_a_warning():
    # Buy-stop entry at 100.3, above the open: the dip to 98.8 may have come first.
    rows = [SIGNAL, (100.0, 100.6, 98.8, 100.4), QUIET, (100.4, 102.5, 100.3, 102.3)]
    rec = long_trade(3, 102.0, "target", entry_price=100.3)
    audit = audit_fills(bars_from_rows(rows), [rec])
    assert codes(audit) == ["possible_entry_bar_touch"]
    assert audit.passed
    assert audit.worst_r == (pytest.approx(-1.0),)
    assert audit.best_r == (pytest.approx(1.7 / 1.3),)


def test_intrabar_entry_touch_that_must_follow_the_entry_is_a_fail():
    # Limit entry at 99.5, below the open: falling from 100 to 98.8 passes
    # 99.5 first, so the stop was hit after the fill.
    rows = [SIGNAL, (100.0, 100.4, 98.8, 99.6), QUIET, (100.4, 102.5, 100.3, 102.3)]
    rec = long_trade(3, 102.0, "target", entry_price=99.5)
    audit = audit_fills(bars_from_rows(rows), [rec])
    assert codes(audit) == ["missed_earlier_touch"]
    assert audit.issues[0].bar == 1
    assert audit.worst_r == audit.best_r == (pytest.approx(-1.0),)


# ------------------------------------------------------------- arithmetic


def test_worst_and_best_case_with_costs_by_hand():
    costs = Costs(half_spread=0.1, fee_rate=0.001, slippage=0.05)
    bars = bars_from_rows([SIGNAL, ENTRY, (100.2, 102.5, 98.5, 100.0)])
    entry, target_fill, stop_fill = 100.15, 101.9, 98.85
    rec = long_trade(2, target_fill, "target", entry_price=entry)
    audit = audit_fills(bars, [rec], costs=costs)
    assert codes(audit) == ["ambiguous_resolved_optimistically"]
    risk = entry - 99.0
    win = (target_fill - entry - 0.001 * (entry + target_fill)) / risk
    loss = (stop_fill - entry - 0.001 * (entry + stop_fill)) / risk
    assert audit.booked_expectancy_r == pytest.approx(win)
    assert audit.best_expectancy_r == pytest.approx(win)
    assert audit.worst_expectancy_r == pytest.approx(loss)


def test_gap_stop_rescored_at_the_open_with_costs():
    costs = Costs(half_spread=0.1, slippage=0.05)
    bars = bars_from_rows([SIGNAL, ENTRY, (98.5, 99.4, 98.2, 99.2)])
    rec = long_trade(2, 99.0 - 0.15, "stop", entry_price=100.15)
    audit = audit_fills(bars, [rec], costs=costs)
    assert codes(audit) == ["gap_stop_fill_too_good"]
    assert audit.worst_r == (pytest.approx((98.5 - 0.15 - 100.15) / 1.15),)


def test_expectancy_and_share_over_several_trades():
    rows = [SIGNAL, ENTRY, (100.2, 102.5, 98.5, 100.0), QUIET, (98.5, 99.4, 98.2, 99.2)]
    bars = bars_from_rows(rows)
    records = [
        long_trade(2, 102.0, "target"),  # ambiguous, booked as target: +2 / -1 / +2
        dict(long_trade(4, 99.0, "stop"), signal_index=2, entry_index=3, entry_price=100.2),
    ]
    audit = audit_fills(bars, records)
    assert codes(audit) == ["ambiguous_resolved_optimistically", "gap_stop_fill_too_good"]
    assert audit.trades_with_fail == 1
    assert audit.fail_share == 0.5
    second_booked = (99.0 - 100.2) / 1.2
    second_true = (98.5 - 100.2) / 1.2
    assert audit.booked_expectancy_r == pytest.approx((2.0 + second_booked) / 2)
    assert audit.worst_expectancy_r == pytest.approx((-1.0 + second_true) / 2)
    assert audit.best_expectancy_r == pytest.approx((2.0 + second_true) / 2)
    assert [i.trade for i in audit.issues_for(1)] == [1]


def test_trades_without_levels_are_audited_but_not_rescored():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET])
    rec = long_trade(2, 100.4, "time", stop=None, target=None)
    audit = audit_fills(bars, [rec])
    assert audit.passed
    assert audit.unscored["no_levels"] == 1
    assert audit.checked["missed_touch"] == 0
    bad = long_trade(2, 100.4, "time", stop=101.0)  # stop above a long's entry
    assert audit_fills(bars, [bad]).unscored["invalid_geometry"] == 1


def test_tolerance_absorbs_rounded_prices():
    bars = bars_from_rows([SIGNAL, (100.123456, 100.6, 100.123456, 100.2), QUIET])
    rec = long_trade(2, 100.4, "time", entry_price=100.12)
    assert codes(audit_fills(bars, [rec])) == ["entry_outside_bar"]
    assert audit_fills(bars, [rec], tolerance=0.005).issues == ()


# ------------------------------------------------------------ input handling


def test_timestamp_mapping_matches_indices():
    bars, res = engine_trades(7, n=1_500)
    t = bars.time
    as_strings, as_datetimes = [], []
    for tr in res.trades:
        base = dict(
            side="BUY" if tr.side > 0 else "SELL",
            entry_price=tr.entry_price,
            exit_price=tr.exit_price,
            stop_loss=tr.stop,
            take_profit=tr.target,
            exit_reason=tr.reason.upper(),
        )
        as_strings.append(
            dict(
                base,
                signal_time=str(t[tr.signal_index]),
                entry_time=str(t[tr.entry_index]).replace("T", " "),
                exit_time=str(t[tr.exit_index]),
            )
        )
        as_datetimes.append(
            dict(
                base,
                entry_time=t[tr.entry_index].astype(datetime.datetime),
                exit_time=t[tr.exit_index],
                signal_index=tr.signal_index,
            )
        )
    expected = TradeLog.from_trades(res.trades, bars)
    assert TradeLog.from_records(as_strings, bars) == expected
    assert TradeLog.from_records(as_datetimes, bars) == expected
    assert audit_fills(bars, as_strings, costs=COSTS).passed


def test_string_and_numeric_bar_times():
    rows = [SIGNAL, ENTRY, QUIET]
    o, h, lo, c = zip(*rows, strict=True)
    rec = dict(side=1, entry_price=100.0, exit_price=100.4, entry_time=None, exit_time=None)
    strings = Bars(o, h, lo, c, time=["2024-01-01", "2024-01-02", "2024-01-03"])
    log = TradeLog.from_records(
        [dict(rec, entry_time="2024-01-02", exit_time="2024-01-03")], strings
    )
    assert (log[0].entry_index, log[0].exit_index) == (1, 2)
    numbers = Bars(o, h, lo, c, time=[10, 20, 30])
    log = TradeLog.from_records([dict(rec, entry_time="20", exit_time=30)], numbers)
    assert (log[0].entry_index, log[0].exit_index) == (1, 2)


def test_timestamp_errors_are_clear():
    bars = random_walk_bars(5, seed=0)
    rec = dict(side=1, entry_price=100.0, exit_price=100.0, exit_index=3)
    with pytest.raises(ValueError, match="no bar has timestamp"):
        TradeLog.from_records([dict(rec, entry_time="1999-01-01T00:00")], bars)
    with pytest.raises(ValueError, match="cannot read"):
        TradeLog.from_records([dict(rec, entry_time="not a time")], bars)
    with pytest.raises(ValueError, match="disagrees"):
        TradeLog.from_records([dict(rec, entry_index=2, entry_time=str(bars.time[1]))], bars)
    untimed = bars_from_rows([SIGNAL, ENTRY])
    with pytest.raises(ValueError, match="no timestamps"):
        TradeLog.from_records([dict(rec, exit_index=1, entry_time="2020-01-01")], untimed)


def test_csv_round_trip(tmp_path):
    bars, res = engine_trades(9, n=1_500, gap_vol=0.002, max_bars=20)
    log = TradeLog.from_trades(res.trades, bars)
    plain, timed = tmp_path / "plain.csv", tmp_path / "timed.csv"
    log.to_csv(plain)
    log.to_csv(timed, bars)
    assert TradeLog.from_csv(plain, bars) == log
    assert TradeLog.from_csv(timed, bars) == log
    assert "entry_time" in timed.read_text().splitlines()[0]
    assert audit_fills(bars, TradeLog.from_csv(timed, bars), costs=COSTS) == audit_fills(bars, res)


def test_csv_with_other_column_names(tmp_path):
    bars = bars_from_rows([SIGNAL, ENTRY, (98.5, 99.4, 98.2, 99.2)])
    path = tmp_path / "log.csv"
    path.write_text(
        "Direction,Bar In,Exit_Bar,Fill In,Exit,SL,TP,Exit Reason,signal_bar\n"
        "long,1,2,100.0,99.0,99.0,102.0,Stop Loss,0\n"
        "-1,1,2,100.0,98.5,101.0,98.0,,\n"
    )
    log = TradeLog.from_csv(path, bars, columns={"entry_index": "Bar In", "entry_price": "Fill In"})
    assert log[0] == TradeRow(1, 1, 2, 100.0, 99.0, 0, 99.0, 102.0, "stop")
    assert log[1] == TradeRow(-1, 1, 2, 100.0, 98.5, None, 101.0, 98.0, None)
    audit = audit_fills(bars, log)
    assert codes(audit) == ["gap_stop_fill_too_good"]
    bad = tmp_path / "bad.csv"
    bad.write_text("side,entry_index,exit_index,entry_price,exit_price\n1,1,2,100,abc\n")
    with pytest.raises(ValueError, match="line 2"):
        TradeLog.from_csv(bad, bars)


def test_trade_objects_and_mixed_records():
    bars, res = engine_trades(10, n=800)
    mixed = [t if k % 2 else t.as_dict() for k, t in enumerate(res.trades)]
    assert TradeLog.from_records(mixed, bars) == TradeLog.from_trades(res.trades, bars)
    assert audit_fills(bars, list(res.trades), costs=COSTS).passed


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(reason="trailing_exit_v2"), "unknown exit reason"),
        (dict(exit_index=5), "past the last bar"),
        (dict(entry_index=-1), ">= 0"),
        (dict(entry_index=1.5), "whole bar index"),
        (dict(entry_index=True), "bar index"),
        (dict(side="hold"), "unknown side"),
        (dict(entry_price="abc"), "number"),
        (dict(exit_price=float("inf")), "finite"),
    ],
)
def test_bad_records_raise(change, match):
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET])
    rec = dict(long_trade(2, 100.4, "time"), **change)
    with pytest.raises(ValueError, match=match):
        audit_fills(bars, [rec])


def test_missing_required_fields_raise():
    bars = bars_from_rows([SIGNAL, ENTRY, QUIET])
    rec = long_trade(2, 100.4, "time")
    del rec["exit_price"]
    with pytest.raises(ValueError, match="missing exit_price"):
        TradeLog.from_records([rec], bars)
    rec = long_trade(2, 100.4, "time")
    del rec["entry_index"]
    with pytest.raises(ValueError, match="entry_index or entry_time"):
        TradeLog.from_records([rec], bars)
    with pytest.raises(ValueError, match="unknown standard field"):
        TradeLog.from_records([], bars, columns={"price": "x"})
    with pytest.raises(TypeError):
        TradeLog.from_records([(1, 2, 3)], bars)


def test_trade_log_checks_its_rows():
    row = TradeRow(1, 1, 2, 100.0, 100.4)
    assert len(TradeLog((row,), 3)) == 1
    with pytest.raises(ValueError, match="past the last bar"):
        TradeLog((row,), 2)
    with pytest.raises(TypeError):
        TradeLog(({"side": 1},), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="past the last bar"):
        TradeLog((TradeRow(1, 1, 2, 100.0, 100.4, signal_index=7),), 3)


def test_options_and_bar_count_are_validated():
    bars, res = engine_trades(11, n=500)
    with pytest.raises(ValueError, match="tolerance"):
        audit_fills(bars, res, tolerance=-1.0)
    with pytest.raises(ValueError, match="bars"):
        audit_fills(bars.prefix(400), res)
    log = TradeLog.from_trades(res.trades, bars)
    with pytest.raises(ValueError, match="bars"):
        audit_fills(bars.prefix(400), log)


def test_result_costs_are_used_by_default():
    bars, res = engine_trades(12, n=1_500, costs=Costs(half_spread=0.3, slippage=0.1))
    assert audit_fills(bars, res).passed
    assert not audit_fills(bars, list(res.trades)).passed  # zero costs assumed


# ----------------------------------------------------------------- reporting


def test_report_is_readable_and_serialisable():
    bars = bars_from_rows([SIGNAL, ENTRY, (98.5, 99.4, 98.2, 99.2)])
    audit = audit_fills(bars, [long_trade(2, 99.0, "stop")])
    assert isinstance(audit, FillAudit)
    text = str(audit)
    assert "FAILED" in text and "gap_stop_fill_too_good" in text and "heuristic" in text
    assert "worst case" in text
    data = audit.as_dict()
    json.dumps(data)
    assert data["counts"] == {"gap_stop_fill_too_good": 1}
    assert data["issues"][0]["bar"] == 2
    issue = audit.issues[0]
    assert isinstance(issue, Issue)
    assert str(issue).startswith("trade 0, bar 2 [fail] gap_stop_fill_too_good")
    assert issue.as_dict()["severity"] == "fail"
    passed = audit_fills(bars, [long_trade(2, 98.5, "stop")])
    assert "PASSED" in str(passed)
    with pytest.raises(AttributeError):
        audit.passed = True  # type: ignore[misc]


def test_every_issue_uses_a_documented_code_and_severity(gappy):
    bars, signal, dist = gappy
    records = naive_engine(bars, signal, dist, gap_stop_at_stop=True, target_first=True, step=2)
    audit = audit_fills(bars, records)
    assert audit.issues
    for issue in audit.issues:
        assert CODES[issue.code][0] == issue.severity
    assert sum(audit.counts.values()) == len(audit.issues)
    assert len(audit.fails) + len(audit.warnings) == len(audit.issues)


def test_trade_row_from_engine_trade():
    t = Trade(1, 0, 1, 2, 100.0, 100.0, 102.0, 99.0, 102.0, "target", 2.0, 1.0, 2.0, 0.0)
    row = TradeRow.from_trade(t)
    assert row.as_dict()["reason"] == "target"
    assert row.signal_index == 0


# ------------------------------------------------------- adversarial review


def test_stop_filled_better_than_the_stop_on_a_bar_that_did_not_gap_fails():
    bars = Bars([100, 100, 100], [101, 101, 101], [99, 99, 90], [100, 100, 95])
    rec = dict(side=1, entry_index=1, exit_index=2, entry_price=100, exit_price=99)
    audit = audit_fills(bars, [dict(rec, stop=95, target=110, reason="stop")])
    assert "stop_fill_too_good" in codes(audit)
    assert not audit.passed
    short = Bars([100, 100, 100], [101, 101, 110], [99, 99, 99], [100, 100, 105])
    rec = dict(side=-1, entry_index=1, exit_index=2, entry_price=100, exit_price=101)
    assert "stop_fill_too_good" in codes(
        audit_fills(short, [dict(rec, stop=105, target=90, reason="stop")])
    )
    # A fill at the stop, or worse, is fine.
    rec = dict(side=1, entry_index=1, exit_index=2, entry_price=100, exit_price=94.5)
    assert codes(audit_fills(bars, [dict(rec, stop=95, target=110, reason="stop")])) == []
