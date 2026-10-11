import math

import numpy as np
import pytest

from backtest_skeptic import Costs, Setup, resolve_bracket, run, run_signals
from backtest_skeptic.synthetic import random_walk_bars
from helpers import flat_then


def once_at(index, setup):
    """A decide() that proposes ``setup`` only at bar ``index``."""

    def decide(i, history):
        return setup if i == index else None

    return decide


def test_long_hits_target_at_target_price():
    bars = flat_then([(100, 101, 99.5, 100.5), (100.5, 102.5, 100.2, 102.0)])
    res = run(bars, once_at(0, (1, 99.0, 102.0)))
    (t,) = res.trades
    assert (t.signal_index, t.entry_index, t.exit_index) == (0, 1, 2)
    assert t.reason == "target"
    assert t.entry_price == 100.0
    assert t.exit_price == 102.0
    assert t.r == pytest.approx(2.0)


def test_stop_gapped_through_fills_at_open_not_stop():
    bars = flat_then([(100, 100.5, 99.5, 100.0), (98.0, 98.5, 97.5, 98.2)])
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0))).trades
    assert t.reason == "stop"
    assert t.exit_price == 98.0
    assert t.r == pytest.approx(-2.0)


def test_short_stop_gap_fills_at_open():
    bars = flat_then([(100, 100.5, 99.5, 100.0), (102.0, 102.5, 101.5, 102.0)])
    (t,) = run(bars, once_at(0, ("SELL", 101.0, 98.0))).trades
    assert t.side == -1
    assert t.reason == "stop"
    assert t.exit_price == 102.0
    assert t.r == pytest.approx(-2.0)


def test_target_gap_is_not_credited_beyond_target():
    bars = flat_then([(100, 100.5, 99.5, 100.0), (104.0, 105.0, 103.5, 104.5)])
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0))).trades
    assert t.reason == "target"
    assert t.exit_price == 102.0


def test_bar_touching_both_levels_books_the_stop_and_flags_it():
    bars = flat_then([(100, 102.5, 98.5, 100.0)])
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0))).trades
    assert t.reason == "stop"
    assert t.ambiguous
    assert t.r == pytest.approx(-1.0)
    (t2,) = run(bars, once_at(0, (1, 99.0, 102.0)), same_bar="target").trades
    assert t2.reason == "target"
    assert t2.ambiguous


def test_last_bar_is_checked_before_end_of_data_exit():
    # The final bar trades through the stop. Booking it at the close (100.5)
    # would turn a loss into a win.
    bars = flat_then([(100, 100.4, 99.6, 100.0), (100.0, 100.8, 98.0, 100.5)])
    (t,) = run(bars, once_at(0, (1, 99.0, 103.0))).trades
    assert t.reason == "stop"
    assert t.exit_index == 2
    assert t.r == pytest.approx(-1.0)


def test_end_of_data_exit_at_last_close():
    bars = flat_then([(100, 100.4, 99.6, 100.0), (100.0, 100.8, 99.5, 100.5)])
    (t,) = run(bars, once_at(0, (1, 99.0, 103.0))).trades
    assert t.reason == "end"
    assert t.exit_price == 100.5
    assert t.r == pytest.approx(0.5)


def test_time_exit_counts_the_entry_bar():
    rows = [(100, 100.4, 99.6, 100.1), (100.1, 100.5, 99.7, 100.2), (100.2, 100.6, 99.8, 100.3)]
    bars = flat_then(rows)
    (t,) = run(bars, once_at(0, (1, 99.0, 103.0)), max_bars=2).trades
    assert t.reason == "time"
    assert t.entry_index == 1
    assert t.exit_index == 2
    assert t.bars_held == 2
    assert t.exit_price == 100.2


def test_entry_open_beyond_stop_is_skipped():
    bars = flat_then([(98.5, 99.0, 98.0, 98.5), (98.5, 99.0, 98.0, 98.5)])
    res = run(bars, once_at(0, (1, 99.0, 102.0)))
    assert res.trades == ()
    assert res.skipped == {"stop_beyond_entry": 1}


def test_entry_open_beyond_target_is_skipped():
    bars = flat_then([(102.5, 103.0, 102.0, 102.5), (102.5, 103.0, 102.0, 102.5)])
    res = run(bars, once_at(0, (1, 99.0, 102.0)))
    assert res.skipped == {"target_beyond_entry": 1}


def test_costs_are_charged_on_both_sides():
    bars = flat_then([(100, 101, 99.5, 100.5), (100.5, 102.5, 100.2, 102.0)])
    costs = Costs(half_spread=0.1, fee_rate=0.001)
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0)), costs=costs).trades
    assert t.entry_price == pytest.approx(100.1)
    assert t.risk == pytest.approx(1.1)
    assert t.exit_price == pytest.approx(101.9)
    assert t.fees == pytest.approx(0.001 * (100.1 + 101.9))
    assert t.r == pytest.approx((101.9 - 100.1 - 0.202) / 1.1)


def test_slippage_applies_to_stop_exit_but_not_target():
    bars = flat_then([(100, 100.5, 98.0, 99.0)])
    costs = Costs(slippage=0.05)
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0)), costs=costs).trades
    assert t.entry_price == pytest.approx(100.05)
    assert t.exit_price == pytest.approx(98.95)
    bars = flat_then([(100, 102.5, 99.5, 102.0)])
    (t,) = run(bars, once_at(0, (1, 99.0, 102.0)), costs=costs).trades
    assert t.exit_price == pytest.approx(102.0)


def test_decide_sees_only_closed_bars_and_cannot_write_them():
    bars = random_walk_bars(50, seed=1)
    seen = []

    def decide(i, history):
        seen.append((i, len(history)))
        with pytest.raises(ValueError):
            history.close[-1] = 0.0
        return None

    run(bars, decide, warmup=5)
    assert all(n == i + 1 for i, n in seen)
    assert seen[0][0] == 5
    assert seen[-1][0] == len(bars) - 2


def test_can_decide_again_on_the_exit_bar():
    bars = flat_then(
        [(100, 102.5, 99.5, 102.0), (102.0, 102.2, 101.8, 102.0), (102.0, 104.5, 101.5, 104.0)]
    )
    calls = []

    def decide(i, history):
        calls.append(i)
        c = history.close[-1]
        return (1, c - 1.0, c + 2.0)

    res = run(bars, decide)
    assert [t.signal_index for t in res.trades][:2] == [0, 1]
    assert res.trades[0].exit_index == 1


def test_invalid_levels_are_counted_not_raised():
    bars = flat_then([(100, 101, 99, 100)] * 3)
    res = run(bars, once_at(0, (1, math.nan, 102.0)))
    assert res.skipped == {"invalid_levels": 1}


def test_bad_return_type_raises():
    bars = flat_then([(100, 101, 99, 100)] * 3)
    with pytest.raises(TypeError):
        run(bars, lambda i, h: [1, 99, 102])


def test_numpy_scalar_levels_are_accepted():
    bars = flat_then([(100, 101, 99.5, 100.5), (100.5, 102.5, 100.2, 102.0)])
    setup = (np.int64(1), np.float32(99.0), np.float32(102.0))
    assert len(run(bars, once_at(0, setup)).trades) == 1


def test_entry_delay_fills_later():
    bars = flat_then([(100, 100.2, 99.8, 100.0), (100.5, 100.7, 100.3, 100.5), (100.5,) * 4])
    (t,) = run(bars, once_at(0, (1, 99.0, 103.0)), entry_delay=1).trades
    assert t.entry_index == 2
    assert t.mid_entry == 100.5


def test_run_and_run_signals_agree():
    bars = random_walk_bars(3_000, seed=7)
    rng = np.random.default_rng(3)
    side = rng.choice([-1, 0, 0, 0, 1], size=len(bars))
    atr = 0.01 * bars.close
    stop = bars.close - side * 1.5 * atr
    target = bars.close + side * 2.0 * atr
    kw = dict(costs=Costs(half_spread=0.01, fee_rate=0.0002), max_bars=40, warmup=10)
    fast = run_signals(bars, side, stop, target, **kw)

    def decide(i, history):
        return None if side[i] == 0 else (int(side[i]), stop[i], target[i])

    slow = run(bars, decide, **kw)
    assert fast.trades == slow.trades
    assert fast.skipped == slow.skipped
    assert len(fast.trades) > 100


def test_run_signals_validates_shapes_and_sides():
    bars = random_walk_bars(10, seed=0)
    with pytest.raises(ValueError):
        run_signals(bars, np.zeros(9), np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError):
        run_signals(bars, np.full(10, 2), np.zeros(10), np.zeros(10))


def test_resolve_bracket_matches_engine():
    bars = flat_then([(100, 101, 99.5, 100.5), (100.5, 102.5, 100.2, 102.0)])
    t = resolve_bracket(bars, 0, Setup(1, 99.0, 102.0))
    assert t is not None and t.reason == "target"
    assert resolve_bracket(bars, 2, Setup(1, 99.0, 102.0)) is None


@pytest.mark.parametrize("stop_mult,target_mult", [(1.0, 1.0), (1.0, 3.0), (2.0, 1.0)])
def test_no_edge_on_a_random_walk(stop_mult, target_mult):
    # On a martingale, a bracket with stop distance a and target distance b
    # hits the target about a/(a+b) of the time and earns about 0R.
    bars = random_walk_bars(60_000, vol=0.004, seed=11)
    side = np.zeros(len(bars))
    side[::7] = 1
    side[3::14] = -1
    d = 0.004 * bars.close
    res = run_signals(
        bars, side, bars.close - side * stop_mult * d, bars.close + side * target_mult * d
    )
    r = res.r
    resolved = np.array([t.reason in ("stop", "target") for t in res.trades])
    hit = np.array([t.reason == "target" for t in res.trades])[resolved]
    expected = stop_mult / (stop_mult + target_mult)
    se = math.sqrt(expected * (1 - expected) / len(hit))
    assert abs(hit.mean() - expected) < 4 * se
    # Ambiguous bars are booked as stops, so the mean can only lean negative.
    assert r.mean() < 4 * r.std(ddof=1) / math.sqrt(len(r))
    assert r.mean() > -0.05


def test_options_are_validated():
    bars = random_walk_bars(10, seed=0)
    with pytest.raises(ValueError):
        run(bars, lambda i, h: None, max_bars=0)
    with pytest.raises(ValueError):
        run(bars, lambda i, h: None, same_bar="first")
    with pytest.raises(ValueError):
        run(bars, lambda i, h: None, entry_delay=-1)
