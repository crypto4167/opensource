import numpy as np
import pytest

from backtest_skeptic.engine import resolve_many
from backtest_skeptic.synthetic import oracle_signal, planted_signal_bars, random_walk_bars


@pytest.mark.parametrize("vol", [0.005, 0.02, 0.05])
def test_random_walk_is_a_martingale_in_price(vol):
    n_paths, n = 3000, 40
    ends = np.array(
        [random_walk_bars(n, vol=vol, seed=s, start_time=None).close[-1] for s in range(n_paths)]
    )
    ratio = ends / 100.0
    se = ratio.std(ddof=1) / np.sqrt(n_paths)
    assert abs(ratio.mean() - 1.0) < 4 * se


def test_drift_sets_the_expected_growth():
    drift, n = 0.002, 50
    ends = np.array(
        [
            random_walk_bars(n, vol=0.01, drift=drift, seed=s, start_time=None).close[-1]
            for s in range(3000)
        ]
    )
    ratio = ends / 100.0
    se = ratio.std(ddof=1) / np.sqrt(len(ratio))
    assert abs(ratio.mean() - np.exp(drift * n)) < 4 * se


def test_bars_are_consistent_and_continuous_across_bars():
    bars = random_walk_bars(5000, vol=0.02, seed=3)
    assert np.all(bars.high >= np.maximum(bars.open, bars.close))
    assert np.all(bars.low <= np.minimum(bars.open, bars.close))
    np.testing.assert_allclose(bars.open[1:], bars.close[:-1], rtol=1e-12)
    gapped = random_walk_bars(5000, vol=0.02, gap_vol=0.01, seed=3)
    assert np.abs(np.log(gapped.open[1:] / gapped.close[:-1])).max() > 0.01


@pytest.mark.parametrize("vol", [0.005, 0.02])
@pytest.mark.parametrize(("ka", "kb"), [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (0.5, 2.0), (3.0, 1.0)])
def test_bracket_hits_target_with_probability_a_over_a_plus_b(vol, ka, kb):
    """On a price martingale a bracket's target is hit with probability a / (a + b).

    Bars touching both levels are booked as stops or as targets, so the true
    rate must lie between the two bookings, within Monte Carlo error.
    """
    bars = random_walk_bars(300_000, vol=vol, seed=11, start_time=None)
    sig = np.arange(0, 290_000, 15)  # spaced so trades barely overlap
    mid = bars.open[sig + 1]
    stop, target = mid - ka * vol * mid, mid + kb * vol * mid
    side = np.ones(len(sig))
    rates = {}
    for same_bar in ("stop", "target"):
        r, exit_index = resolve_many(bars, sig, side, stop, target, same_bar=same_bar)
        assert not np.isnan(r).any()
        assert np.mean(exit_index == len(bars) - 1) < 1e-3  # nearly all resolve
        rates[same_bar] = np.mean(np.isclose(r, kb / ka))
    p = ka / (ka + kb)
    se = np.sqrt(p * (1 - p) / len(sig))
    assert rates["stop"] <= rates["target"]
    assert rates["stop"] - 4 * se < p < rates["target"] + 4 * se


def test_shorts_on_a_martingale_have_the_same_odds():
    vol, ka, kb = 0.01, 1.0, 2.0
    bars = random_walk_bars(300_000, vol=vol, seed=12, start_time=None)
    sig = np.arange(0, 290_000, 15)
    mid = bars.open[sig + 1]
    r, _ = resolve_many(
        bars, sig, -np.ones(len(sig)), mid + ka * vol * mid, mid - kb * vol * mid, same_bar="target"
    )
    p = ka / (ka + kb)
    se = np.sqrt(p * (1 - p) / len(sig))
    assert abs(np.mean(np.isclose(r, kb / ka)) - p) < 4 * se + 0.01


def test_timestamps_follow_the_interval():
    bars = random_walk_bars(3, seed=0, start_time="2024-01-01T00:00", interval="15min")
    assert bars.time is not None
    assert np.all(np.diff(bars.time) == np.timedelta64(900, "s"))
    assert random_walk_bars(3, seed=0, start_time=None).time is None
    with pytest.raises(ValueError, match="interval"):
        random_walk_bars(3, seed=0, interval="15x")


def test_planted_signal_moves_the_next_bar_and_oracle_reads_the_future():
    bars, signal = planted_signal_bars(100_000, edge=0.3, signal_prob=0.5, seed=4)
    assert signal[-1] == 0
    ret = np.log(bars.close / bars.open)
    on = signal[:-1] != 0
    assert np.mean(signal[:-1][on] * ret[1:][on]) > 0.2 * 0.01
    oracle = oracle_signal(bars, horizon=2)
    assert np.all(oracle[-2:] == 0)
    future = np.sign(bars.close[2:] - bars.close[:-2])
    assert np.all(oracle[:-2] == np.where(future > 0, 1, -1))
