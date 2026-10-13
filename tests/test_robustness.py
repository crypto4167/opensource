import json
import math
import warnings
from datetime import date, datetime, timedelta, timezone
from itertools import pairwise

import numpy as np
import pytest

from backtest_skeptic import Costs, run, run_signals
from backtest_skeptic.metrics import t_statistic
from backtest_skeptic.robustness import (
    ClusteredT,
    CostRow,
    _breakeven,
    _default_block,
    block_bootstrap,
    clustered_t,
    concentration,
    cost_stress,
    drop_best,
    entry_delay_test,
    period_keys,
    subperiods,
)
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars
from helpers import flat_then


def brackets(bars, side, width=0.02):
    """(side, stop, target) arrays: stop and target ``width`` away from each close."""
    side = np.asarray(side, dtype=np.float64)
    close = bars.close
    return side, close * (1 - side * width), close * (1 + side * width)


def once_at(index, setup):
    def decide(i, history):
        return setup if i == index else None

    return decide


@pytest.fixture(scope="module")
def planted():
    """Bars with a real edge that lasts exactly one bar after each signal."""
    bars, signal = planted_signal_bars(5000, edge=0.5, signal_prob=0.2, seed=7)
    return bars, brackets(bars, signal)


@pytest.fixture(scope="module")
def walk():
    """A driftless random walk traded with coin-flip signals."""
    bars = random_walk_bars(8000, seed=3)
    side = np.random.default_rng(11).choice([-1, 0, 1], p=[0.1, 0.8, 0.1], size=len(bars))
    return bars, brackets(bars, side)


# ---------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------


def test_concentration_hand_computed():
    res = concentration([5.0, 3.0, 1.0, 1.0], top=(1, 2, 0.5))
    assert res.total == 10.0
    assert res.mean == 2.5
    one, two, half = res.rows
    assert (one.k, one.share, one.even_share) == (1, 0.5, 0.25)
    assert one.total_after == 5.0
    assert one.mean_after == pytest.approx(5.0 / 3.0)
    assert one.n_after == 3
    assert (two.k, two.share, two.total_after, two.mean_after) == (2, 0.8, 2.0, 1.0)
    assert half.label == "50%"
    assert half.k == 2
    assert half.share == 0.8
    assert res.top_items == (("#0", 5.0), ("#1", 3.0))
    assert res.defined
    assert not res.fragile


def test_concentration_fraction_rounds_up_without_float_noise():
    # 0.07 * 100 is 7.000000000000001 in floating point; it must count as 7.
    res = concentration(np.arange(100.0) + 1.0, top=(0.07, 0.071))
    assert [row.k for row in res.rows] == [7, 8]


def test_concentration_by_group_ranks_group_sums():
    values = [1.0, 2.0, 3.0, -1.0, 0.5]
    groups = ["a", "a", "b", "b", "c"]
    res = concentration(values, groups=groups, top=(1,))
    assert res.grouped
    assert (res.n_values, res.n_items) == (5, 3)
    (row,) = res.rows
    assert row.share == pytest.approx(3.0 / 5.5)
    assert row.n_after == 3
    assert row.total_after == pytest.approx(2.5)
    assert row.mean_after == pytest.approx(2.5 / 3.0)
    assert res.top_items == (("a", 3.0),)


@pytest.mark.parametrize("values", [[1.0, -2.0], [1.0, -1.0], [-0.5, -0.25]])
def test_concentration_share_undefined_for_non_positive_total(values):
    # Regression: dividing by a negative total printed a large negative share,
    # which read as "not concentrated".
    res = concentration(values, top=(1,))
    assert not res.defined
    assert not res.fragile
    assert all(row.share is None for row in res.rows)
    assert any("undefined" in note for note in res.notes)
    text = str(res)
    assert "share undefined" in text
    assert "-" not in text.split("share")[1].split(";")[0]
    assert res.as_dict()["rows"][0]["share"] is None


def test_concentration_negative_control_evenly_spread_edge():
    x = np.random.default_rng(1).normal(0.5, 1.0, 2000)
    res = concentration(x)
    assert not res.fragile
    assert res.rows[0].share < 0.01
    assert all(row.total_after > 0 for row in res.rows)
    assert res.notes == ()


def test_concentration_positive_control_one_outlier_carries_everything():
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 1.0, 500)
    noise -= noise.mean() + 0.01  # everything else loses 0.01 per value
    x = np.append(noise, 50.0)
    res = concentration(x)
    assert res.rows[0].share > 1.0
    assert res.rows[0].total_after < 0
    assert res.fragile
    assert any("lost money" in note for note in res.notes)
    assert res.top_items[0] == ("#500", 50.0)


def test_concentration_caps_k_with_a_note():
    res = concentration([1.0, 2.0], top=(5,))
    assert res.rows[0].k == 2
    assert res.rows[0].n_after == 0
    assert res.rows[0].mean_after is None
    assert any("capped" in note for note in res.notes)


@pytest.mark.parametrize("top", [(0,), (1.0,), (1.5,), (True,), ("1",), ()])
def test_concentration_rejects_bad_top(top):
    with pytest.raises(ValueError):
        concentration([1.0, 2.0, 3.0], top=top)


def test_concentration_rejects_bad_values_and_groups():
    with pytest.raises(ValueError, match="empty"):
        concentration([])
    with pytest.raises(ValueError, match="finite"):
        concentration([1.0, math.nan])
    with pytest.raises(ValueError, match="one key per value"):
        concentration([1.0, 2.0], groups=["a"])
    with pytest.raises(ValueError, match="missing"):
        concentration([1.0, 2.0], groups=["a", None])
    with pytest.raises(ValueError, match="string"):
        concentration("123")


def test_concentration_accepts_trades_and_backtest_results(planted):
    bars, strat = planted
    res = run_signals(bars, *strat, max_bars=1)
    a = concentration(res)
    b = concentration(res.trades)
    c = concentration(res.r)
    assert a == b == c
    json.dumps(a.as_dict())


# ---------------------------------------------------------------------------
# drop_best
# ---------------------------------------------------------------------------


def test_drop_best_hand_computed():
    res = drop_best([5.0, 3.0, 1.0, -1.0], 1)
    assert res.dropped == ("#0",)
    assert res.dropped_total == 5.0
    b, a = res.before, res.after
    assert (b.n, b.total, b.mean, b.win_share) == (4, 8.0, 2.0, 0.75)
    assert b.t_stat == pytest.approx(t_statistic([5.0, 3.0, 1.0, -1.0]))
    assert (a.n, a.total, a.mean) == (3, 3.0, 1.0)
    assert a.win_share == pytest.approx(2.0 / 3.0)
    assert a.t_stat == pytest.approx(1.0 / (2.0 / math.sqrt(3.0)))
    assert drop_best([5.0, 3.0, 1.0, -1.0], 0.25) == res


def test_drop_best_by_group_removes_every_value_of_the_best_group():
    res = drop_best([1.0, 2.0, 3.0, -1.0, 0.5], 1, groups=["a", "a", "b", "b", "c"])
    assert res.grouped
    assert res.dropped == ("a",)
    assert res.dropped_total == 3.0
    assert res.after.n == 3
    assert res.after.mean == pytest.approx(2.5 / 3.0)


def test_drop_best_negative_control_broad_edge_survives():
    x = np.random.default_rng(3).normal(0.3, 1.0, 2000)
    res = drop_best(x, 0.01)
    assert res.k == 20
    assert not res.sign_flips
    assert res.after.mean > 0.2
    assert res.after.t_stat > 5


def test_drop_best_positive_control_lottery_ticket():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(-0.1, 0.05, 990), np.full(10, 20.0)])
    res = drop_best(x, 0.01)
    assert res.before.mean > 0
    assert res.after.mean < 0
    assert res.sign_flips
    assert any("depends on a few" in note for note in res.notes)
    assert "before" in str(res) and "after" in str(res)
    json.dumps(res.as_dict())


@pytest.mark.parametrize("k", [4, 5, 0, -1, True, 1.0, 1.5, "1"])
def test_drop_best_rejects_bad_k(k):
    with pytest.raises(ValueError):
        drop_best([5.0, 3.0, 1.0, -1.0], k)


# ---------------------------------------------------------------------------
# period_keys
# ---------------------------------------------------------------------------


def test_period_keys_iso_week_matches_python_for_every_day():
    start = date(1965, 12, 20)
    days = [start + timedelta(days=i) for i in range(24_000)]  # into 2031
    expected = [f"{d.isocalendar()[0]:04d}-W{d.isocalendar()[1]:02d}" for d in days]
    got = period_keys(np.array(days, dtype="datetime64[D]"), "W")
    assert got.tolist() == expected


@pytest.mark.parametrize(
    ("day", "week"),
    [
        ("2021-01-03", "2020-W53"),  # Sunday belongs to the previous ISO year
        ("2020-12-31", "2020-W53"),
        ("2019-12-30", "2020-W01"),  # Monday belongs to the next ISO year
        ("2024-01-01", "2024-W01"),
        ("2023-01-01", "2022-W52"),
        ("1969-12-31", "1970-W01"),
    ],
)
def test_period_keys_iso_week_edge_cases(day, week):
    assert period_keys([day], "W").tolist() == [week]


def test_period_keys_day_month_quarter_year():
    times = ["2024-02-29T23:59:59", "2024-04-01T00:00", "2023-12-31T12:00", "1969-12-31T23:00"]
    assert period_keys(times, "D").tolist() == [
        "2024-02-29",
        "2024-04-01",
        "2023-12-31",
        "1969-12-31",
    ]
    assert period_keys(times, "M").tolist() == ["2024-02", "2024-04", "2023-12", "1969-12"]
    assert period_keys(times, "Q").tolist() == ["2024-Q1", "2024-Q2", "2023-Q4", "1969-Q4"]
    assert period_keys(times, "Y").tolist() == ["2024", "2024", "2023", "1969"]


def test_period_keys_accepts_datetime64_objects_and_strings():
    expected = ["2024-03-15", "2024-03-16"]
    as_np = np.array(["2024-03-15T10:00", "2024-03-16T00:00"], dtype="datetime64[ns]")
    assert period_keys(as_np, "D").tolist() == expected
    as_dt = [datetime(2024, 3, 15, 10), datetime(2024, 3, 16)]
    assert period_keys(as_dt, "D").tolist() == expected
    assert period_keys([date(2024, 3, 15), date(2024, 3, 16)], "D").tolist() == expected
    assert period_keys(["2024-03-15", "2024-03-16T00:00:00"], "D").tolist() == expected


def test_period_keys_converts_timezone_aware_input_to_utc_without_warnings():
    plus2 = timezone(timedelta(hours=2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aware = period_keys([datetime(2024, 1, 1, 1, 0, tzinfo=plus2)], "D")
        strings = period_keys(["2024-01-01T01:00+02:00", "2024-01-01T23:30Z"], "D")
    assert aware.tolist() == ["2023-12-31"]
    assert strings.tolist() == ["2023-12-31", "2024-01-01"]


def test_period_keys_on_bar_timestamps():
    bars = random_walk_bars(48, seed=0)  # hourly from 2020-01-01
    keys = period_keys(bars.time, "D")
    assert sorted(set(keys.tolist())) == ["2020-01-01", "2020-01-02"]
    assert period_keys([], "D").tolist() == []


@pytest.mark.parametrize(
    ("times", "freq", "match"),
    [
        (["2024-01-01"], "X", "freq"),
        (["2024-01-01"], "d", "freq"),
        ([1, 2, 3], "D", "datetime64"),
        (["not a date"], "D", "ISO-8601"),
        (np.array(["NaT"], dtype="datetime64[s]"), "D", "NaT"),
        ([datetime(2024, 1, 1), None], "D", "unsupported"),
        ("2024-01-01", "D", "one-dimensional"),
    ],
)
def test_period_keys_rejects_bad_input(times, freq, match):
    with pytest.raises(ValueError, match=match):
        period_keys(times, freq)


# ---------------------------------------------------------------------------
# subperiods
# ---------------------------------------------------------------------------


def test_subperiods_hand_computed():
    res = subperiods([1.0, 2.0, -1.0, 4.0], ["a", "a", "b", "c"])
    a, b, c = res.periods
    assert (a.key, a.n, a.total, a.mean, a.win_share) == ("a", 2, 3.0, 1.5, 1.0)
    assert a.t_stat == pytest.approx(3.0)  # 1.5 / (0.7071 / sqrt(2))
    assert (b.key, b.n, b.total, b.t_stat, b.win_share) == ("b", 1, -1.0, None, 0.0)
    assert (c.key, c.total) == ("c", 4.0)
    assert res.total == 6.0
    assert res.share_positive == pytest.approx(2.0 / 3.0)
    assert res.worst.key == "b"
    assert res.best.key == "c"
    assert res.best_share == pytest.approx(4.0 / 6.0)
    assert res.concentrated
    assert any("period c" in note for note in res.notes)


def _monthly(x):
    times = np.datetime64("2023-01-01T00:00") + np.arange(len(x)) * np.timedelta64(7, "h")
    return period_keys(times, "M")


def test_subperiods_negative_control_steady_edge():
    x = np.random.default_rng(5).normal(0.2, 1.0, 1200)
    res = subperiods(x, _monthly(x))
    assert res.n_periods == 12
    assert not res.concentrated
    assert res.best_share < 0.5
    assert res.share_positive >= 0.75
    assert [p.key for p in res.periods] == sorted(p.key for p in res.periods)


def test_subperiods_positive_control_one_lucky_month():
    x = np.random.default_rng(5).normal(0.0, 1.0, 1200)
    keys = _monthly(x)
    x[keys == "2023-07"] += 3.0
    res = subperiods(x, keys)
    assert res.concentrated
    assert res.best.key == "2023-07"
    assert res.best_share > 0.5
    assert "2023-07" in str(res)


def test_subperiods_non_positive_total_has_no_share():
    res = subperiods([-1.0, 0.5], ["x", "y"])
    assert res.best_share is None
    assert not res.concentrated
    assert any("not positive" in note for note in res.notes)
    json.dumps(res.as_dict())


def test_subperiods_long_table_is_summarised_in_str():
    x = np.ones(100)
    res = subperiods(x, [f"k{i:03d}" for i in range(100)])
    text = str(res)
    assert "100 periods; the table is in .periods" in text
    assert len(text.splitlines()) < 10


def test_subperiods_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="one key per value"):
        subperiods([1.0, 2.0, 3.0], ["a", "b"])


# ---------------------------------------------------------------------------
# block_bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("j", range(10))
def test_block_bootstrap_circular_wrap_draws_every_index_equally(j):
    # Regression: clamping blocks at the end of the series gave the last value
    # (1 + 2 + 3 + 4) / 10 = 1.0 expected draws per block instead of 0.4.
    n = 10
    indicator = np.zeros(n)
    indicator[j] = 1.0
    res = block_bootstrap(indicator, block=4, n_boot=20_000, seed=j)
    assert res.samples.mean() == pytest.approx(1.0 / n, abs=0.004)


@pytest.mark.parametrize(("n", "block"), [(2, 2), (8, 2), (27, 3), (28, 4), (1000, 10), (1001, 11)])
def test_block_bootstrap_default_block_is_ceil_cube_root(n, block):
    res = block_bootstrap(np.arange(n, dtype=float), n_boot=2, seed=0)
    assert res.block == block


def test_default_block_survives_float_error_in_the_cube_root():
    # (10**18 + 1) ** (1/3) rounds to exactly 1e6, so a float ceil gives 10**6.
    assert _default_block(10**18 + 1) == 10**6 + 1
    assert _default_block(10**18) == 10**6
    assert _default_block(1) == 1


def test_block_bootstrap_constant_values():
    res = block_bootstrap(np.full(50, 2.0), n_boot=200)
    assert res.estimate == 2.0
    assert (res.ci_low, res.ci_high) == (2.0, 2.0)
    assert res.prob_positive == 1.0
    assert res.std_error == 0.0


def test_block_bootstrap_block_one_matches_iid_standard_error():
    x = np.random.default_rng(6).normal(0.0, 1.0, 400)
    res = block_bootstrap(x, block=1, n_boot=4000, seed=1)
    assert res.std_error == pytest.approx(x.std() / math.sqrt(len(x)), rel=0.1)
    assert res.estimate == pytest.approx(x.mean())


def test_block_bootstrap_negative_control_no_edge(walk):
    bars, strat = walk
    r = run_signals(bars, *strat, max_bars=1).r
    res = block_bootstrap(r, n_boot=4000, seed=2)
    assert res.ci_low < 0 < res.ci_high
    assert 0.025 < res.prob_positive < 0.975


def test_block_bootstrap_positive_control_planted_edge(planted):
    bars, strat = planted
    r = run_signals(bars, *strat, max_bars=1).r
    res = block_bootstrap(r, n_boot=4000, seed=2)
    assert res.ci_low > 0.1
    assert res.prob_positive > 0.99


def test_block_bootstrap_widens_for_serially_dependent_data():
    rng = np.random.default_rng(7)
    e = rng.normal(size=3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i]
    iid = block_bootstrap(x, block=1, n_boot=2000, seed=3)
    blocks = block_bootstrap(x, block=30, n_boot=2000, seed=3)
    # The long-run standard error is sqrt((1 + 0.8) / (1 - 0.8)) = 3 times the iid one.
    assert blocks.std_error > 2.0 * iid.std_error


def test_block_bootstrap_callable_statistic_and_determinism():
    x = np.random.default_rng(8).normal(0.1, 1.0, 300)
    a = block_bootstrap(x, statistic=np.median, n_boot=500, seed=4)
    b = block_bootstrap(x, statistic=np.median, n_boot=500, seed=4)
    assert a.statistic == "median"
    assert a.estimate == pytest.approx(float(np.median(x)))
    assert np.array_equal(a.samples, b.samples)
    assert not a.samples.flags.writeable
    named = block_bootstrap(x, statistic="median", n_boot=500, seed=4)
    assert np.allclose(named.samples, a.samples)


def test_block_bootstrap_result_does_not_depend_on_chunk_size(monkeypatch):
    import backtest_skeptic.robustness as robustness

    x = np.random.default_rng(9).normal(size=600)
    whole = block_bootstrap(x, n_boot=300, seed=5)
    monkeypatch.setattr(robustness, "_CHUNK_ELEMENTS", 1000)
    chunked = block_bootstrap(x, n_boot=300, seed=5)
    assert np.array_equal(whole.samples, chunked.samples)


def test_block_bootstrap_output():
    res = block_bootstrap([0.5, -0.2, 0.3, 0.1, 0.4], n_boot=100, seed=0)
    d = res.as_dict()
    assert "samples" not in d
    assert len(res.as_dict(include_samples=True)["samples"]) == 100
    json.dumps(d)
    assert "95% CI" in str(res)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block": 0},
        {"block": 6},
        {"block": 2.0},
        {"n_boot": 1},
        {"ci": 1.0},
        {"ci": 0.0},
        {"statistic": "mode"},
        {"statistic": lambda s: s},  # not a scalar
        {"statistic": lambda s: math.nan},
        {"seed": -1},
    ],
)
def test_block_bootstrap_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        block_bootstrap([1.0, 2.0, 3.0, 4.0, 5.0], **{"n_boot": 10, **kwargs})


def test_block_bootstrap_needs_two_values():
    with pytest.raises(ValueError, match="two"):
        block_bootstrap([1.0])


# ---------------------------------------------------------------------------
# clustered_t
# ---------------------------------------------------------------------------


def test_clustered_t_hand_computed():
    # residuals -1.5 -0.5 | 0.5 1.5 -> cluster sums -2, 2; se = sqrt(2/1 * 8) / 4 = 1
    res = clustered_t([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert isinstance(res, ClusteredT)
    mean, se, t, g = res
    assert (mean, g) == (2.5, 2)
    assert se == pytest.approx(1.0)
    assert t == pytest.approx(2.5)
    assert "CR1" in str(res)
    assert res.as_dict() == {"mean": 2.5, "se": se, "t": t, "n_clusters": 2}


def test_clustered_t_with_singleton_clusters_is_the_ordinary_t():
    x = np.random.default_rng(10).normal(0.2, 1.0, 50)
    res = clustered_t(x, np.arange(50))
    assert res.se == pytest.approx(x.std(ddof=1) / math.sqrt(50))
    assert res.t == pytest.approx(t_statistic(x))


def test_clustered_t_negative_control_independent_data():
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, 1000)
    res = clustered_t(x, rng.integers(0, 50, 1000))
    assert abs(res.t) < 3
    assert res.n_clusters == 50


def test_clustered_t_positive_control_correlated_clusters():
    # 40 days x 50 trades sharing a daily shock: intra-cluster correlation 0.5.
    rng = np.random.default_rng(12)
    clusters = np.repeat(np.arange(40), 50)
    x = rng.normal(0.0, 1.0, 40)[clusters] + rng.normal(0.0, 1.0, 2000)
    naive_se = x.std(ddof=1) / math.sqrt(len(x))
    res = clustered_t(x, clusters)
    assert res.se > 3.0 * naive_se  # theory: sqrt(1 + 49 * 0.5) ~ 5
    assert abs(res.t) < abs(x.mean() / naive_se)


def test_clustered_t_degenerate_cases():
    assert clustered_t([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]).t == math.inf
    assert clustered_t([0.0, 0.0, 0.0], [0, 1, 1]).t == 0.0
    with pytest.raises(ValueError, match="at least 2 clusters"):
        clustered_t([1.0, 2.0, 3.0], ["a", "a", "a"])
    with pytest.raises(ValueError, match="one key per value"):
        clustered_t([1.0, 2.0, 3.0], ["a", "b"])


# ---------------------------------------------------------------------------
# cost_stress
# ---------------------------------------------------------------------------


def test_cost_stress_hand_computed_reruns_the_engine():
    # Long at 100 (bar 1 open), stop 99, target 102, hit on bar 1.
    # m=0: R = 2. m=1 (h=0.1): entry 100.1, risk 1.1, exit 101.9, R = 1.8/1.1.
    # m=2 (h=0.2): entry 100.2, risk 1.2, exit 101.8, R = 1.6/1.2.
    # A post-hoc "R minus round-trip cost / risk" would give 1.8 at m=1.
    bars = flat_then([(100.0, 102.5, 99.8, 102.0)])
    res = cost_stress(
        bars, once_at(0, (1, 99.0, 102.0)), costs=Costs(half_spread=0.1), multipliers=(2, 0, 1)
    )
    assert [row.multiplier for row in res.rows] == [0.0, 1.0, 2.0]
    assert [row.expectancy_r for row in res.rows] == pytest.approx([2.0, 1.8 / 1.1, 1.6 / 1.2])
    assert [row.n_trades for row in res.rows] == [1, 1, 1]
    assert res.status == "profitable_at_all"
    assert res.breakeven_multiplier is None
    assert "profitable at every multiplier tested" in res.message


def test_cost_stress_matches_direct_engine_runs(planted):
    bars, strat = planted
    costs = Costs(half_spread=0.02, fee_rate=0.0002, slippage=0.01)
    res = cost_stress(bars, strat, costs=costs, max_bars=3)
    for row in res.rows:
        direct = run_signals(bars, *strat, costs=costs.scaled(row.multiplier), max_bars=3)
        assert row.n_trades == len(direct.trades)
        assert row.expectancy_r == pytest.approx(float(direct.r.mean()), abs=1e-12)
        assert row.total_r == pytest.approx(float(direct.r.sum()), abs=1e-9)


def test_cost_stress_callable_and_arrays_agree(planted):
    bars, (side, stop, target) = planted
    bars = bars[:1500]
    side, stop, target = side[:1500], stop[:1500], target[:1500]

    def decide(i, history):
        return (side[i], stop[i], target[i]) if side[i] != 0 else None

    costs = Costs(fee_rate=0.00125)
    a = cost_stress(bars, decide, costs=costs, multipliers=(1, 2), max_bars=1)
    b = cost_stress(bars, (side, stop, target), costs=costs, multipliers=(1, 2), max_bars=1)
    assert a == b


def test_cost_stress_positive_control_planted_edge_crosses(planted):
    bars, strat = planted
    res = cost_stress(bars, strat, costs=Costs(fee_rate=0.00125), max_bars=1)
    exps = [row.expectancy_r for row in res.rows]
    assert all(e1 > e2 for e1, e2 in pairwise(exps))
    assert res.status == "crossing"
    assert 1.5 < res.breakeven_multiplier < 2.5
    (m1, e1), (m2, e2) = [(r.multiplier, r.expectancy_r) for r in res.rows[2:4]]
    assert e1 > 0 >= e2
    assert res.breakeven_multiplier == pytest.approx(m1 + (m2 - m1) * e1 / (e1 - e2))
    assert "break-even at about" in str(res)
    json.dumps(res.as_dict())


def test_cost_stress_negative_control_random_walk_is_unprofitable(walk):
    bars, strat = walk
    costs = Costs(fee_rate=0.00125)
    res = cost_stress(bars, strat, costs=costs, max_bars=1)
    assert res.at(1.0).expectancy_r < 0
    assert res.status == "unprofitable_at_lowest"
    assert res.breakeven_multiplier is None
    assert "unprofitable at the lowest multiplier tested" in res.message
    from_one = cost_stress(bars, strat, costs=costs, multipliers=(1, 2, 3), max_bars=1)
    assert from_one.rows[0].multiplier == 1.0
    assert from_one.status == "unprofitable_at_lowest"
    assert "(1x costs" in from_one.message


def _rows(points):
    return [CostRow(m, 10, e, 10 * e, None, 0) for m, e in points]


def test_breakeven_interpolation_and_statuses():
    status, m, _, notes = _breakeven(_rows([(1, 0.3), (2, 0.1), (3, -0.1)]))
    assert (status, m, notes) == ("crossing", pytest.approx(2.5), [])
    status, m, msg, _ = _breakeven(_rows([(0.5, 0.0), (1, -0.1)]))
    assert (status, m) == ("unprofitable_at_lowest", None)
    assert "lowest multiplier tested" in msg
    status, m, msg, _ = _breakeven(_rows([(1, 0.2), (2, 0.1)]))
    assert (status, m) == ("profitable_at_all", None)
    status, m, _, notes = _breakeven(_rows([(1, 0.2), (2, -0.2), (3, 0.05)]))
    assert (status, m) == ("crossing", pytest.approx(1.5))
    assert any("not monotonic" in note for note in notes)
    assert _breakeven([CostRow(1.0, 0, None, 0.0, None, 3)])[0] == "no_trades"


def test_cost_stress_notes_zero_costs():
    bars = flat_then([(100.0, 102.5, 99.8, 102.0)])
    res = cost_stress(bars, once_at(0, (1, 99.0, 102.0)), costs=Costs())
    assert any("every cost component is zero" in note for note in res.notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multipliers": ()},
        {"multipliers": (1, 1)},
        {"multipliers": (-1, 1)},
        {"multipliers": (math.inf,)},
        {"multipliers": (True,)},
        {"max_bars": 3, "costs_typo": 1},
        {"costs": None},
    ],
)
def test_cost_stress_rejects_bad_arguments(kwargs):
    bars = flat_then([(100.0, 102.5, 99.8, 102.0)])
    args = {"costs": Costs(half_spread=0.1), **kwargs}
    with pytest.raises(ValueError):
        cost_stress(bars, once_at(0, (1, 99.0, 102.0)), **args)


def test_cost_stress_rejects_bad_strategy_and_engine_costs():
    bars = flat_then([(100.0, 102.5, 99.8, 102.0)])
    with pytest.raises(ValueError, match="strategy"):
        cost_stress(bars, [1, 2], costs=Costs())
    with pytest.raises(ValueError, match="Bars"):
        cost_stress([1, 2], once_at(0, (1, 99.0, 102.0)), costs=Costs())


# ---------------------------------------------------------------------------
# entry_delay_test
# ---------------------------------------------------------------------------


def test_entry_delay_positive_control_one_bar_edge_collapses(planted):
    bars, strat = planted
    res = entry_delay_test(bars, strat, max_bars=1)
    assert [row.delay for row in res.rows] == [0, 1, 2, 3]
    assert res.edge_at_first
    assert res.at(0).expectancy_r > 0.15
    assert abs(res.at(1).expectancy_r) < 0.06
    assert res.collapse_delay == 1
    assert any("collapses at delay 1" in note for note in res.notes)
    json.dumps(res.as_dict())


def test_entry_delay_negative_control_random_walk_stays_near_zero(walk):
    bars, strat = walk
    res = entry_delay_test(bars, strat, max_bars=1)
    assert not res.edge_at_first
    assert res.collapse_delay is None
    for row in res.rows:
        assert abs(row.expectancy_r) < 0.1
        assert abs(row.t_stat) < 3
        assert row.retention is None
    assert any("no edge" in note for note in res.notes)


def test_entry_delay_persistent_edge_decays_gradually():
    bars = random_walk_bars(4000, drift=0.003, seed=13)
    strat = brackets(bars, np.ones(len(bars)))
    res = entry_delay_test(bars, strat, max_bars=1)
    assert res.edge_at_first
    assert res.collapse_delay is None
    assert all(row.retention > 0.5 for row in res.rows)
    assert any("decays gradually" in note for note in res.notes)


def test_entry_delay_rows_match_direct_engine_runs(planted):
    bars, strat = planted
    costs = Costs(half_spread=0.01)
    res = entry_delay_test(bars, strat, delays=(2, 0), costs=costs, max_bars=2)
    assert [row.delay for row in res.rows] == [0, 2]
    for row in res.rows:
        direct = run_signals(bars, *strat, costs=costs, max_bars=2, entry_delay=row.delay)
        assert row.n_trades == len(direct.trades)
        assert row.expectancy_r == pytest.approx(float(direct.r.mean()), abs=1e-12)
    assert res.costs == costs


def test_entry_delay_with_decide_callable():
    bars = flat_then([(100.0, 100.5, 99.5, 100.0), (100.0, 102.5, 99.8, 102.0)])
    res = entry_delay_test(bars, once_at(0, (1, 99.0, 102.0)), delays=(0, 1))
    assert res.at(0).n_trades == 1
    assert res.at(0).expectancy_r == pytest.approx(2.0)
    direct = run(bars, once_at(0, (1, 99.0, 102.0)), entry_delay=1)
    assert res.at(1).n_trades == len(direct.trades)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delays": ()},
        {"delays": (0, 0)},
        {"delays": (-1,)},
        {"delays": (0.5,)},
        {"entry_delay": 1},
        {"costs": "none"},
        {"warmup": 0, "bogus": 1},
    ],
)
def test_entry_delay_rejects_bad_arguments(kwargs):
    bars = flat_then([(100.0, 102.5, 99.8, 102.0)])
    with pytest.raises(ValueError):
        entry_delay_test(bars, once_at(0, (1, 99.0, 102.0)), **kwargs)
