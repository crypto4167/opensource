import json
import math

import numpy as np
import pytest

from backtest_skeptic import Bars, Setup
from backtest_skeptic.lookahead import (
    LookaheadReport,
    check_factory,
    perturb_future,
    perturbation_test,
    self_test,
    truncation_test,
)
from backtest_skeptic.synthetic import oracle_signal, planted_signal_bars, random_walk_bars


@pytest.fixture(scope="module")
def walk():
    return random_walk_bars(400, seed=11)


# ------------------------------------------------------------------ probes


def rolling_mean(x, w):
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        cs = np.cumsum(np.concatenate([[0.0], x]))
        out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def causal_mean(bars):
    return rolling_mean(bars.close, 5)


def ewma(bars):
    out = np.empty(len(bars))
    level = float(bars.close[0])
    for i, v in enumerate(bars.close.tolist()):
        level += 0.2 * (v - level)
        out[i] = level
    return out


def centred_mean(bars):
    out = np.full(len(bars), np.nan)
    if len(bars) > 2:
        out[:-2] = rolling_mean(bars.close, 5)[2:]
    return out


def zscore(bars):
    x = bars.close
    sd = x.std()
    return np.zeros(len(x)) if sd == 0 else (x - x.mean()) / sd


def shift_back(bars):
    out = np.full(len(bars), np.nan)
    out[:-1] = bars.close[1:]
    return out


def backfill_sparse(bars):
    x = bars.close
    n = len(x)
    sparse = np.full(n, np.nan)
    idx = np.arange(n)
    sparse[idx % 7 == 3] = x[idx % 7 == 3]
    nxt_idx = np.where(np.isnan(sparse), n, idx)
    nxt = np.minimum.accumulate(nxt_idx[::-1])[::-1]
    out = np.full(n, np.nan)
    ok = nxt < n
    out[ok] = sparse[nxt[ok]]
    return out


def window_too_far(bars):
    out = np.full(len(bars), np.nan)
    out[:-1] = rolling_mean(bars.close, 5)[1:]
    return out


HONEST = [causal_mean, ewma]
LEAKY = [centred_mean, zscore, shift_back, backfill_sparse, window_too_far]


def by_length(table):
    """fn(bars) that returns ``table[len(bars)]`` - for hand-computed cases."""

    def fn(bars):
        return np.asarray(table[len(bars)], dtype=float)

    return fn


# --------------------------------------------------------------- truncation


@pytest.mark.parametrize("check", ["last", "prefix"])
@pytest.mark.parametrize("fn", HONEST, ids=lambda f: f.__name__)
def test_truncation_passes_honest_functions(walk, fn, check):
    rep = truncation_test(fn, walk, check=check)
    assert rep.passed, str(rep)
    assert rep.kind == "truncation"
    assert rep.n_mismatches == 0
    assert rep.first_bad_index is None
    assert rep.n_checked == len(rep.checkpoints) >= 30


@pytest.mark.parametrize("check", ["last", "prefix"])
@pytest.mark.parametrize("fn", LEAKY, ids=lambda f: f.__name__)
def test_truncation_catches_leaky_functions(walk, fn, check):
    rep = truncation_test(fn, walk, check=check)
    assert not rep.passed
    assert rep.leaks
    assert rep.first_bad_index is not None
    assert f"earliest leaking row is {rep.first_bad_index}" in str(rep)


@pytest.mark.parametrize("horizon", [1, 5])
def test_truncation_catches_oracle_signal(walk, horizon):
    rep = truncation_test(lambda b: oracle_signal(b, horizon), walk)
    assert rep.leaks
    # Every prefix shorter than the data sees 0 where the full run saw +-1.
    assert rep.n_mismatches == sum(1 for k in rep.checkpoints if k <= len(walk) - horizon)


def test_nan_against_finite_is_a_mismatch_regression():
    # Regression: an earlier implementation treated NaN-vs-number as equal,
    # which let backward fills and centred windows pass.
    bars = Bars.from_close([1.0, 2.0, 3.0, 4.0, 5.0])
    table = {k: [*range(1, k), math.nan] for k in range(1, 5)}
    table[5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    rep = truncation_test(by_length(table), bars, checkpoints=[2, 3, 4, 5])
    assert not rep.passed
    assert rep.n_checked == 3  # prefix 5 is the full data and checks nothing
    assert rep.n_values == 3
    assert rep.n_mismatches == 3
    assert rep.first_bad_index == 1
    assert [(m.checkpoint, m.row, m.expected) for m in rep.mismatches] == [
        (2, 1, 2.0),
        (3, 2, 3.0),
        (4, 3, 4.0),
    ]
    assert all(m.column is None and math.isnan(m.got) for m in rep.mismatches)


def test_nan_matches_nan():
    bars = random_walk_bars(60, seed=1)
    rep = truncation_test(lambda b: np.full(len(b), np.nan), bars, check="prefix")
    assert rep.passed
    assert any("NaN on every row" in note for note in rep.notes)


def test_infinity_only_matches_the_same_infinity_regression():
    # Regression: with rtol > 0 the tolerance rtol * inf is inf, so a naive
    # |a - b| <= tol test would call inf and 1.0 equal.
    bars = Bars.from_close([1.0, 2.0, 3.0])
    rep = truncation_test(
        by_length({2: [math.inf, 1.0], 3: [math.inf, math.inf, -math.inf]}),
        bars,
        checkpoints=[2, 3],
        rtol=1e-3,
        check="prefix",
    )
    assert rep.n_mismatches == 1
    (m,) = rep.mismatches
    assert (m.checkpoint, m.row, m.expected, m.got) == (2, 1, math.inf, 1.0)


def test_relative_tolerance():
    bars = random_walk_bars(80, seed=2)

    def noisy(b):  # rounding noise that depends on the series length
        return b.close * (1.0 + 1e-12 * len(b))

    assert truncation_test(noisy, bars).passed
    rep = truncation_test(noisy, bars, rtol=0.0)
    assert rep.leaks


def test_absolute_tolerance():
    bars = random_walk_bars(50, seed=3)

    def drifting(b):
        return b.close + 1e-7 * len(b)

    assert truncation_test(drifting, bars).leaks
    assert truncation_test(drifting, bars, atol=1e-5).passed


def test_two_dimensional_output_reports_the_leaking_column(walk):
    rep = truncation_test(lambda b: np.column_stack([causal_mean(b), centred_mean(b)]), walk)
    assert rep.leaks
    assert {m.column for m in rep.mismatches} == {1}
    assert "column 1" in str(rep)
    honest = truncation_test(lambda b: np.column_stack([causal_mean(b), ewma(b)]), walk)
    assert honest.passed


def test_pandas_outputs_work_through_asarray(walk):
    pd = pytest.importorskip("pandas")

    def frame(b):
        s = pd.Series(b.close)
        return pd.DataFrame(
            {"slow": s.rolling(10).mean(), "centred": s.rolling(5, center=True).mean()}
        )

    rep = truncation_test(frame, walk)
    assert rep.column_names == ("slow", "centred")
    assert {m.column for m in rep.mismatches} == {1}
    assert "'centred'" in str(rep)
    assert truncation_test(lambda b: pd.Series(b.close).ewm(span=10).mean(), walk).passed
    assert truncation_test(lambda b: pd.Series(b.close).shift(-1), walk).leaks
    assert truncation_test(lambda b: pd.Series(b.close).bfill(), walk).passed  # nothing to fill


def test_string_signals_are_compared_by_equality(walk):
    def labels(b, ahead):
        x = b.close
        out = np.full(len(x), "", dtype=object)
        if ahead:
            out[:-1] = np.where(x[1:] > x[:-1], "BUY", "SELL")
        else:
            out[1:] = np.where(x[1:] > x[:-1], "BUY", "SELL")
        return out

    assert truncation_test(lambda b: labels(b, False), walk).passed
    rep = truncation_test(lambda b: labels(b, True), walk)
    assert rep.leaks
    assert rep.mismatches[0].got == ""
    assert perturbation_test(lambda b: labels(b, True), walk).leaks


def test_none_in_object_output_counts_as_nan(walk):
    def with_none(b):
        out = [None] * len(b)
        for i in range(len(b) - 1):
            out[i] = 1 if b.close[i + 1] > b.close[i] else -1
        return out

    rep = truncation_test(with_none, walk)
    assert rep.leaks
    assert all(m.got is not None and math.isnan(m.got) for m in rep.mismatches)


def test_default_checkpoints_cover_rows_where_a_sparse_signal_fires():
    bars = random_walk_bars(3000, seed=4)
    events = [517, 1609, 2442]

    def sparse_cheat(b):
        out = np.zeros(len(b))
        for r in events:
            if r + 1 < len(b):
                out[r] = 1.0 if b.close[r + 1] > b.close[r] else -1.0
        return out

    rep = truncation_test(sparse_cheat, bars)
    assert {r + 1 for r in events} <= set(rep.checkpoints)
    assert rep.leaks
    assert {m.row for m in rep.mismatches} == set(events)


def test_every_prefix_is_used_when_the_series_is_short():
    bars = random_walk_bars(12, seed=5)
    rep = truncation_test(causal_mean, bars, n_checkpoints=40)
    assert rep.checkpoints == tuple(range(1, 12))


def test_checkpoint_choice_is_seeded(walk):
    a = truncation_test(causal_mean, walk, seed=7).checkpoints
    b = truncation_test(causal_mean, walk, seed=7).checkpoints
    c = truncation_test(causal_mean, walk, seed=8).checkpoints
    assert a == b
    assert a != c


def test_prefix_mode_pins_the_earliest_leaking_row(walk):
    rep = truncation_test(centred_mean, walk, checkpoints=[50], check="prefix")
    assert rep.n_values == 50
    assert rep.n_mismatches == 2
    assert [m.row for m in rep.mismatches] == [48, 49]
    assert rep.first_bad_index == 48


def test_exception_below_min_history_is_skipped(walk):
    def needs_ten(b):
        if len(b) < 10:
            raise IndexError("not enough bars")
        return rolling_mean(b.close, 10)

    rep = truncation_test(needs_ten, walk, checkpoints=[3, 9, 10, 50, 200], min_history=10)
    assert rep.passed, str(rep)
    assert rep.skipped == (3, 9)
    assert rep.n_checked == 3


def test_exception_at_or_above_min_history_fails_the_test(walk):
    def needs_ten(b):
        if len(b) < 10:
            raise IndexError("not enough bars")
        return rolling_mean(b.close, 10)

    rep = truncation_test(needs_ten, walk, checkpoints=[3, 9, 10, 50], min_history=5)
    assert not rep.passed
    assert rep.skipped == (3,)
    assert len(rep.errors) == 1
    assert rep.errors[0].startswith("prefix 9: IndexError: not enough bars")
    assert rep.n_mismatches == 0
    assert "raised on 1 probe" in str(rep)


def test_function_that_raises_on_every_prefix_does_not_pass(walk):
    def only_full(b):
        if len(b) < len(walk):
            raise RuntimeError("boom")
        return np.zeros(len(b))

    rep = truncation_test(only_full, walk)
    assert not rep.passed
    assert rep.n_checked == 0
    assert len(rep.errors) == len(rep.checkpoints)


def test_all_checkpoints_skipped_is_inconclusive_not_a_pass(walk):
    def needs_ten(b):
        if len(b) < 10:
            raise IndexError("short")
        return np.zeros(len(b))

    rep = truncation_test(needs_ten, walk, checkpoints=[2, 5], min_history=10)
    assert not rep.passed
    assert rep.n_checked == 0
    assert "INCONCLUSIVE" in str(rep)


def test_wrong_output_length_raises(walk):
    with pytest.raises(ValueError, match="one row per bar"):
        truncation_test(lambda b: b.close[:-1], walk)
    with pytest.raises(ValueError, match="prefix"):
        truncation_test(lambda b: np.zeros(max(len(b), len(walk))), walk)
    with pytest.raises(ValueError, match="column"):
        truncation_test(lambda b: np.zeros((len(b), 1 if len(b) < len(walk) else 2)), walk)
    with pytest.raises(ValueError, match="shape"):
        truncation_test(lambda b: np.zeros((len(b), 2, 2)), walk)


def test_non_deterministic_function_is_rejected(walk):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="deterministic"):
        truncation_test(lambda b: rng.normal(size=len(b)), walk)
    with pytest.raises(ValueError, match="deterministic"):
        perturbation_test(lambda b: rng.normal(size=len(b)), walk)


def test_truncation_rejects_bad_options(walk):
    with pytest.raises(ValueError, match="outside"):
        truncation_test(causal_mean, walk, checkpoints=[0])
    with pytest.raises(ValueError, match="outside"):
        truncation_test(causal_mean, walk, checkpoints=[len(walk) + 1])
    with pytest.raises(ValueError, match="integers"):
        truncation_test(causal_mean, walk, checkpoints=[True])
    with pytest.raises(ValueError, match="no checkpoint"):
        truncation_test(causal_mean, walk, checkpoints=[])
    with pytest.raises(ValueError, match="check"):
        truncation_test(causal_mean, walk, check="all")
    with pytest.raises(ValueError, match="rtol"):
        truncation_test(causal_mean, walk, rtol=-1.0)
    with pytest.raises(ValueError, match="atol"):
        truncation_test(causal_mean, walk, atol=math.nan)
    with pytest.raises(ValueError, match="min_history"):
        truncation_test(causal_mean, walk, min_history=len(walk) + 1)
    with pytest.raises(ValueError, match="n_checkpoints"):
        truncation_test(causal_mean, walk, n_checkpoints=0)
    with pytest.raises(ValueError, match="Bars"):
        truncation_test(causal_mean, walk.close)
    with pytest.raises(ValueError, match="callable"):
        truncation_test(42, walk)


def test_exception_from_the_full_run_propagates(walk):
    def broken(b):
        raise KeyError("missing column")

    with pytest.raises(KeyError):
        truncation_test(broken, walk)


# -------------------------------------------------------------- perturbation


def test_perturb_future_keeps_the_past_and_changes_every_future_bar(walk):
    cut = 150
    p = perturb_future(walk, cut, seed=3)
    for name in ("open", "high", "low", "close"):
        orig, new = getattr(walk, name), getattr(p, name)
        assert np.array_equal(new[: cut + 1], orig[: cut + 1])
        ratio = new[cut + 1 :] / orig[cut + 1 :]
        assert (ratio != 1.0).all()
    # Opens and closes are scaled by factors in [1.5, 3]; highs and lows are rebuilt.
    for name in ("open", "close"):
        ratio = getattr(p, name)[cut + 1 :] / getattr(walk, name)[cut + 1 :]
        assert ratio.min() >= 1.5 and ratio.max() <= 3.0
    assert (p.high >= np.maximum(p.open, p.close)).all()
    assert (p.low <= np.minimum(p.open, p.close)).all()
    assert np.array_equal(p.volume, walk.volume)
    assert np.array_equal(p.time, walk.time)


def test_perturb_future_inverted_scales_down_and_is_seeded(walk):
    down = perturb_future(walk, 10, seed=1, invert=True)
    ratio = down.close[11:] / walk.close[11:]
    assert ratio.max() <= 1 / 1.5 and ratio.min() >= 1 / 3.0
    again = perturb_future(walk, 10, seed=1, invert=True)
    assert np.array_equal(again.close, down.close)
    assert not np.array_equal(perturb_future(walk, 10, seed=2).close, down.close)


def test_perturb_future_rejects_bad_cuts_and_zero_prices(walk):
    with pytest.raises(ValueError, match="cut"):
        perturb_future(walk, len(walk) - 1)
    with pytest.raises(ValueError, match="cut"):
        perturb_future(walk, -1)
    zeros = Bars.from_close([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        perturb_future(zeros, 1)


@pytest.mark.parametrize("fn", HONEST, ids=lambda f: f.__name__)
def test_perturbation_passes_honest_functions(walk, fn):
    rep = perturbation_test(fn, walk)
    assert rep.passed, str(rep)
    assert rep.kind == "perturbation"
    assert rep.n_checked == len(rep.checkpoints) * 3


@pytest.mark.parametrize("fn", LEAKY, ids=lambda f: f.__name__)
def test_perturbation_catches_leaky_functions(walk, fn):
    rep = perturbation_test(fn, walk)
    assert rep.leaks, str(rep)
    assert not rep.passed


@pytest.mark.parametrize("horizon", [1, 5])
def test_perturbation_catches_oracle_signal(walk, horizon):
    rep = perturbation_test(lambda b: oracle_signal(b, horizon), walk)
    assert rep.leaks
    for m in rep.mismatches:
        assert m.checkpoint - horizon < m.row <= m.checkpoint


def test_perturbation_scales_the_future_both_ways(walk):
    # The oracle already says "up" at this cut, so scaling the future up
    # changes nothing; only the scaled-down trial exposes the leak.
    sig = oracle_signal(walk, 1)
    cut = int(np.flatnonzero(sig[100:] == 1)[0]) + 100
    up_only = perturbation_test(lambda b: oracle_signal(b, 1), walk, cuts=[cut], n_trials=1)
    assert up_only.passed
    both = perturbation_test(lambda b: oracle_signal(b, 1), walk, cuts=[cut], n_trials=2)
    assert both.leaks
    assert [(m.row, m.trial, m.expected, m.got) for m in both.mismatches] == [(cut, 1, 1.0, -1.0)]


def test_perturbation_hand_computed_shift():
    bars = random_walk_bars(6, seed=9)
    rep = perturbation_test(shift_back, bars, cuts=[3], n_trials=2)
    assert rep.checkpoints == (3,)
    assert rep.n_checked == 2
    assert rep.n_values == 8  # rows 0..3, two trials
    assert rep.n_mismatches == 2
    assert [(m.row, m.trial) for m in rep.mismatches] == [(3, 0), (3, 1)]
    assert rep.mismatches[0].expected == pytest.approx(bars.close[4])


def test_planted_signal_lookup_is_causal():
    bars, signal = planted_signal_bars(500, seed=12)

    def lookup(b):
        return signal[: len(b)]

    assert truncation_test(lookup, bars).passed
    assert perturbation_test(lookup, bars).passed


def test_perturbation_reports_exceptions_as_errors(walk):
    ceiling = float(walk.close.max()) * 1.2

    def fragile(b):
        if b.close.max() > ceiling:
            raise OverflowError("price out of range")
        return causal_mean(b)

    rep = perturbation_test(fragile, walk, cuts=[100, 200])
    assert not rep.passed
    assert rep.n_mismatches == 0
    assert any(e.startswith("cut 100, trial 0: OverflowError") for e in rep.errors)


def test_perturbation_rejects_bad_options(walk):
    with pytest.raises(ValueError, match="outside"):
        perturbation_test(causal_mean, walk, cuts=[len(walk) - 1])
    with pytest.raises(ValueError, match="n_trials"):
        perturbation_test(causal_mean, walk, n_trials=0)
    with pytest.raises(ValueError, match="n_cuts"):
        perturbation_test(causal_mean, walk, n_cuts=0)
    with pytest.raises(ValueError, match="at least 2"):
        perturbation_test(causal_mean, Bars.from_close([1.0]))


# ------------------------------------------------------------------- factory


def honest_factory(bars):
    ma = rolling_mean(bars.close, 10)  # causal: precomputing it is fine

    def decide(i, history):
        c = float(history.close[-1])
        if np.isnan(ma[i]) or c <= ma[i]:
            return None
        return (1, c * 0.99, c * 1.02)

    return decide


def history_only_factory(bars):
    def decide(i, history):
        if len(history) < 20:
            return None
        c = history.close
        if c[-1] < c[-20:].mean():
            return {"side": "SELL", "stop": c[-1] * 1.01, "target": c[-1] * 0.98}
        return None

    return decide


def leaky_factory(bars):
    future_mean = centred_mean(bars)  # precomputed from the full data

    def decide(i, history):
        c = float(history.close[-1])
        if np.isnan(future_mean[i]) or c >= future_mean[i]:
            return None
        return Setup(1, c * 0.99, c * 1.02)

    return decide


def oracle_factory(bars):
    sig = oracle_signal(bars, 3)

    def decide(i, history):
        c = float(history.close[-1])
        if sig[i] > 0 and i % 4 == 0:
            return (1, c * 0.99, c * 1.01)
        return None

    return decide


@pytest.mark.parametrize("factory", [honest_factory, history_only_factory])
def test_factory_passes_honest_strategies(walk, factory):
    rep = check_factory(factory, walk)
    assert rep.passed, str(rep)
    assert rep.kind == "factory"
    assert rep.n_checked == len(rep.checkpoints) * 2
    assert rep.n_values == 2 * sum(c + 1 for c in rep.checkpoints)


@pytest.mark.parametrize("factory", [leaky_factory, oracle_factory])
def test_factory_catches_leak_through_precomputed_full_data(walk, factory):
    rep = check_factory(factory, walk)
    assert rep.leaks, str(rep)
    assert not rep.passed
    m = rep.mismatches[0]
    assert m.row <= m.checkpoint
    assert "decide(" in str(rep)


def test_factory_cuts_include_bars_with_a_setup(walk):
    rep = check_factory(oracle_factory, walk, n_cuts=8)
    decide = oracle_factory(walk)
    active = {i for i in range(len(walk) - 1) if decide(i, walk.prefix(i + 1)) is not None}
    assert active & set(rep.checkpoints)


def test_factory_warmup_and_invalid_levels(walk):
    def factory(bars):
        def decide(i, history):
            if i < 10:
                raise IndexError("decide called during warm-up")
            if i % 5 == 0:
                return (1, math.nan, 2.0)  # invalid levels: the engine skips these
            return None

        return decide

    with pytest.raises(IndexError):
        check_factory(factory, walk)
    rep = check_factory(factory, walk, warmup=10)
    assert rep.passed, str(rep)
    assert any("no trade" in note for note in rep.notes)


def test_factory_errors_on_perturbed_data_are_reported(walk):
    ceiling = float(walk.close.max()) * 1.2

    def factory(bars):
        if bars.close.max() > ceiling:
            raise ZeroDivisionError("bad scaling")
        return honest_factory(bars)

    rep = check_factory(factory, walk, cuts=[100])
    assert not rep.passed
    assert rep.errors[0].startswith("cut 100, trial 0: ZeroDivisionError")


def test_factory_level_tolerance(walk):
    def factory(bars):
        bump = 1e-12 * float(bars.close[-1])  # depends on the future, but negligibly

        def decide(i, history):
            c = float(history.close[-1])
            return (1, c * 0.99 + bump, c * 1.02)

        return decide

    assert check_factory(factory, walk, cuts=[200], rtol=1e-9).passed
    assert check_factory(factory, walk, cuts=[200], rtol=0.0).leaks


def test_factory_rejects_bad_input(walk):
    with pytest.raises(ValueError, match="callable"):
        check_factory(lambda b: 3, walk)
    with pytest.raises(ValueError, match="outside"):
        check_factory(honest_factory, walk, cuts=[len(walk) - 1])
    with pytest.raises(ValueError, match="warmup"):
        check_factory(honest_factory, walk, warmup=len(walk))


# ----------------------------------------------------------- report & self-test


def test_report_as_dict_and_str(walk):
    bad = truncation_test(centred_mean, walk)
    assert isinstance(bad, LookaheadReport)
    d = bad.as_dict()
    json.dumps(d)  # plain Python types only
    assert d["passed"] is False
    assert d["first_bad_index"] == bad.first_bad_index
    assert d["mismatches"][0]["row"] == bad.first_bad_index
    assert len(d["mismatches"]) <= 50
    text = str(bad)
    assert text.startswith("truncation test FAILED: LOOKAHEAD")
    assert f"row {bad.first_bad_index} is" in text
    good = perturbation_test(causal_mean, walk)
    assert str(good).startswith("perturbation test PASSED")
    with pytest.raises(AttributeError):
        good.passed = False  # type: ignore[misc]


def test_mismatches_are_sorted_and_capped(walk):
    rep = truncation_test(zscore, walk, check="prefix")
    rows = [m.row for m in rep.mismatches]
    assert rows == sorted(rows)
    assert len(rep.mismatches) == 50
    assert rep.n_mismatches > 50
    assert rep.first_bad_index == rows[0] == 0


def test_self_test_judges_every_probe_correctly():
    results = self_test()
    assert set(results) >= {
        "causal_rolling_mean",
        "ewma",
        "centred_rolling_mean",
        "full_sample_zscore",
        "shift_minus_1",
        "backfill_sparse",
        "window_includes_next_bar",
        "factory_honest",
        "factory_precomputed_leak",
    }
    assert all(results.values()), results


# ------------------------------------------------------- adversarial review


def _next_close(b):
    out = np.full(len(b), np.nan)
    out[:-1] = b.close[1:]
    return out


def test_full_length_prefix_checks_nothing(walk):
    n = len(walk)
    report = truncation_test(_next_close, walk, checkpoints=[n])
    assert not report.passed
    assert report.n_checked == 0
    assert not truncation_test(_next_close, walk, min_history=n).passed
