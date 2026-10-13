"""Tests for the one-call audit report."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from backtest_skeptic import Bars, Costs, TradeLog, run_signals, synthetic
from backtest_skeptic.engine import BacktestResult
from backtest_skeptic.report import (
    CHECK_TITLES,
    THRESHOLDS,
    AuditReport,
    CheckResult,
    audit,
)

N_SIMS = 300
BRACKET = 0.006
MAX_BARS = 5


def _bracket_run(bars: Bars, side: np.ndarray, costs: Costs | None = None) -> BacktestResult:
    s = side.astype(np.float64)
    c = bars.close
    return run_signals(
        bars, s, c * (1 - s * BRACKET), c * (1 + s * BRACKET), max_bars=MAX_BARS, costs=costs
    )


@pytest.fixture(scope="module")
def planted() -> tuple[Bars, BacktestResult]:
    bars, signal = synthetic.planted_signal_bars(4000, edge=0.5, signal_prob=0.1, seed=0)
    return bars, _bracket_run(bars, signal)


@pytest.fixture(scope="module")
def random_walk() -> tuple[Bars, BacktestResult]:
    bars = synthetic.random_walk_bars(4000, seed=1)
    rng = np.random.default_rng(1)
    side = np.where(rng.random(len(bars)) < 0.1, rng.choice([-1, 1], len(bars)), 0)
    return bars, _bracket_run(bars, side)


def test_planted_signal_passes_baseline_and_random_entry(
    planted: tuple[Bars, BacktestResult],
) -> None:
    bars, result = planted
    report = audit(bars, result, n_sims=N_SIMS)
    assert [c.name for c in report.checks] == list(CHECK_TITLES)
    assert report.check("fill_realism").status == "PASS"
    assert report.check("no_skill_baseline").status == "PASS"
    assert report.check("random_entry").status == "PASS"
    assert report.check("sharpe").status == "PASS"
    assert report.worst_status in ("PASS", "WARN")
    assert report.source == "backtest result"
    assert report.n_trades == len(result)


def test_random_walk_strategy_warns(random_walk: tuple[Bars, BacktestResult]) -> None:
    bars, result = random_walk
    report = audit(bars, result, n_sims=N_SIMS)
    assert report.check("fill_realism").status == "PASS"
    assert report.check("no_skill_baseline").status == "WARN"
    assert report.check("random_entry").status == "WARN"
    assert report.check("sharpe").status == "WARN"
    assert report.worst_status == "WARN"
    assert "Not distinguishable from random entries" in report.check("random_entry").summary


def test_external_log_without_levels_skips_with_reasons(
    random_walk: tuple[Bars, BacktestResult],
) -> None:
    bars, result = random_walk
    records = [
        {
            k: v
            for k, v in t.as_dict().items()
            if k in ("side", "entry_index", "exit_index", "entry_price", "exit_price")
        }
        for t in result.trades
    ]
    log = TradeLog.from_records(records, bars)
    report = audit(bars, log, n_sims=N_SIMS)
    assert report.source == "trade log"
    baseline = report.check("no_skill_baseline")
    rand = report.check("random_entry")
    assert baseline.status == "SKIP" and "stop and target" in baseline.summary
    assert rand.status == "SKIP" and "no stop or target" in rand.summary
    assert report.check("fill_realism").status == "PASS"
    assert any("price units" in note for note in report.notes)


def test_external_log_that_reproduces_runs_every_check(
    random_walk: tuple[Bars, BacktestResult],
) -> None:
    bars, result = random_walk
    log = TradeLog.from_trades(result.trades, bars)
    report = audit(bars, log, max_bars=MAX_BARS, n_sims=N_SIMS)
    direct = audit(bars, result, n_sims=N_SIMS)
    for name in ("no_skill_baseline", "random_entry", "sharpe"):
        assert report.check(name).status == direct.check(name).status
        assert report.check(name).details == direct.check(name).details


def test_external_log_that_does_not_reproduce_still_gets_baseline(
    random_walk: tuple[Bars, BacktestResult],
) -> None:
    bars, result = random_walk
    log = TradeLog.from_trades(result.trades, bars)
    # With max_bars=1 the engine closes every trade on its entry bar, so trades that
    # lasted longer no longer reproduce.
    report = audit(bars, log, max_bars=1, n_sims=N_SIMS)
    rand = report.check("random_entry")
    assert rand.status == "SKIP"
    assert "does not reproduce" in rand.summary
    baseline = report.check("no_skill_baseline")
    assert baseline.status == "WARN"
    reference = audit(bars, result, n_sims=N_SIMS).check("no_skill_baseline")
    assert baseline.details["win_rate"] == pytest.approx(reference.details["win_rate"])


def test_phantom_fill_fails() -> None:
    bars = synthetic.random_walk_bars(50, seed=3)
    j = 10
    log = TradeLog.from_records(
        [
            {
                "side": 1,
                "entry_index": 5,
                "exit_index": j,
                "entry_price": float(bars.open[5]),
                "exit_price": float(bars.high[j]) + 1.0,
            }
        ],
        bars,
    )
    report = audit(bars, log, n_sims=10)
    fill = report.check("fill_realism")
    assert fill.status == "FAIL"
    assert "phantom_exit_fill" in fill.summary
    assert report.worst_status == "FAIL"


def test_no_trades_skips_everything() -> None:
    bars = synthetic.random_walk_bars(30, seed=0)
    report = audit(bars, [], n_sims=10)
    assert all(c.status == "SKIP" for c in report.checks)
    assert report.worst_status == "SKIP"


def test_output_formats(planted: tuple[Bars, BacktestResult]) -> None:
    bars, result = planted
    report = audit(bars, result, n_sims=50, periods_per_year=250)
    data = json.loads(report.to_json())
    assert data["worst_status"] == report.worst_status
    assert [c["name"] for c in data["checks"]] == list(CHECK_TITLES)
    assert data["thresholds"]["alpha"] == THRESHOLDS["alpha"]
    assert data["checks"][-1]["details"]["deflated_sharpe_ratio"] is None
    text = report.to_text()
    md = report.to_markdown()
    for out in (text, md):
        assert "deflated Sharpe ratio is not computed" in out
        assert "TrialRegistry" in out
        assert "!" not in out
    assert md.startswith("# Backtest audit")
    assert "| Fill realism |" in md
    assert "annualised" in report.check("sharpe").summary
    assert str(report) == text


def test_details_are_plain_json() -> None:
    check = CheckResult(
        "sharpe", "WARN", "x.", {"a": np.float64(math.inf), "b": (np.int64(3),), "c": np.nan}
    )
    assert check.details == {"a": None, "b": [3], "c": None}
    with pytest.raises(ValueError, match="status"):
        CheckResult("x", "BAD", "x.")  # type: ignore[arg-type]
    report = AuditReport(checks=(check,), n_trades=0, n_bars=0, source="engine trades")
    json.loads(report.to_json())


def test_mismatched_inputs_raise(planted: tuple[Bars, BacktestResult]) -> None:
    bars, result = planted
    with pytest.raises(ValueError, match="costs"):
        audit(bars, result, costs=Costs(half_spread=0.01))
    with pytest.raises(ValueError, match="bars"):
        audit(bars[:100], result)
    with pytest.raises(TypeError):
        audit(bars, "trades.csv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_sims"):
        audit(bars, result, n_sims=0)


def test_costs_flow_through(random_walk: tuple[Bars, BacktestResult]) -> None:
    bars, _ = random_walk
    costs = Costs(half_spread=0.01, fee_rate=0.0002, slippage=0.005)
    rng = np.random.default_rng(5)
    side = np.where(rng.random(len(bars)) < 0.05, 1, 0)
    result = _bracket_run(bars, side, costs=costs)
    log = TradeLog.from_trades(result.trades, bars)
    # The right costs reproduce the log; zero costs make honest fills look wrong.
    good = audit(bars, log, costs=costs, max_bars=MAX_BARS, n_sims=50)
    assert good.check("fill_realism").status != "FAIL"
    assert good.check("random_entry").status != "SKIP"
    bad = audit(bars, log, max_bars=MAX_BARS, n_sims=50)
    assert bad.check("random_entry").status == "SKIP"
