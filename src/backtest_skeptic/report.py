"""One-call audit of a backtest: every check the inputs support, in one report.

:func:`audit` takes the bars and the trades of a backtest and runs, in order:

1. **Fill realism** (:func:`~backtest_skeptic.fills.audit_fills`): can the bars
   support every logged fill?
2. **No-skill baseline** (:func:`~backtest_skeptic.baseline.no_skill_test`):
   do the trades hit their targets more often than the bracket geometry
   alone predicts, and more often than costs require?
3. **Random-entry null** (:func:`~backtest_skeptic.random_entry.random_entry_test`):
   does the strategy beat random entries with the same sides, brackets,
   costs and exit rules?
4. **Concentration** (:func:`~backtest_skeptic.robustness.concentration` and
   :func:`~backtest_skeptic.robustness.drop_best`): is the total carried by a
   few days or a few trades?
5. **Stability** (:func:`~backtest_skeptic.robustness.subperiods` by quarter and
   :func:`~backtest_skeptic.robustness.block_bootstrap` of mean R): is the
   result spread over time, and is its mean clear of zero?
6. **Sharpe statistics** (:func:`~backtest_skeptic.stats.probabilistic_sharpe_ratio`
   and :func:`~backtest_skeptic.stats.min_track_record_length`), with each
   trade's result treated as one period's return.

A check the inputs cannot support is reported as ``SKIP`` with the reason.
Each check ends in ``PASS``, ``WARN``, ``FAIL`` or ``SKIP``. Only the fill
audit can ``FAIL``: a fill the bars cannot contain is an error in the
backtest. The other checks are evidence about the edge and can at worst
``WARN``.

The deflated Sharpe ratio is never computed here. It needs the number of
configurations tried before this one was picked, which the trades do not
contain, and a made-up count would give a made-up answer. Record trials with
:class:`~backtest_skeptic.holdout.TrialRegistry` and pass its count and
Sharpe variance to :func:`~backtest_skeptic.stats.deflated_sharpe_ratio` (or
use ``backtest-skeptic dsr``).

Every threshold is a heuristic and lives in :data:`THRESHOLDS`.

Inputs
------
``trades`` can be a :class:`~backtest_skeptic.engine.BacktestResult`, a
sequence of engine :class:`~backtest_skeptic.Trade` objects, or a
:class:`~backtest_skeptic.fills.TradeLog` from any backtester. For a
``TradeLog``, the audit first tries to reproduce every row through this
package's engine with the given ``costs``, ``max_bars`` and ``same_bar``
(signal bar from the row, or the bar before entry when the row has none).
When every row reproduces, the reproduced trades are used and every check
can run. Otherwise the random-entry test is skipped with the first row that
did not reproduce, and the no-skill baseline runs on trades rebuilt from the
logged prices, stops and targets.

Per-trade results are net R (units of initial risk) when every trade has a
stop on the losing side of its entry, and net P&L in price units otherwise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from . import stats
from .baseline import BaselineResult, no_skill_test
from .data import Bars
from .engine import BacktestResult, SameBar, resolve_bracket
from .fills import TradeLog, TradeRow, audit_fills
from .random_entry import random_entry_test
from .robustness import block_bootstrap, concentration, drop_best, period_keys, subperiods
from .trades import Costs, ExitReason, Trade

__all__ = ["CHECK_TITLES", "STATUSES", "THRESHOLDS", "AuditReport", "CheckResult", "audit"]

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]

STATUSES: tuple[Status, ...] = ("SKIP", "PASS", "WARN", "FAIL")
"""Statuses from least to most severe."""

THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        # Significance level for the no-skill and random-entry p-values.
        "alpha": 0.05,
        # Above this share of time and end-of-data exits, the baseline is judged
        # on its all-exits score, which has no exclusion bias. no_skill_test
        # warns at the same share.
        "unresolved_share": 0.10,
        # Concentration: the best `top_days` exit days, or the best
        # `top_trade_fraction` of trades, producing more than `max_top_share`
        # of the total (1.0 = the whole total, so everything else lost money).
        "top_days": 5,
        "top_trade_fraction": 0.10,
        "max_top_share": 1.0,
        # Drop-best: fraction of trades removed before checking the sign of the mean.
        "drop_best_fraction": 0.05,
        # Stability: at least this share of calendar quarters must be positive ...
        "min_positive_quarters": 0.60,
        # ... and no quarter may produce more than this share of a positive total.
        "max_quarter_share": 0.50,
        # Coverage of the block-bootstrap interval for mean R.
        "bootstrap_ci": 0.95,
        # Sharpe: PSR against zero must reach this ...
        "min_psr": 0.95,
        # ... and the trade count must reach the MinTRL at this confidence.
        "min_trl_confidence": 0.95,
    }
)
"""Heuristic thresholds used by :func:`audit`. None has a canonical source."""

CHECK_TITLES: Mapping[str, str] = MappingProxyType(
    {
        "fill_realism": "Fill realism",
        "no_skill_baseline": "No-skill baseline",
        "random_entry": "Random-entry null",
        "concentration": "Concentration",
        "stability": "Stability",
        "sharpe": "Sharpe statistics",
    }
)
"""Display titles of the checks, by name, in the order :func:`audit` runs them."""

_DSR_NOTE = (
    "The deflated Sharpe ratio is not computed: it needs the number of strategy "
    "configurations tried before this one was chosen, which the trades do not record. "
    "Count them with backtest_skeptic.holdout.TrialRegistry and pass the count and the "
    "trials' Sharpe variance to backtest_skeptic.stats.deflated_sharpe_ratio "
    "(or run `backtest-skeptic dsr`)."
)
_THRESHOLD_NOTE = "Thresholds are heuristics; see backtest_skeptic.report.THRESHOLDS."
_SHOWN_ISSUES = 10
_PRICE_RTOL = 1e-9


# ------------------------------------------------------------------ results


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check.

    Attributes
    ----------
    name: identifier, a key of :data:`CHECK_TITLES`.
    status: ``"PASS"``, ``"WARN"``, ``"FAIL"`` or ``"SKIP"``.
    summary: one or two plain sentences with the numbers behind the status.
    details: the figures, as plain JSON-able values (non-finite floats are
        ``None``).
    """

    name: str
    status: Status
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}, got {self.status!r}")
        object.__setattr__(self, "details", _jsonable(dict(self.details)))

    @property
    def title(self) -> str:
        return CHECK_TITLES.get(self.name, self.name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(frozen=True)
class AuditReport:
    """Every check :func:`audit` ran, plus the settings it ran with.

    Attributes
    ----------
    checks: one :class:`CheckResult` per check, in the order they ran.
    n_trades: trades audited.
    n_bars: bars the trades were audited against.
    source: what was passed in (``"backtest result"``, ``"engine trades"``
        or ``"trade log"``).
    settings: costs, ``max_bars``, ``same_bar``, ``n_sims``, ``seed`` and
        ``periods_per_year``.
    notes: report-level remarks, such as why no deflated Sharpe ratio is given.
    """

    checks: tuple[CheckResult, ...]
    n_trades: int
    n_bars: int
    source: str
    settings: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def worst_status(self) -> Status:
        """Most severe status across the checks (``FAIL > WARN > PASS > SKIP``)."""
        return max((c.status for c in self.checks), key=STATUSES.index, default="SKIP")

    def check(self, name: str) -> CheckResult:
        """The check called ``name``."""
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "n_bars": self.n_bars,
            "source": self.source,
            "worst_status": self.worst_status,
            "settings": _jsonable(dict(self.settings)),
            "checks": [c.as_dict() for c in self.checks],
            "notes": list(self.notes),
            "thresholds": dict(THRESHOLDS),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """The report as strict JSON (no NaN or Infinity)."""
        return json.dumps(self.as_dict(), indent=indent, allow_nan=False)

    def to_text(self) -> str:
        """Plain-text report for a terminal."""
        width = max(len(c.title) for c in self.checks) if self.checks else 0
        lines = [
            f"Backtest audit: {self.n_trades} trades on {self.n_bars} bars ({self.source})",
            f"Worst status: {self.worst_status}",
            "",
        ]
        for c in self.checks:
            lines.append(f"[{c.status}] {c.title:<{width}}  {c.summary}")
        lines.append("")
        lines.extend(self.notes)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """GitHub-flavoured Markdown report."""
        lines = [
            "# Backtest audit",
            "",
            f"{self.n_trades} trades on {self.n_bars} bars ({self.source}). "
            f"Worst status: **{self.worst_status}**.",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
        ]
        for c in self.checks:
            summary = c.summary.replace("|", "\\|")
            lines.append(f"| {c.title} | {c.status} | {summary} |")
        if self.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_text()


# ------------------------------------------------------------------ helpers


def _jsonable(value: Any) -> Any:
    """Plain JSON values: numpy scalars to Python, tuples to lists, NaN/inf to None."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _pts(x: float) -> str:
    return f"{100.0 * x:+.1f} pts"


def _share(x: float) -> str:
    """A threshold fraction as a percentage, without trailing zeros (0.1 -> 10%)."""
    return f"{100.0 * x:g}%"


def _p(p: float) -> str:
    return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"


def _num(x: float, unit: str) -> str:
    return f"{x:+.3f}R" if unit == "R" else f"{x:+.4g}"


def _costs_dict(costs: Costs) -> dict[str, float]:
    return {
        "half_spread": costs.half_spread,
        "fee_rate": costs.fee_rate,
        "slippage": costs.slippage,
    }


def _sentence(text: str) -> str:
    """Capitalise the first letter and end with a full stop."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".?":
        text += "."
    return text[0].upper() + text[1:]


def _skip(name: str, reason: str) -> CheckResult:
    return CheckResult(name, "SKIP", "Skipped. " + _sentence(reason))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _PRICE_RTOL * max(1.0, abs(a), abs(b))


# ------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class _Inputs:
    log: TradeLog
    source: str
    costs: Costs
    max_bars: int | None
    same_bar: SameBar
    entry_delay: int | None
    engine_trades: tuple[Trade, ...] | None
    engine_reason: str
    baseline_trades: tuple[Trade, ...] | None
    baseline_reason: str
    values: NDArray[np.float64] | None
    unit: str
    days: NDArray[np.str_] | None
    quarters: NDArray[np.str_] | None
    time_note: str


def _reproduce(
    bars: Bars, log: TradeLog, costs: Costs, max_bars: int | None, same_bar: SameBar
) -> tuple[tuple[Trade, ...] | None, str]:
    """Engine trades for every row, or ``None`` and why not."""
    out: list[Trade] = []
    for k, row in enumerate(log.rows):
        if row.stop is None or row.target is None:
            return None, f"trade {k} has no stop or target, so the engine cannot replay it"
        signal = row.signal_index if row.signal_index is not None else row.entry_index - 1
        delay = row.entry_index - signal - 1
        if signal < 0 or delay < 0:
            return None, f"trade {k} has no signal bar before its entry bar"
        try:
            trade = resolve_bracket(
                bars,
                signal,
                (row.side, row.stop, row.target),
                costs=costs,
                max_bars=max_bars,
                entry_delay=delay,
                same_bar=same_bar,
            )
        except ValueError as exc:
            return None, f"trade {k} cannot be replayed ({exc})"
        if trade is None:
            return None, f"the engine would not fill trade {k} (signal bar {signal})"
        same = (
            trade.entry_index == row.entry_index
            and trade.exit_index == row.exit_index
            and _close(trade.entry_price, row.entry_price)
            and _close(trade.exit_price, row.exit_price)
            and (row.reason is None or row.reason == trade.reason)
        )
        if not same:
            return None, (
                f"trade {k} does not reproduce under the engine with these costs, max_bars "
                f"and same_bar (logged exit at bar {row.exit_index} for "
                f"{row.exit_price:.6g}, engine exit at bar {trade.exit_index} for "
                f"{trade.exit_price:.6g})"
            )
        out.append(trade)
    return tuple(out), ""


def _rebuild(row: TradeRow, costs: Costs) -> Trade:
    """A Trade carrying the logged prices, for the no-skill baseline."""
    assert row.stop is not None and row.target is not None
    side, e, x = row.side, row.entry_price, row.exit_price
    hs, slip = costs.half_spread, costs.slippage
    risk = side * (e - row.stop)
    fees = costs.fee_rate * (abs(e) + abs(x))
    reason: ExitReason
    if row.reason in ("stop", "target", "time", "end"):
        reason = row.reason
    elif row.reason is None and _close(x, row.target - side * hs):
        reason = "target"
    elif row.reason is None and side * (x + side * (hs + slip) - row.stop) <= _PRICE_RTOL * abs(x):
        reason = "stop"
    else:
        # A market exit that reached neither level ("other", or unknown and
        # away from both levels): scored like a time exit.
        reason = "time"
    signal = row.signal_index if row.signal_index is not None else max(0, row.entry_index - 1)
    return Trade(
        side=side,
        signal_index=signal,
        entry_index=row.entry_index,
        exit_index=row.exit_index,
        mid_entry=e - side * (hs + slip),
        entry_price=e,
        exit_price=x,
        stop=row.stop,
        target=row.target,
        reason=reason,
        r=(side * (x - e) - fees) / risk,
        risk=risk,
        reward=side * (row.target - e),
        fees=fees,
    )


def _values(log: TradeLog, costs: Costs) -> tuple[NDArray[np.float64], str]:
    """Net R per trade when every stop is on the losing side, else net P&L."""
    rows = log.rows
    f = costs.fee_rate
    pnl = np.array(
        [
            r.side * (r.exit_price - r.entry_price) - f * (abs(r.entry_price) + abs(r.exit_price))
            for r in rows
        ],
        dtype=np.float64,
    )
    risks = [None if r.stop is None else r.side * (r.entry_price - r.stop) for r in rows]
    if rows and all(risk is not None and risk > 0 for risk in risks):
        return pnl / np.array(risks, dtype=np.float64), "R"
    return pnl, "price"


def _periods(
    bars: Bars, log: TradeLog
) -> tuple[NDArray[np.str_] | None, NDArray[np.str_] | None, str]:
    if bars.time is None:
        return None, None, "the bars have no timestamps"
    if len(log) == 0:
        return None, None, "there are no trades"
    exits = bars.time[np.array([r.exit_index for r in log.rows], dtype=np.intp)]
    try:
        return period_keys(exits, "D"), period_keys(exits, "Q"), ""
    except ValueError as exc:
        return None, None, f"the bar timestamps cannot be read as dates ({exc})"


def _prepare(
    bars: Bars,
    trades: BacktestResult | Sequence[Trade] | TradeLog,
    costs: Costs | None,
    max_bars: int | None,
    same_bar: SameBar,
) -> _Inputs:
    if costs is not None and not isinstance(costs, Costs):
        raise TypeError(f"costs must be a Costs instance or None, got {type(costs).__name__}")
    entry_delay: int | None = None
    engine: tuple[Trade, ...] | None
    engine_reason = ""
    if isinstance(trades, BacktestResult):
        if trades.n_bars != len(bars):
            raise ValueError(
                f"the result was run on {trades.n_bars} bars but {len(bars)} bars were given"
            )
        if costs is not None and costs != trades.costs:
            raise ValueError(
                f"costs={costs!r} differ from the costs the backtest used ({trades.costs!r}); "
                "omit costs when passing a BacktestResult"
            )
        if max_bars is not None and max_bars != trades.max_bars:
            raise ValueError(
                f"max_bars={max_bars} differs from the backtest's max_bars={trades.max_bars}"
            )
        source = "backtest result"
        used_costs, max_bars, same_bar = trades.costs, trades.max_bars, trades.same_bar
        entry_delay = trades.entry_delay
        engine = trades.trades
        log = TradeLog.from_trades(engine, bars)
    elif isinstance(trades, TradeLog):
        if trades.n_bars != len(bars):
            raise ValueError(
                f"the log was built for {trades.n_bars} bars but {len(bars)} bars were given"
            )
        source = "trade log"
        used_costs = costs if costs is not None else Costs()
        log = trades
        engine, engine_reason = _reproduce(bars, log, used_costs, max_bars, same_bar)
    else:
        if isinstance(trades, (str, bytes, Trade)) or not isinstance(trades, Sequence):
            raise TypeError(
                "trades must be a BacktestResult, a sequence of Trade objects or a TradeLog, "
                f"got {type(trades).__name__}"
            )
        items = tuple(trades)
        for k, t in enumerate(items):
            if not isinstance(t, Trade):
                raise TypeError(f"trade {k} is a {type(t).__name__}, not a Trade")
        source = "engine trades"
        used_costs = costs if costs is not None else Costs()
        engine = items
        log = TradeLog.from_trades(items, bars)

    if engine is not None:
        baseline: tuple[Trade, ...] | None = engine
        baseline_reason = ""
    elif any(r.stop is None or r.target is None for r in log.rows):
        baseline = None
        baseline_reason = (
            "the trade log has no stop and target levels, which the no-skill baseline needs"
        )
    else:
        bad = next(
            (
                k
                for k, r in enumerate(log.rows)
                if r.stop is not None and r.side * (r.entry_price - r.stop) <= 0
            ),
            None,
        )
        if bad is not None:
            baseline = None
            baseline_reason = f"trade {bad} has its stop at or beyond its entry fill"
        else:
            baseline = tuple(_rebuild(r, used_costs) for r in log.rows)
            baseline_reason = ""

    if engine is not None and len(engine):
        values: NDArray[np.float64] | None = np.array([t.r for t in engine], dtype=np.float64)
        unit = "R"
    elif len(log):
        values, unit = _values(log, used_costs)
    else:
        values, unit = None, "R"
    days, quarters, time_note = _periods(bars, log)
    return _Inputs(
        log=log,
        source=source,
        costs=used_costs,
        max_bars=max_bars,
        same_bar=same_bar,
        entry_delay=entry_delay,
        engine_trades=engine,
        engine_reason=engine_reason,
        baseline_trades=baseline,
        baseline_reason=baseline_reason,
        values=values,
        unit=unit,
        days=days,
        quarters=quarters,
        time_note=time_note,
    )


# ------------------------------------------------------------------- checks


def _fill_check(bars: Bars, inp: _Inputs) -> CheckResult:
    name = "fill_realism"
    if len(inp.log) == 0:
        return _skip(name, "there are no trades to audit")
    fa = audit_fills(bars, inp.log, costs=inp.costs)
    counts = ", ".join(f"{code} x{n}" for code, n in fa.counts.items())
    details: dict[str, Any] = {
        "n_trades": fa.n_trades,
        "passed": fa.passed,
        "trades_with_fail": fa.trades_with_fail,
        "fail_share": fa.fail_share,
        "n_warnings": len(fa.warnings),
        "counts": dict(fa.counts),
        "ambiguous_exits": fa.ambiguous_exits,
        "ambiguous_target": fa.ambiguous_target,
        "booked_expectancy_r": fa.booked_expectancy_r,
        "worst_expectancy_r": fa.worst_expectancy_r,
        "best_expectancy_r": fa.best_expectancy_r,
        "issues": [str(i) for i in fa.issues[:_SHOWN_ISSUES]],
        "n_issues": len(fa.issues),
    }
    band = ""
    if fa.worst_expectancy_r is not None and fa.best_expectancy_r is not None:
        band = (
            f" Replayed from the bars, expectancy lies between {fa.worst_expectancy_r:+.3f}R "
            f"and {fa.best_expectancy_r:+.3f}R per trade; the log books "
            f"{fa.booked_expectancy_r:+.3f}R."
        )
    if not fa.passed:
        share = _pct(fa.fail_share or 0.0)
        summary = (
            f"{fa.trades_with_fail} of {fa.n_trades} trades ({share}) have fills the bars "
            f"cannot support: {counts}." + band
        )
        return CheckResult(name, "FAIL", summary, details)
    if fa.warnings:
        summary = (
            f"No impossible fills in {fa.n_trades} trades, but {len(fa.warnings)} fills are "
            f"optimistic or cannot be settled from OHLC bars: {counts}." + band
        )
        return CheckResult(name, "WARN", summary, details)
    summary = f"Every fill in {fa.n_trades} trades is consistent with the bars." + band
    return CheckResult(name, "PASS", summary, details)


def _baseline_check(inp: _Inputs) -> CheckResult:
    name = "no_skill_baseline"
    if len(inp.log) == 0:
        return _skip(name, "there are no trades")
    if inp.baseline_trades is None:
        return _skip(name, inp.baseline_reason)
    try:
        b = no_skill_test(inp.baseline_trades, costs=inp.costs)
    except ValueError as exc:
        return _skip(name, str(exc))
    alpha = THRESHOLDS["alpha"]
    all_exits = b.unresolved_share > THRESHOLDS["unresolved_share"]
    p = b.all_exits_p_value if all_exits else b.p_value
    details = _baseline_details(b, all_exits)
    if all_exits:
        head = (
            f"{_pct(b.unresolved_share)} of trades exited on time or at the end of the data, so "
            f"every exit is scored by how far the price got toward the target: edge "
            f"{_pts(b.all_exits_edge)} over no skill across {b.all_exits_n} trades ({_p(p)})."
        )
        margin = (
            f" On the {b.n_resolved} trades that hit a level, the win rate "
            f"{_pct(b.win_rate)} is {_pts(b.margin)} from the break-even rate "
            f"{_pct(b.breakeven_rate)}."
        )
    else:
        head = (
            f"Win rate {_pct(b.win_rate)} over {b.n_resolved} trades that hit the stop or target, "
            f"against a no-skill rate of {_pct(b.no_skill_rate)} ({_p(p)})."
        )
        margin = (
            f" The break-even rate after costs is {_pct(b.breakeven_rate)}, a margin of "
            f"{_pts(b.margin)}."
        )
    if p >= alpha:
        verdict = " Not distinguishable from random entries with the same brackets."
        status: Status = "WARN"
    elif b.margin <= 0:
        verdict = " The entries beat chance but not the break-even rate after costs."
        status = "WARN"
    else:
        verdict = " The entries beat both chance and costs."
        status = "PASS"
    return CheckResult(name, status, head + margin + verdict, details)


def _baseline_details(b: BaselineResult, all_exits: bool) -> dict[str, Any]:
    keys = (
        "n_trades",
        "n_resolved",
        "excluded",
        "wins",
        "win_rate",
        "no_skill_rate",
        "breakeven_rate",
        "skill_edge",
        "margin",
        "p_value",
        "unresolved_share",
        "exclusion_bias",
        "all_exits_n",
        "all_exits_edge",
        "all_exits_p_value",
        "no_skill_expectancy_r",
        "mean_r",
        "warnings",
    )
    full = b.as_dict()
    out = {k: full[k] for k in keys}
    out["scored_on"] = "all_exits" if all_exits else "resolved"
    return out


def _random_entry_check(bars: Bars, inp: _Inputs, n_sims: int, seed: int | None) -> CheckResult:
    name = "random_entry"
    if len(inp.log) == 0:
        return _skip(name, "there are no trades")
    if inp.engine_trades is None:
        return _skip(name, inp.engine_reason)
    try:
        res = random_entry_test(
            bars,
            inp.engine_trades,
            costs=inp.costs,
            max_bars=inp.max_bars,
            same_bar=inp.same_bar,
            entry_delay=inp.entry_delay,
            n_sims=n_sims,
            seed=seed,
        )
    except ValueError as exc:
        return _skip(name, str(exc))
    assert res.p_value is not None
    details = res.as_dict()
    details.pop("null")
    summary = (
        f"Expectancy {res.observed:+.3f}R per trade against {res.null_mean:+.3f}R for random "
        f"entries with the same sides, brackets and costs ({res.n_sims} simulations, "
        f"{_p(res.p_value)})."
    )
    if res.p_value < THRESHOLDS["alpha"]:
        return CheckResult(
            name, "PASS", summary + " The entry timing beats random timing.", details
        )
    return CheckResult(name, "WARN", summary + " Not distinguishable from random entries.", details)


def _concentration_check(inp: _Inputs) -> CheckResult:
    name = "concentration"
    x = inp.values
    if x is None:
        return _skip(name, "there are no trades")
    total = float(x.sum())
    unit = inp.unit
    if total <= 0:
        return _skip(
            name,
            f"the total is {_num(total, unit)}, so there is no profit for a few days or trades "
            "to carry",
        )
    if len(x) < 2:
        return _skip(name, "a single trade cannot be checked for concentration")
    max_share = THRESHOLDS["max_top_share"]
    frac = THRESHOLDS["top_trade_fraction"]
    top_days = int(THRESHOLDS["top_days"])
    flags: list[str] = []
    parts: list[str] = []
    details: dict[str, Any] = {"unit": unit, "total": total}

    if inp.days is not None:
        by_day = concentration(x, groups=inp.days, top=(top_days,))
        row = by_day.rows[0]
        details["top_days"] = by_day.as_dict()
        if row.share is not None:
            parts.append(
                f"the best {row.k} of {by_day.n_items} exit days produced {_pct(row.share)} of "
                f"the {_num(total, unit)} total"
            )
            if row.share > max_share:
                flags.append("days")
    else:
        details["top_days"] = None
        parts.append(f"no grouping by day because {inp.time_note}")

    by_trade = concentration(x, top=(frac,))
    trow = by_trade.rows[0]
    details["top_trades"] = by_trade.as_dict()
    if trow.share is not None:
        parts.append(f"the best {_share(frac)} of trades ({trow.k}) produced {_pct(trow.share)}")
        if trow.share > max_share:
            flags.append("trades")

    db = drop_best(x, THRESHOLDS["drop_best_fraction"])
    details["drop_best"] = db.as_dict()
    before, after = db.before.mean, db.after.mean
    assert before is not None and after is not None
    parts.append(
        f"dropping the best {_share(THRESHOLDS['drop_best_fraction'])} of trades ({db.k}) moves "
        f"the mean from {_num(before, unit)} to {_num(after, unit)}"
    )
    if db.sign_flips:
        flags.append("drop_best")
    details["flags"] = flags
    summary = _sentence("; ".join(parts))
    if flags:
        return CheckResult(
            name, "WARN", summary + " The result rests on a few days or trades.", details
        )
    return CheckResult(name, "PASS", summary, details)


def _stability_check(inp: _Inputs, n_boot: int, seed: int | None) -> CheckResult:
    name = "stability"
    x = inp.values
    if x is None:
        return _skip(name, "there are no trades")
    if len(x) < 2:
        return _skip(name, "a single trade has no stability to measure")
    unit = inp.unit
    flags: list[str] = []
    parts: list[str] = []
    details: dict[str, Any] = {"unit": unit}

    if inp.quarters is not None:
        sp = subperiods(x, inp.quarters)
        details["quarters"] = sp.as_dict()
        n_pos = sum(1 for p in sp.periods if p.total > 0)
        if sp.n_periods < 2:
            parts.append(
                f"all trades exit in one quarter ({sp.best.key}), so quarters are not compared"
            )
        else:
            part = f"{n_pos} of {sp.n_periods} quarters ({_pct(sp.share_positive)}) were positive"
            if sp.share_positive < THRESHOLDS["min_positive_quarters"]:
                flags.append("positive_quarters")
            if sp.best_share is not None:
                part += (
                    f" and the best quarter ({sp.best.key}) produced {_pct(sp.best_share)} "
                    "of the total"
                )
                if sp.best_share > THRESHOLDS["max_quarter_share"]:
                    flags.append("dominant_quarter")
            parts.append(part)
    else:
        details["quarters"] = None
        parts.append(f"no quarterly breakdown because {inp.time_note}")

    ci = THRESHOLDS["bootstrap_ci"]
    try:
        bs = block_bootstrap(x, n_boot=max(2, n_boot), seed=seed, ci=ci)
    except ValueError as exc:
        details["bootstrap"] = None
        parts.append(f"no bootstrap ({exc})")
    else:
        details["bootstrap"] = bs.as_dict()
        lo, hi = _num(bs.ci_low, unit), _num(bs.ci_high, unit)
        parts.append(
            f"the {100 * ci:g}% block-bootstrap interval for the mean per trade is [{lo}, {hi}]"
        )
        if bs.ci_low <= 0 <= bs.ci_high:
            flags.append("ci_includes_zero")
    details["flags"] = flags
    summary = _sentence("; ".join(parts))
    if flags:
        reasons = []
        if "ci_includes_zero" in flags:
            reasons.append("the interval includes zero")
        if "positive_quarters" in flags:
            reasons.append("too few quarters made money")
        if "dominant_quarter" in flags:
            reasons.append("one quarter carries the result")
        return CheckResult(name, "WARN", summary + " " + _sentence("; ".join(reasons)), details)
    return CheckResult(name, "PASS", summary, details)


def _sharpe_check(inp: _Inputs, periods_per_year: float | None) -> CheckResult:
    name = "sharpe"
    x = inp.values
    if x is None:
        return _skip(name, "there are no trades")
    try:
        sr = stats.sharpe_ratio(x)
        skew, kurt = stats.moments(x)
        psr = stats.probabilistic_sharpe_ratio(x)
        conf = THRESHOLDS["min_trl_confidence"]
        mintrl = stats.min_track_record_length(sr, 0.0, skew, kurt, conf)
    except ValueError as exc:
        return _skip(name, str(exc))
    n = len(x)
    details: dict[str, Any] = {
        "unit": inp.unit,
        "sharpe_per_trade": sr,
        "n": n,
        "skew": skew,
        "kurtosis": kurt,
        "psr": psr,
        "min_track_record_length": mintrl,
        "min_trl_confidence": conf,
        "periods_per_year": periods_per_year,
        "sharpe_annual": None,
        "deflated_sharpe_ratio": None,
        "deflated_sharpe_ratio_note": _DSR_NOTE,
    }
    annual = ""
    if periods_per_year is not None:
        details["sharpe_annual"] = stats.annualize_sharpe(sr, periods_per_year)
        annual = (
            f", {details['sharpe_annual']:.2f} annualised at {periods_per_year:g} trades a year"
        )
    trl = "infinite" if math.isinf(mintrl) else f"{math.ceil(mintrl)} trades"
    summary = (
        f"Sharpe ratio {sr:.3f} per trade{annual} over {n} trades (skew {skew:.2f}, raw "
        f"kurtosis {kurt:.2f}): PSR against zero is {psr:.3f} and the minimum track record "
        f"for {_share(conf)} confidence is {trl}."
    )
    short = n < mintrl
    if psr < THRESHOLDS["min_psr"] or short:
        reasons = []
        if psr < THRESHOLDS["min_psr"]:
            reasons.append(f"PSR is below {THRESHOLDS['min_psr']:g}")
        if short:
            reasons.append(
                "the Sharpe ratio is not above zero"
                if math.isinf(mintrl)
                else "the track record is shorter than the minimum"
            )
        return CheckResult(name, "WARN", summary + " " + _sentence(" and ".join(reasons)), details)
    return CheckResult(name, "PASS", summary, details)


# -------------------------------------------------------------------- audit


def audit(
    bars: Bars,
    trades: BacktestResult | Sequence[Trade] | TradeLog,
    *,
    costs: Costs | None = None,
    max_bars: int | None = None,
    same_bar: SameBar = "stop",
    n_sims: int = 1000,
    seed: int | None = 0,
    periods_per_year: float | None = None,
) -> AuditReport:
    """Run every check the inputs support and collect the results.

    Parameters
    ----------
    bars:
        The bars the backtest ran on.
    trades:
        A :class:`~backtest_skeptic.engine.BacktestResult` (its costs,
        ``max_bars``, ``same_bar`` and entry delay are used), a sequence of
        engine :class:`~backtest_skeptic.Trade` objects, or a
        :class:`~backtest_skeptic.fills.TradeLog` from any backtester.
    costs:
        The costs the backtest charged. Defaults to the result's costs, or to
        zero costs. Passing costs that differ from a result's raises
        ``ValueError``.
    max_bars, same_bar:
        The engine settings the trades were made with, used to reproduce a
        trade log and to resolve random entries. Ignored for a
        ``BacktestResult``, which carries its own (a ``max_bars`` that
        contradicts it raises ``ValueError``).
    n_sims:
        Random-entry simulations, and block-bootstrap resamples.
    seed:
        Seed for the random-entry test and the bootstrap.
    periods_per_year:
        Trades per year, used only to print an annualised Sharpe ratio.

    Returns
    -------
    AuditReport
        One :class:`CheckResult` per check; see the module docstring.

    Raises ``TypeError`` or ``ValueError`` when the inputs do not fit
    together (a log built for other bars, trade indices past the last bar,
    costs that contradict a result).
    """
    if not isinstance(bars, Bars):
        raise TypeError(f"bars must be a Bars instance, got {type(bars).__name__}")
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims < 1:
        raise ValueError(f"n_sims must be a positive integer, got {n_sims!r}")
    if same_bar not in ("stop", "target"):
        raise ValueError(f"same_bar must be 'stop' or 'target', got {same_bar!r}")
    if periods_per_year is not None and not (
        math.isfinite(periods_per_year) and periods_per_year > 0
    ):
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    inp = _prepare(bars, trades, costs, max_bars, same_bar)
    n_sims = int(n_sims)
    runs: tuple[Callable[[], CheckResult], ...] = (
        lambda: _fill_check(bars, inp),
        lambda: _baseline_check(inp),
        lambda: _random_entry_check(bars, inp, n_sims, seed),
        lambda: _concentration_check(inp),
        lambda: _stability_check(inp, n_sims, seed),
        lambda: _sharpe_check(inp, periods_per_year),
    )
    checks = tuple(run() for run in runs)
    notes = [_DSR_NOTE, _THRESHOLD_NOTE]
    if inp.unit == "price":
        notes.insert(
            0,
            "Per-trade results are net P&L in price units, because not every trade has a "
            "stop on the losing side of its entry to define 1R.",
        )
    settings = {
        "costs": _costs_dict(inp.costs),
        "max_bars": inp.max_bars,
        "same_bar": inp.same_bar,
        "n_sims": n_sims,
        "seed": seed,
        "periods_per_year": periods_per_year,
    }
    return AuditReport(
        checks=checks,
        n_trades=len(inp.log),
        n_bars=len(bars),
        source=inp.source,
        settings=settings,
        notes=tuple(notes),
    )
