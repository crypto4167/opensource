"""Fill-realism audit of any trade log against OHLC bars.

A backtest's trade log says, for every trade, at what price it got in and at
what price it got out. Most results that look too good are fill bugs, and
fill bugs leave fingerprints in the log: a price the market never traded, an
order filled before its signal existed, a stop filled at the stop price after
the market gapped through it, an exit logged bars after the price had already
hit the stop, a bar that touched both the stop and the target booked as a win.
This module reads the log next to the bars and points at each one.

It works on trades from **any** backtester, not just this package's engine.
Pass :class:`~backtest_skeptic.Trade` objects, a
:class:`~backtest_skeptic.engine.BacktestResult`, or plain records
(:meth:`TradeLog.from_records`, :meth:`TradeLog.from_csv`) that give each
trade's side, entry and exit price and entry and exit bar, as bar indices or
as timestamps matched against ``bars.time``. Stops, targets, signal bars and
exit reasons are optional; each one that is present enables more checks.

This is a **heuristic audit** with no canonical citation. A ``fail`` is a fill
the bars cannot support under the stated assumptions; a ``warn`` is a fill
that is possible but optimistic, or that the bars cannot settle either way.

Conventions
-----------
Bars are mid prices (see :class:`~backtest_skeptic.Bars`) and logged prices
are actual fills, so the audit first takes the costs back out:

* entry mid = ``entry_price - side * (half_spread + slippage)``;
* exit mid = ``exit_price + side * half_spread``, plus ``side * slippage``
  for every exit except a target exit (a resting limit order).

``costs`` must therefore be the costs the backtest charged. With the wrong
costs, honest fills can land outside the bar and be flagged. Comparisons
allow a floating-point slack of ``1e-9`` times the price plus ``tolerance``
(price units), which you should set to about half a tick if the log rounds
prices.

An entry whose mid equals the entry bar's open is taken to have filled at
the open, so the whole entry bar happened while the position was open. One
that equals the bar's close (and not its open) is taken to have filled at the
close, after the rest of the bar. Anything else is an intrabar fill, for
example a limit or stop entry order. Exits are classified the same way.

Checks
------
1. ``exit_before_entry`` (fail): the exit bar comes before the entry bar.
2. Signal-bar fills, when the signal bar is known. ``entry_before_signal``
   (fail): the entry bar comes before the signal bar. ``phantom_signal_bar_fill``
   (fail): the entry filled on the signal bar at a price other than its
   close; that price traded before the signal it is meant to follow existed.
   ``filled_at_signal_close`` (warn): the entry filled exactly at the close
   the signal was computed from, which assumes an order can be computed and
   filled on the same print. Fill at the next bar's open instead.
3. ``entry_outside_bar`` (fail): the entry mid is outside the entry bar's
   ``[low, high]``. Raised in either direction: a fill the bar cannot contain
   means the log, the bars or the costs do not line up.
4. Exit outside the exit bar's ``[low, high]``: ``phantom_exit_fill`` (fail)
   when better for the trader than any price that traded,
   ``exit_outside_bar`` (warn) when worse. A target exit at the target price
   on a bar that opened beyond the target is allowed (it is conservative). A
   stop exit at the open of a bar that opened beyond the stop is always
   inside the bar.
5. ``gap_stop_fill_too_good`` (fail): a stop exit (reason ``stop``, or an exit
   at the stop level) on a bar that opened beyond the stop, filled better than
   that open. A stop order turns into a market order at the first price
   through it, which is the open. Several popular engines fill such stops at
   the stop price, which hides the gap loss.
   ``stop_fill_too_good`` (fail): a stop exit on a bar that did not open
   beyond the stop, filled better than the stop. A fixed stop triggers at the
   stop and fills there or worse.
   ``target_fill_better_than_target`` (warn): a target exit inside the bar
   (not at its close) at a better price than the target, or than the open
   when the bar opened beyond the target. A resting limit fills at its
   price.
6. ``missed_earlier_touch`` (fail): replaying the bars from the entry bar up
   to the bar before the exit (and the exit bar itself for an exit at its
   close that is not a stop or target exit), some bar traded through the
   stop or the target, so the bracket would have closed the trade earlier
   than the log says. The entry bar is included; an audit that only looks at
   the exit bar passes an engine that skips bars. Also raised when a target is
   booked on a bar that opened through the stop: the stop filled at the open
   first. A level only touched exactly, not traded through, is not flagged,
   because engines differ on whether a touch fills.
   ``possible_entry_bar_touch`` (warn): after an intrabar entry, the entry bar
   traded through a level that the price may have reached before the entry.
   Assuming a continuous price path from the open, a level is reached
   *after* the entry when the entry price lies between the open and the
   level (the path must cross the entry first); such a touch is a fail.
7. ``ambiguous_resolved_optimistically`` (warn): the exit bar touched both
   the stop and the target, opened between them, and the log books the
   target. OHLC data cannot say which level came first. Ambiguous bars booked
   as stops are fine and are counted in :attr:`FillAudit.ambiguous_exits`.

Worst and best case
-------------------
For every trade with a stop, a target and valid geometry (stop and target on
opposite sides of the entry fill), the audit replays the bracket from the
logged entry to the logged exit bar and scores it in R (units of initial risk
``side * (entry_price - stop)``, after costs, with fees ``fee_rate * (|entry|
+ |exit|)`` as in :mod:`~backtest_skeptic.engine`). Stops that gapped fill at
the open, targets fill at the target, bars that opened beyond a level fill
that level first, and a trade no level closed keeps its logged exit price (a
stop or target exit the bars do not confirm becomes a market exit at the exit
bar's close). The two replays differ only where OHLC data cannot decide:

* **worst case**: a bar touching both levels hits the stop first, and a level
  touched on an intrabar entry or exit bar counts when it is the stop;
* **best case**: such a bar hits the target first, and an uncertain touch
  counts when it is the target.

:attr:`FillAudit.booked_expectancy_r`, :attr:`FillAudit.worst_expectancy_r` and
:attr:`FillAudit.best_expectancy_r` are the mean R of the same trades under
the logged prices and the two replays, so the gap between worst and best is
the ambiguity band the bars leave open. A booked figure above the best case
almost always means the log got fills the bars cannot support. One below the
worst case is possible when the log books the stop on a bar that opened
beyond the target, which is pessimistic.

Limitations
-----------
* Bars cannot show the order of prices inside a bar. The audit assumes only
  that the open is the bar's first price and, for intrabar entries, that the
  price moves continuously from the open; anything else that depends on the
  order is a warning or part of the worst/best band, never a fail.
* Stops and targets are taken to be fixed for the life of the trade. For
  trailing stops, scaling in or out, or partial fills, leave out the levels
  (the price checks still run) and use the exit reason ``other`` (the
  reasons ``trailing`` and ``trailing_stop`` are read as ``other``).
* An intrabar fill is checked against the bar's range only; the audit cannot
  tell whether a limit order at that price would really have filled (queue
  position, touch versus trade-through).
* Volume, liquidity and market impact are not modelled.
* A misaligned log (bars shifted by one, wrong time zone) shows up as many
  fails across the checks rather than as a single diagnosis.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .data import Bars
from .engine import BacktestResult
from .trades import Costs, Trade, parse_side

__all__ = [
    "CODES",
    "ExitKind",
    "FillAudit",
    "Issue",
    "Severity",
    "TradeLog",
    "TradeRow",
    "audit_fills",
]

FloatArray = NDArray[np.float64]

Severity = Literal["fail", "warn"]
ExitKind = Literal["stop", "target", "time", "end", "other"]
_Mode = Literal["open", "close", "intrabar"]

_REL = 1e-9  # floating-point slack, relative to the price compared
_SHOWN = 10  # issues listed by FillAudit.__str__

CODES: Mapping[str, tuple[Severity, str]] = MappingProxyType(
    {
        "exit_before_entry": ("fail", "the exit bar comes before the entry bar"),
        "entry_before_signal": (
            "fail",
            "the entry filled on a bar before the bar whose close produced the signal",
        ),
        "phantom_signal_bar_fill": (
            "fail",
            "filled on the signal bar at a price other than its close, which traded "
            "before the signal existed",
        ),
        "filled_at_signal_close": (
            "warn",
            "filled at the very close the signal was computed from",
        ),
        "entry_outside_bar": ("fail", "entry mid price outside the entry bar's low-high range"),
        "phantom_exit_fill": (
            "fail",
            "exit mid price outside the exit bar's range, better for the trader than any "
            "traded price",
        ),
        "exit_outside_bar": (
            "warn",
            "exit mid price outside the exit bar's range, worse for the trader",
        ),
        "gap_stop_fill_too_good": (
            "fail",
            "stop filled better than the open of a bar that gapped through it",
        ),
        "stop_fill_too_good": (
            "fail",
            "stop filled better than the stop on a bar that did not open through it",
        ),
        "target_fill_better_than_target": (
            "warn",
            "target filled inside the bar at a better price than the target",
        ),
        "missed_earlier_touch": (
            "fail",
            "the price traded through the stop or target before the logged exit",
        ),
        "possible_entry_bar_touch": (
            "warn",
            "the entry bar traded through a level, possibly before the intrabar entry",
        ),
        "ambiguous_resolved_optimistically": (
            "warn",
            "the exit bar touched both levels and the log books the target",
        ),
    }
)

_REASONS: dict[str, ExitKind] = {
    "stop": "stop",
    "sl": "stop",
    "stop_loss": "stop",
    "stoploss": "stop",
    "target": "target",
    "tp": "target",
    "take_profit": "target",
    "takeprofit": "target",
    "profit_target": "target",
    "time": "time",
    "timeout": "time",
    "time_exit": "time",
    "max_bars": "time",
    "expiry": "time",
    "end": "end",
    "eod": "end",
    "end_of_data": "end",
    "other": "other",
    "signal": "other",
    "exit_signal": "other",
    "manual": "other",
    "market": "other",
    "trailing": "other",
    "trailing_stop": "other",
}

# Standard field name -> accepted column names (case-insensitive), in order.
_FIELDS: dict[str, tuple[str, ...]] = {
    "side": ("side", "direction"),
    "entry_price": ("entry_price", "entry_fill", "entry"),
    "exit_price": ("exit_price", "exit_fill", "exit"),
    "entry_index": ("entry_index", "entry_bar"),
    "exit_index": ("exit_index", "exit_bar"),
    "signal_index": ("signal_index", "signal_bar"),
    "entry_time": ("entry_time", "entry_timestamp", "entry_datetime", "entry_date"),
    "exit_time": ("exit_time", "exit_timestamp", "exit_datetime", "exit_date"),
    "signal_time": ("signal_time", "signal_timestamp", "signal_datetime", "signal_date"),
    "stop": ("stop", "stop_loss", "sl", "stop_price"),
    "target": ("target", "take_profit", "tp", "target_price"),
    "reason": ("reason", "exit_reason"),
}

_CSV_COLUMNS = (
    "side",
    "signal_index",
    "entry_index",
    "exit_index",
    "entry_price",
    "exit_price",
    "stop",
    "target",
    "reason",
)


# ------------------------------------------------------------------ parsing


def _missing(value: Any) -> bool:
    """None, an empty string or NaN: an optional field that was left out."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _price(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        out = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _index(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{name} must be a bar index, got {value!r}")
    if isinstance(value, (int, np.integer)):
        out = int(value)
    else:
        num = _price(value, name)
        if not num.is_integer():
            raise ValueError(f"{name} must be a whole bar index, got {value!r}")
        out = int(num)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


def _side(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip()
        try:
            return parse_side(float(text))
        except ValueError:
            return parse_side(text)
    return parse_side(value)


def _reason(value: Any) -> ExitKind | None:
    if _missing(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"exit reason must be a string, got {value!r}")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    kind = _REASONS.get(key)
    if kind is None:
        raise ValueError(
            f"unknown exit reason {value!r}; use stop, target, time, end or other "
            "(or leave it empty)"
        )
    return kind


def _time_index(bars: Bars, value: Any, what: str) -> int:
    """Index of the bar whose timestamp equals ``value``."""
    times = bars.time
    if times is None:
        raise ValueError(f"{what}: the bars have no timestamps, so give bar indices instead")
    kind = times.dtype.kind
    key: Any = value.strip() if isinstance(value, str) else value
    try:
        if kind == "M":
            key = np.datetime64(key)
        elif kind in "iuf" and isinstance(key, str):
            key = float(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: cannot read {value!r} as a timestamp ({exc})") from None
    try:
        return bars.index_of(key)
    except (KeyError, TypeError):
        raise ValueError(f"{what}: no bar has timestamp {value!r}") from None


# --------------------------------------------------------------- trade rows


@dataclass(frozen=True)
class TradeRow:
    """One trade as the audit sees it.

    Attributes
    ----------
    side: ``+1`` long, ``-1`` short.
    entry_index, exit_index: bars on which the trade was entered and closed
        (0-based positions in the bars). Equal for a trade opened and closed
        within one bar.
    entry_price, exit_price: actual fills, including spread and slippage.
    signal_index: bar whose close produced the signal, if known.
    stop, target: fixed bracket levels in price units, if known.
    reason: ``"stop"``, ``"target"``, ``"time"``, ``"end"`` or ``"other"``
        (any other market exit), or ``None`` when unknown. With no reason, an
        exit at the target or stop level is treated as a target or stop exit.
    """

    side: int
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    signal_index: int | None = None
    stop: float | None = None
    target: float | None = None
    reason: ExitKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _side(self.side))
        object.__setattr__(self, "entry_index", _index(self.entry_index, "entry_index"))
        object.__setattr__(self, "exit_index", _index(self.exit_index, "exit_index"))
        object.__setattr__(self, "entry_price", _price(self.entry_price, "entry_price"))
        object.__setattr__(self, "exit_price", _price(self.exit_price, "exit_price"))
        if self.signal_index is not None:
            object.__setattr__(self, "signal_index", _index(self.signal_index, "signal_index"))
        for name in ("stop", "target"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if _missing(value) else _price(value, name))
        object.__setattr__(self, "reason", _reason(self.reason))

    @classmethod
    def from_trade(cls, trade: Trade) -> TradeRow:
        """Row for a :class:`~backtest_skeptic.Trade` from this package's engine."""
        return cls(
            side=trade.side,
            entry_index=trade.entry_index,
            exit_index=trade.exit_index,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            signal_index=trade.signal_index,
            stop=trade.stop,
            target=trade.target,
            reason=trade.reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "signal_index": self.signal_index,
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop": self.stop,
            "target": self.target,
            "reason": self.reason,
        }


def _key(name: Any) -> str:
    """Column name for matching: case, spaces and hyphens do not matter."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _lookup(record: Mapping[str, Any], columns: Mapping[str, str]) -> dict[str, Any]:
    """Pick the standard fields out of one record, skipping missing values."""
    lower = {_key(k): k for k in record}
    out: dict[str, Any] = {}
    for std, aliases in _FIELDS.items():
        names = (columns[std],) if std in columns else aliases
        for name in names:
            key = lower.get(_key(name))
            if key is not None:
                value = record[key]
                if not _missing(value):
                    out[std] = value
                break
    return out


def _bar_ref(
    fields: Mapping[str, Any], name: str, bars: Bars, what: str, required: bool
) -> int | None:
    """Bar index from ``<name>_index`` or ``<name>_time``; both must agree if given."""
    raw_index = fields.get(f"{name}_index")
    raw_time = fields.get(f"{name}_time")
    index = None if raw_index is None else _index(raw_index, f"{what}: {name}_index")
    if raw_time is not None:
        from_time = _time_index(bars, raw_time, f"{what}: {name}_time")
        if index is not None and index != from_time:
            raise ValueError(
                f"{what}: {name}_index {index} disagrees with {name}_time {raw_time!r}, "
                f"which is bar {from_time}"
            )
        return from_time
    if index is None and required:
        raise ValueError(f"{what}: needs {name}_index or {name}_time")
    return index


def _row_from_mapping(
    record: Mapping[str, Any], bars: Bars, columns: Mapping[str, str], what: str
) -> TradeRow:
    fields = _lookup(record, columns)
    for std in ("side", "entry_price", "exit_price"):
        if std not in fields:
            looked = (columns[std],) if std in columns else _FIELDS[std]
            raise ValueError(f"{what}: missing {std} (looked for column {' / '.join(looked)})")
    entry = _bar_ref(fields, "entry", bars, what, required=True)
    exit_ = _bar_ref(fields, "exit", bars, what, required=True)
    assert entry is not None and exit_ is not None
    try:
        return TradeRow(
            side=_side(fields["side"]),
            entry_index=entry,
            exit_index=exit_,
            entry_price=fields["entry_price"],
            exit_price=fields["exit_price"],
            signal_index=_bar_ref(fields, "signal", bars, what, required=False),
            stop=fields.get("stop"),
            target=fields.get("target"),
            reason=fields.get("reason"),
        )
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


@dataclass(frozen=True)
class TradeLog:
    """A normalised trade log tied to a set of bars.

    Build it with :meth:`from_trades`, :meth:`from_records` or
    :meth:`from_csv`; every bar reference is checked against ``n_bars``.
    """

    rows: tuple[TradeRow, ...]
    n_bars: int

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        n = _index(self.n_bars, "n_bars")
        for k, row in enumerate(rows):
            if not isinstance(row, TradeRow):
                raise TypeError(f"trade {k}: expected a TradeRow, got {type(row).__name__}")
            for name in ("signal_index", "entry_index", "exit_index"):
                value = getattr(row, name)
                if value is not None and value >= n:
                    raise ValueError(
                        f"trade {k}: {name} {value} is past the last bar ({n - 1}); "
                        "the log does not belong to these bars"
                    )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "n_bars", n)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[TradeRow]:
        return iter(self.rows)

    def __getitem__(self, k: int) -> TradeRow:
        return self.rows[k]

    @classmethod
    def from_trades(cls, trades: Iterable[Trade], bars: Bars) -> TradeLog:
        """Log of :class:`~backtest_skeptic.Trade` objects run on ``bars``."""
        return cls(tuple(TradeRow.from_trade(t) for t in trades), len(bars))

    @classmethod
    def from_records(
        cls,
        records: Iterable[Trade | TradeRow | Mapping[str, Any]],
        bars: Bars,
        *,
        columns: Mapping[str, str] | None = None,
    ) -> TradeLog:
        """Log from mappings (or :class:`~backtest_skeptic.Trade` objects).

        Required per record: ``side`` (``+1/-1``, ``BUY/SELL``, ``long/short``),
        ``entry_price``, ``exit_price``, and the entry and exit bars as
        ``entry_index``/``exit_index`` or as ``entry_time``/``exit_time``
        (matched exactly against ``bars.time``; strings are parsed as
        ``datetime64`` when the bars have datetime timestamps). Optional:
        ``signal_index`` or ``signal_time``, ``stop``, ``target``, ``reason``.
        When both an index and a time are given they must name the same bar.
        Column names are matched case-insensitively and a few common aliases
        are accepted (``stop_loss``/``sl``, ``take_profit``/``tp``,
        ``exit_reason``, ``direction`` ...); ``columns`` maps a standard name to
        the name your log uses, e.g. ``{"entry_price": "AvgEntry"}``. Empty
        strings and NaN count as missing. Other keys are ignored.
        """
        cols = dict(columns or {})
        unknown = sorted(set(cols) - set(_FIELDS))
        if unknown:
            raise ValueError(f"unknown standard field(s) in columns: {', '.join(unknown)}")
        rows: list[TradeRow] = []
        for k, rec in enumerate(records):
            if isinstance(rec, TradeRow):
                rows.append(rec)
            elif isinstance(rec, Trade):
                rows.append(TradeRow.from_trade(rec))
            elif isinstance(rec, Mapping):
                rows.append(_row_from_mapping(rec, bars, cols, f"trade {k}"))
            else:
                raise TypeError(
                    f"trade {k}: expected a Trade or a mapping, got {type(rec).__name__}"
                )
        return cls(tuple(rows), len(bars))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        bars: Bars,
        *,
        columns: Mapping[str, str] | None = None,
    ) -> TradeLog:
        """Log from a CSV file with a header row; same fields as :meth:`from_records`.

        Errors name the CSV line number. :meth:`to_csv` writes a file this
        reads back unchanged.
        """
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: empty file")
            records = list(reader)
        cols = dict(columns or {})
        unknown = sorted(set(cols) - set(_FIELDS))
        if unknown:
            raise ValueError(f"unknown standard field(s) in columns: {', '.join(unknown)}")
        rows = tuple(
            _row_from_mapping(rec, bars, cols, f"{path}, line {line}")
            for line, rec in enumerate(records, start=2)
        )
        return cls(rows, len(bars))

    def to_records(self, bars: Bars | None = None) -> list[dict[str, Any]]:
        """Rows as dicts; with timestamped ``bars``, adds ``*_time`` columns."""
        times = None if bars is None else bars.time
        if bars is not None and len(bars) != self.n_bars:
            raise ValueError(f"bars have {len(bars)} rows but the log was built for {self.n_bars}")
        out: list[dict[str, Any]] = []
        for row in self.rows:
            rec = row.as_dict()
            if times is not None:
                rec["signal_time"] = None if row.signal_index is None else times[row.signal_index]
                rec["entry_time"] = times[row.entry_index]
                rec["exit_time"] = times[row.exit_index]
            out.append(rec)
        return out

    def to_csv(self, path: str | Path, bars: Bars | None = None) -> None:
        """Write the log as CSV (floats at full precision, missing values empty)."""
        records = self.to_records(bars)
        names = list(_CSV_COLUMNS)
        if bars is not None and bars.time is not None:
            names += ["signal_time", "entry_time", "exit_time"]

        def cell(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return repr(value)
            return str(value)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(names)
            for rec in records:
                writer.writerow([cell(rec[name]) for name in names])


# ------------------------------------------------------------------ results


@dataclass(frozen=True)
class Issue:
    """One problem with one trade.

    Attributes
    ----------
    code: what was found; see :data:`CODES`.
    trade: position of the trade in the log (0-based).
    severity: ``"fail"`` (the bars cannot support this fill) or ``"warn"``
        (possible but optimistic, or undecidable from OHLC data).
    message: plain-language explanation with the prices involved.
    bar: the bar the issue is about, when there is one.
    """

    code: str
    trade: int
    severity: Severity
    message: str
    bar: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "trade": self.trade,
            "severity": self.severity,
            "message": self.message,
            "bar": self.bar,
        }

    def __str__(self) -> str:
        where = f"trade {self.trade}" + ("" if self.bar is None else f", bar {self.bar}")
        return f"{where} [{self.severity}] {self.code}: {self.message}"


@dataclass(frozen=True)
class FillAudit:
    """Outcome of :func:`audit_fills`.

    R figures are in units of each trade's initial risk, net of costs.

    Attributes
    ----------
    n_trades: trades audited.
    issues: every issue found, in trade order.
    counts: number of issues per code (codes with no issue are left out).
    trades_with_fail: trades with at least one ``fail``.
    fail_share: ``trades_with_fail / n_trades`` (``None`` for an empty log).
    passed: no ``fail`` anywhere. Warnings do not stop a pass.
    ambiguous_exits: stop or target exits on a bar that touched both levels
        with the order undecidable (the bar opened between them, or it was an
        intrabar entry bar).
    ambiguous_target: of those, the ones booked as target hits.
    checked: how many trades each group of checks could examine:
        ``bar_range`` (checks 1, 3, 4), ``signal_bar`` (needs the signal bar),
        ``gap_stop`` (stop exits after the entry bar, with the stop known),
        ``missed_touch`` (needs a stop or a target), ``ambiguous_exit``
        (stop or target exits with both levels known) and ``rescored``.
    n_rescored: trades scored under the logged prices and both replays.
    unscored: trades left out of the rescoring, by reason: ``no_levels``
        (stop or target unknown), ``exit_before_entry`` and
        ``invalid_geometry`` (stop or target not on its side of the entry fill).
    booked_expectancy_r, worst_expectancy_r, best_expectancy_r: mean R of the
        rescored trades under the logged prices, the worst-case replay and
        the best-case replay (``None`` when nothing was rescored).
    rescored: positions of the rescored trades in the log.
    booked_r, worst_r, best_r: per-trade R, aligned with ``rescored``.
    costs: the costs taken out of the logged fills.
    tolerance: extra price slack allowed in comparisons.
    """

    n_trades: int
    issues: tuple[Issue, ...]
    counts: dict[str, int]
    trades_with_fail: int
    fail_share: float | None
    passed: bool
    ambiguous_exits: int
    ambiguous_target: int
    checked: dict[str, int]
    n_rescored: int
    unscored: dict[str, int]
    booked_expectancy_r: float | None
    worst_expectancy_r: float | None
    best_expectancy_r: float | None
    rescored: tuple[int, ...]
    booked_r: tuple[float, ...]
    worst_r: tuple[float, ...]
    best_r: tuple[float, ...]
    costs: Costs
    tolerance: float

    @property
    def fails(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "fail")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warn")

    def issues_for(self, trade: int) -> tuple[Issue, ...]:
        """Issues raised for the trade at position ``trade`` in the log."""
        return tuple(i for i in self.issues if i.trade == trade)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "issues": [i.as_dict() for i in self.issues],
            "counts": dict(self.counts),
            "trades_with_fail": self.trades_with_fail,
            "fail_share": self.fail_share,
            "passed": self.passed,
            "ambiguous_exits": self.ambiguous_exits,
            "ambiguous_target": self.ambiguous_target,
            "checked": dict(self.checked),
            "n_rescored": self.n_rescored,
            "unscored": dict(self.unscored),
            "booked_expectancy_r": self.booked_expectancy_r,
            "worst_expectancy_r": self.worst_expectancy_r,
            "best_expectancy_r": self.best_expectancy_r,
            "rescored": list(self.rescored),
            "booked_r": list(self.booked_r),
            "worst_r": list(self.worst_r),
            "best_r": list(self.best_r),
            "costs": {
                "half_spread": self.costs.half_spread,
                "fee_rate": self.costs.fee_rate,
                "slippage": self.costs.slippage,
            },
            "tolerance": self.tolerance,
        }

    def __str__(self) -> str:
        if self.passed:
            verdict = "PASSED (no fails)"
        else:
            share = "" if self.fail_share is None else f" ({self.fail_share:.1%})"
            verdict = f"FAILED: {self.trades_with_fail} trade(s){share} with a fail"
        lines = [f"Fill audit (heuristic) of {self.n_trades} trades: {verdict}"]
        if self.counts:
            order = list(CODES)
            parts = [
                f"{CODES[code][0]} {code} x{n}"
                for code, n in sorted(self.counts.items(), key=lambda kv: order.index(kv[0]))
            ]
            lines.append("  issues: " + ", ".join(parts))
        lines.append(
            f"  ambiguous exit bars: {self.ambiguous_exits} "
            f"({self.ambiguous_target} booked as target)"
        )
        if (
            self.booked_expectancy_r is not None
            and self.worst_expectancy_r is not None
            and self.best_expectancy_r is not None
        ):
            lines.append(
                f"  expectancy over {self.n_rescored} rescored trades: booked "
                f"{self.booked_expectancy_r:+.4f}R, worst case {self.worst_expectancy_r:+.4f}R, "
                f"best case {self.best_expectancy_r:+.4f}R"
            )
        else:
            lines.append("  no trade could be rescored (needs a stop and a target)")
        left_out = {k: v for k, v in self.unscored.items() if v}
        if left_out:
            lines.append(
                "  not rescored: "
                + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in left_out.items())
            )
        unchecked = self.n_trades - self.checked.get("signal_bar", 0)
        if unchecked:
            lines.append(
                f"  {unchecked} trade(s) without a signal bar: signal-bar fills not checked"
            )
        for issue in self.issues[:_SHOWN]:
            lines.append(f"  {issue}")
        if len(self.issues) > _SHOWN:
            lines.append(f"  ... and {len(self.issues) - _SHOWN} more issue(s)")
        return "\n".join(lines)


# -------------------------------------------------------------------- audit


def _fmt(x: float) -> str:
    return format(x, ".6g")


@dataclass(frozen=True)
class _Window:
    """Which touches on one bar count while the position is open.

    ``stop``/``target``: 0 ignored, 1 uncertain (may fall outside the time the
    position was open), 2 certain. ``live_open``: the position was open at the
    bar's open, so a gap through a level fills there.
    """

    live_open: bool
    stop: int
    target: int


_LIVE = _Window(True, 2, 2)


class _Auditor:
    def __init__(self, bars: Bars, costs: Costs, tolerance: float) -> None:
        self.o = bars.open
        self.h = bars.high
        self.lo = bars.low
        self.c = bars.close
        self.hs = costs.half_spread
        self.slip = costs.slippage
        self.fee = costs.fee_rate
        self.tol = tolerance
        self.issues: list[Issue] = []
        self.checked: Counter[str] = Counter()
        self.unscored = {"no_levels": 0, "exit_before_entry": 0, "invalid_geometry": 0}
        self.ambiguous = 0
        self.ambiguous_target = 0
        self.rescored: list[int] = []
        self.booked: list[float] = []
        self.worst: list[float] = []
        self.best: list[float] = []

    # ---------------------------------------------------------------- helpers

    def eps(self, ref: float) -> float:
        return self.tol + _REL * abs(ref)

    def near(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.eps(b)

    def add(self, code: str, k: int, message: str, bar: int | None) -> None:
        self.issues.append(Issue(code, k, CODES[code][0], message, bar))

    def mode(self, mid: float, j: int) -> _Mode:
        if self.near(mid, float(self.o[j])):
            return "open"
        if self.near(mid, float(self.c[j])):
            return "close"
        return "intrabar"

    def window(
        self, t: TradeRow, j: int, entry_mode: _Mode, entry_mid: float, exit_mode: _Mode | None
    ) -> _Window | None:
        """Touch window of bar ``j``; ``exit_mode`` is ``None`` for a stop or target exit."""
        e, x = t.entry_index, t.exit_index
        live_open, stop_w, target_w = True, 2, 2
        if j == e:
            if entry_mode == "close":
                return None
            if entry_mode == "intrabar":
                o = float(self.o[j])
                live_open = False
                # The path from the open to a level crosses the entry price
                # first when the entry lies between them.
                stop_w = 2 if t.stop is not None and _between(entry_mid, o, t.stop) else 1
                target_w = 2 if t.target is not None and _between(entry_mid, o, t.target) else 1
        if j == x and exit_mode is not None:
            if exit_mode == "open":
                return None
            if exit_mode == "intrabar":
                stop_w, target_w = min(stop_w, 1), min(target_w, 1)
        return _Window(live_open, stop_w, target_w)

    def through(self, t: TradeRow, j: int) -> tuple[bool, bool]:
        """Did bar ``j`` trade beyond the stop / the target (more than the slack)?"""
        hi, lo = float(self.h[j]), float(self.lo[j])
        s = g = False
        if t.stop is not None:
            s = lo < t.stop - self.eps(t.stop) if t.side > 0 else hi > t.stop + self.eps(t.stop)
        if t.target is not None:
            g = (
                hi > t.target + self.eps(t.target)
                if t.side > 0
                else lo < t.target - self.eps(t.target)
            )
        return s, g

    def first_through(self, t: TradeRow, a: int, b: int) -> int:
        """First bar in ``[a, b)`` that traded beyond a level, else -1."""
        if b <= a:
            return -1
        hit = np.zeros(b - a, dtype=bool)
        hi, lo = self.h[a:b], self.lo[a:b]
        if t.stop is not None:
            ev = self.eps(t.stop)
            hit |= lo < t.stop - ev if t.side > 0 else hi > t.stop + ev
        if t.target is not None:
            ev = self.eps(t.target)
            hit |= hi > t.target + ev if t.side > 0 else lo < t.target - ev
        return a + int(np.argmax(hit)) if hit.any() else -1

    def describe_through(self, t: TradeRow, j: int, level: str) -> str:
        hi, lo = float(self.h[j]), float(self.lo[j])
        value = t.stop if level == "stop" else t.target
        assert value is not None
        low_side = (level == "stop") == (t.side > 0)
        extreme = f"low {_fmt(lo)}" if low_side else f"high {_fmt(hi)}"
        return f"the {level} {_fmt(value)} ({extreme})"

    # ----------------------------------------------------------------- checks

    def audit(self, k: int, t: TradeRow) -> None:
        side, e, x = t.side, t.entry_index, t.exit_index
        hs, slip = self.hs, self.slip
        o, h, lo, c = self.o, self.h, self.lo, self.c
        entry_mid = t.entry_price - side * (hs + slip)
        target_exit = t.reason == "target" or (
            t.reason is None
            and t.target is not None
            and self.near(t.exit_price + side * hs, t.target)
        )
        exit_mid = t.exit_price + side * (hs + (0.0 if target_exit else slip))
        stop_exit = t.reason == "stop" or (
            not target_exit
            and t.reason in (None, "other")
            and t.stop is not None
            and self.near(exit_mid, t.stop)
        )
        level_exit = stop_exit or target_exit
        entry_mode = self.mode(entry_mid, e)
        exit_mode: _Mode | None = None if level_exit else self.mode(exit_mid, x)
        self.checked["bar_range"] += 1

        # 1. exit before entry
        if x < e:
            self.add("exit_before_entry", k, f"exit bar {x} is before entry bar {e}", x)

        # 2. signal-bar fills
        s = t.signal_index
        if s is not None:
            self.checked["signal_bar"] += 1
            if e < s:
                self.add(
                    "entry_before_signal",
                    k,
                    f"entry bar {e} is before signal bar {s}: the order filled before "
                    "its signal existed",
                    e,
                )
            elif e == s:
                close = float(c[s])
                if self.near(entry_mid, close):
                    self.add(
                        "filled_at_signal_close",
                        k,
                        f"entry at the close {_fmt(close)} of signal bar {s}, the very print "
                        "the signal was computed from; fill at the next bar's open instead",
                        s,
                    )
                else:
                    self.add(
                        "phantom_signal_bar_fill",
                        k,
                        f"entry on signal bar {s} at mid {_fmt(entry_mid)}, but the signal only "
                        f"exists after that bar closes at {_fmt(close)}; any other price on "
                        "the signal bar traded before the signal existed",
                        s,
                    )

        # 3. entry inside the entry bar
        low_e, high_e = float(lo[e]), float(h[e])
        above, below = entry_mid > high_e + self.eps(high_e), entry_mid < low_e - self.eps(low_e)
        if above or below:
            better = below if side > 0 else above
            self.add(
                "entry_outside_bar",
                k,
                f"entry mid {_fmt(entry_mid)} (fill {_fmt(t.entry_price)} less costs) is outside "
                f"bar {e}'s range [{_fmt(low_e)}, {_fmt(high_e)}], "
                f"{'better' if better else 'worse'} for the trader than any traded price; "
                "check the bar alignment and that costs match the backtest's",
                e,
            )

        # 4. exit inside the exit bar
        low_x, high_x, open_x = float(lo[x]), float(h[x]), float(o[x])
        above, below = exit_mid > high_x + self.eps(high_x), exit_mid < low_x - self.eps(low_x)
        if above or below:
            gapped_target = (
                target_exit
                and t.target is not None
                and self.near(exit_mid, t.target)
                and side * (open_x - t.target) > 0
            )
            if not gapped_target:
                better = above if side > 0 else below
                where = (
                    f"exit mid {_fmt(exit_mid)} (fill {_fmt(t.exit_price)} with costs taken out) "
                    f"is outside bar {x}'s range [{_fmt(low_x)}, {_fmt(high_x)}]"
                )
                if better:
                    self.add(
                        "phantom_exit_fill",
                        k,
                        f"{where}, better for the trader than any price that traded",
                        x,
                    )
                else:
                    self.add(
                        "exit_outside_bar",
                        k,
                        f"{where}, worse for the trader than any price that traded; check the "
                        "bar alignment and that costs match the backtest's",
                        x,
                    )

        # 5. gapped stop filled better than the open
        if t.stop is not None and stop_exit and x > e:
            self.checked["gap_stop"] += 1
            stop = t.stop
            if side * (open_x - stop) < -self.eps(stop) and side * (exit_mid - open_x) > self.eps(
                open_x
            ):
                self.add(
                    "gap_stop_fill_too_good",
                    k,
                    f"bar {x} opened at {_fmt(open_x)}, through the stop {_fmt(stop)}, but the "
                    f"stop filled at mid {_fmt(exit_mid)}; a stop becomes a market order at "
                    f"the open, so the fill can be no better than {_fmt(open_x)}",
                    x,
                )
        if t.stop is not None and stop_exit:
            stop = t.stop
            gapped = x > e and side * (open_x - stop) < -self.eps(stop)
            if not gapped and side * (exit_mid - stop) > self.eps(stop):
                self.add(
                    "stop_fill_too_good",
                    k,
                    f"the stop {_fmt(stop)} filled at mid {_fmt(exit_mid)} on bar {x}, which "
                    "did not open through it; a fixed stop fills at the stop or worse",
                    x,
                )

        if t.target is not None and target_exit and x >= e:
            # A resting limit fills at the target, or at the open if the bar
            # opened beyond it. An exit at the close may be close-triggered.
            target = t.target
            limit = target if x == e else target + max(0.0, side * (open_x - target)) * side
            if side * (exit_mid - limit) > self.eps(limit) and not self.near(exit_mid, float(c[x])):
                self.add(
                    "target_fill_better_than_target",
                    k,
                    f"the target {_fmt(target)} filled at mid {_fmt(exit_mid)} inside bar {x}; "
                    f"a limit order at the target fills at {_fmt(limit)}",
                    x,
                )

        if x < e:
            self.unscored["exit_before_entry"] += 1
            return

        # 6. a level traded through before the logged exit
        if t.stop is not None or t.target is not None:
            self.checked["missed_touch"] += 1
            self.missed_touch(k, t, entry_mode, entry_mid, exit_mode)

        # 7. exit bar touched both levels
        if t.stop is not None and t.target is not None and level_exit:
            self.checked["ambiguous_exit"] += 1
            self.ambiguity(k, t, entry_mode, target_exit)

        # worst / best rescoring
        if t.stop is None or t.target is None:
            self.unscored["no_levels"] += 1
            return
        risk = side * (t.entry_price - t.stop)
        reward = side * (t.target - t.entry_price)
        if not (risk > 0 and reward > 0):
            self.unscored["invalid_geometry"] += 1
            return
        self.checked["rescored"] += 1
        self.rescored.append(k)
        self.booked.append(self.r(t, t.exit_price, risk))
        for optimistic, out in ((False, self.worst), (True, self.best)):
            fill = self.replay(t, entry_mode, entry_mid, exit_mode, level_exit, optimistic)
            out.append(self.r(t, fill, risk))

    def missed_touch(
        self, k: int, t: TradeRow, entry_mode: _Mode, entry_mid: float, exit_mode: _Mode | None
    ) -> None:
        e, x = t.entry_index, t.exit_index
        # Bars that must not have reached a level: entry .. exit - 1, plus the
        # exit bar itself when a non-bracket exit happened at its close.
        last = x if exit_mode == "close" else x - 1
        if last < e:
            return
        uncertain: str | None = None
        win = self.window(t, e, entry_mode, entry_mid, exit_mode if e == x else None)
        if win is not None:
            s, g = self.through(t, e)
            if s and win.stop == 2:
                self.report_missed(k, t, e, "stop")
                return
            if g and win.target == 2:
                self.report_missed(k, t, e, "target")
                return
            uncertain = (
                "stop" if s and win.stop == 1 else "target" if g and win.target == 1 else None
            )
        j = self.first_through(t, e + 1, last + 1)
        if j >= 0:
            s, _ = self.through(t, j)
            self.report_missed(k, t, j, "stop" if s else "target")
        elif uncertain is not None:
            self.add(
                "possible_entry_bar_touch",
                k,
                f"entry bar {e} traded through {self.describe_through(t, e, uncertain)}, "
                f"possibly after the intrabar entry at mid {_fmt(entry_mid)}; OHLC data cannot "
                f"tell, but if it did the trade closed on bar {e}, not bar {x}",
                e,
            )

    def report_missed(self, k: int, t: TradeRow, j: int, level: str) -> None:
        x = t.exit_index
        when = (
            f"the trade is logged as exiting on bar {x}"
            if j < x
            else f"the trade is logged as exiting at bar {x}'s close"
        )
        self.add(
            "missed_earlier_touch",
            k,
            f"bar {j} traded through {self.describe_through(t, j, level)} but {when}; the "
            "bracket would have closed it earlier",
            j,
        )

    def ambiguity(self, k: int, t: TradeRow, entry_mode: _Mode, target_exit: bool) -> None:
        assert t.stop is not None and t.target is not None
        side, e, x = t.side, t.entry_index, t.exit_index
        stop, target = t.stop, t.target
        hi, lo, op = float(self.h[x]), float(self.lo[x]), float(self.o[x])
        if side > 0:
            hit_s, hit_t = lo <= stop + self.eps(stop), hi >= target - self.eps(target)
        else:
            hit_s, hit_t = hi >= stop - self.eps(stop), lo <= target + self.eps(target)
        if not (hit_s and hit_t):
            return
        if x == e and entry_mode == "close":
            return
        if x > e or entry_mode == "open":
            if side * (op - stop) < -self.eps(stop):
                if target_exit:
                    self.add(
                        "missed_earlier_touch",
                        k,
                        f"bar {x} opened at {_fmt(op)}, through the stop {_fmt(stop)}, so the "
                        f"stop filled at the open before the target {_fmt(target)} could be "
                        "reached; the log books the target",
                        x,
                    )
                return
            if side * (op - target) > self.eps(target):
                return  # the target filled at the open, first
        self.ambiguous += 1
        if target_exit:
            self.ambiguous_target += 1
            self.add(
                "ambiguous_resolved_optimistically",
                k,
                f"bar {x} touched both the stop {_fmt(stop)} and the target {_fmt(target)}; "
                "OHLC data cannot tell which came first and the log books the target",
                x,
            )

    # ------------------------------------------------------------- rescoring

    def r(self, t: TradeRow, exit_fill: float, risk: float) -> float:
        fees = self.fee * (abs(t.entry_price) + abs(exit_fill))
        return (t.side * (exit_fill - t.entry_price) - fees) / risk

    def stop_fill(self, side: int, exit_mid: float) -> float:
        """Fill of a market-style exit (stop, time, end) at mid ``exit_mid``."""
        return exit_mid - side * (self.hs + self.slip)

    def target_fill(self, side: int, target: float) -> float:
        return target - side * self.hs

    def resolve(self, t: TradeRow, j: int, win: _Window, optimistic: bool) -> float | None:
        """Exit fill if the bracket closes the trade on bar ``j``, else ``None``."""
        assert t.stop is not None and t.target is not None
        side, stop, target = t.side, t.stop, t.target
        op, hi, lo = float(self.o[j]), float(self.h[j]), float(self.lo[j])
        if win.live_open:
            if side * (op - stop) <= 0:
                return self.stop_fill(side, op)
            if side * (op - target) >= 0:
                return self.target_fill(side, target)
        if side > 0:
            hit_s, hit_t = lo <= stop, hi >= target
        else:
            hit_s, hit_t = hi >= stop, lo <= target
        hit_s, hit_t = hit_s and win.stop > 0, hit_t and win.target > 0
        if optimistic:
            if hit_t:
                return self.target_fill(side, target)
            if hit_s and win.stop == 2:
                return self.stop_fill(side, stop)
        else:
            if hit_s:
                return self.stop_fill(side, stop)
            if hit_t and win.target == 2:
                return self.target_fill(side, target)
        return None

    def replay(
        self,
        t: TradeRow,
        entry_mode: _Mode,
        entry_mid: float,
        exit_mode: _Mode | None,
        level_exit: bool,
        optimistic: bool,
    ) -> float:
        assert t.stop is not None and t.target is not None
        side, e, x = t.side, t.entry_index, t.exit_index
        win = self.window(t, e, entry_mode, entry_mid, exit_mode if e == x else None)
        if win is not None:
            fill = self.resolve(t, e, win, optimistic)
            if fill is not None:
                return fill
        if x > e:
            # Bars strictly inside the trade are fully live: first touch decides.
            a, b = e + 1, x
            if b > a:
                hi, lo = self.h[a:b], self.lo[a:b]
                if side > 0:
                    hit = (lo <= t.stop) | (hi >= t.target)
                else:
                    hit = (hi >= t.stop) | (lo <= t.target)
                if hit.any():
                    fill = self.resolve(t, a + int(np.argmax(hit)), _LIVE, optimistic)
                    assert fill is not None
                    return fill
            win = self.window(t, x, entry_mode, entry_mid, exit_mode)
            if win is not None:
                fill = self.resolve(t, x, win, optimistic)
                if fill is not None:
                    return fill
        if level_exit:
            # A stop or target exit the bars do not confirm: a market exit at
            # the exit bar's close, with slippage.
            return self.stop_fill(side, float(self.c[x]))
        return t.exit_price


def _between(value: float, a: float, b: float) -> bool:
    return min(a, b) <= value <= max(a, b)


def audit_fills(
    bars: Bars,
    trades: TradeLog | BacktestResult | Iterable[Trade | TradeRow | Mapping[str, Any]],
    *,
    costs: Costs | None = None,
    tolerance: float = 0.0,
) -> FillAudit:
    """Audit every fill in a trade log against the bars it was traded on.

    Parameters
    ----------
    bars:
        The bars the backtest ran on (mid prices). Bar indices in the log are
        positions in these bars.
    trades:
        A :class:`TradeLog`, a :class:`~backtest_skeptic.engine.BacktestResult`,
        or an iterable of :class:`~backtest_skeptic.Trade` objects or
        mappings (see :meth:`TradeLog.from_records`).
    costs:
        The costs the backtest charged, taken back out of each fill before it
        is compared with the bars. Defaults to the result's costs for a
        :class:`~backtest_skeptic.engine.BacktestResult` and to zero costs
        otherwise.
    tolerance:
        Extra price slack (price units, >= 0) in every comparison, on top of
        a relative ``1e-9``. Set it to about half a tick when the log rounds
        prices.

    Returns
    -------
    FillAudit
        Issues per trade, counts per code, the share of trades with a fail,
        and booked, worst-case and best-case expectancy. See the module
        docstring for what each check catches and its limits.
    """
    tol = _price(tolerance, "tolerance")
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0, got {tol}")
    if isinstance(trades, BacktestResult):
        if trades.n_bars != len(bars):
            raise ValueError(
                f"the result was run on {trades.n_bars} bars but {len(bars)} bars were given"
            )
        log = TradeLog.from_trades(trades.trades, bars)
        costs = trades.costs if costs is None else costs
    elif isinstance(trades, TradeLog):
        if trades.n_bars != len(bars):
            raise ValueError(
                f"the log was built for {trades.n_bars} bars but {len(bars)} bars were given"
            )
        log = trades
    else:
        log = TradeLog.from_records(trades, bars)
    costs = Costs() if costs is None else costs
    if not isinstance(costs, Costs):
        raise TypeError(f"costs must be a Costs, got {type(costs).__name__}")

    auditor = _Auditor(bars, costs, tol)
    for k, row in enumerate(log.rows):
        auditor.audit(k, row)

    issues = tuple(auditor.issues)
    counts = Counter(i.code for i in issues)
    failing = {i.trade for i in issues if i.severity == "fail"}
    n = len(log)
    checked = {
        name: auditor.checked[name]
        for name in (
            "bar_range",
            "signal_bar",
            "gap_stop",
            "missed_touch",
            "ambiguous_exit",
            "rescored",
        )
    }

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return FillAudit(
        n_trades=n,
        issues=issues,
        counts={code: counts[code] for code in CODES if counts[code]},
        trades_with_fail=len(failing),
        fail_share=len(failing) / n if n else None,
        passed=not failing,
        ambiguous_exits=auditor.ambiguous,
        ambiguous_target=auditor.ambiguous_target,
        checked=checked,
        n_rescored=len(auditor.rescored),
        unscored=dict(auditor.unscored),
        booked_expectancy_r=mean(auditor.booked),
        worst_expectancy_r=mean(auditor.worst),
        best_expectancy_r=mean(auditor.best),
        rescored=tuple(auditor.rescored),
        booked_r=tuple(auditor.booked),
        worst_r=tuple(auditor.worst),
        best_r=tuple(auditor.best),
        costs=costs,
        tolerance=tol,
    )
