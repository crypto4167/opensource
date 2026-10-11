"""A deliberately pessimistic bar-by-bar bracket-order engine.

Most backtest bugs that inflate results are fill bugs: filling at the close of
the bar that produced the signal, filling a stop at the stop price when the
market gapped through it, giving a trade the target when one bar touched both
the stop and the target, or never checking the last bar. This engine fixes the
fill rules once and makes them explicit, so a strategy and its random-entry
controls are scored by exactly the same code.

Fill rules
----------
* ``decide(i, history)`` runs after bar ``i`` closes. ``history`` holds bars
  ``0..i`` only, so the callback cannot read the future even by accident.
* An order fills at the **open of bar i + 1** (``i + 1 + entry_delay`` with a
  delay), plus half the spread plus slippage against the trader.
* If that open is already beyond the stop or the target, the trade is skipped
  and counted in :attr:`BacktestResult.skipped`.
* From the entry bar on, each bar is checked for the stop and the target.
  A stop the market gapped through fills at the bar's open, not the stop
  price. A target fills at the target price, even after a favourable gap.
* When one bar touches both levels the stop is booked (``same_bar="stop"``)
  and the trade is flagged ``ambiguous``.
* ``max_bars`` closes the position at the close of its ``max_bars``-th bar,
  counting the entry bar. Positions still open at the end of the data close
  at the last close, after the last bar has been checked for stop and target.
* Exits pay half the spread; stop, time and end-of-data exits also pay
  slippage. Fees are ``fee_rate`` times the notional of each fill.
* One position at a time, fixed risk of 1R per trade, no compounding.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import Bars
from .trades import Costs, ExitReason, Setup, Trade, parse_setup

__all__ = ["BacktestResult", "Decide", "resolve_bracket", "run", "run_signals"]

Decide = Callable[[int, Bars], Any]
SameBar = Literal["stop", "target"]

_CHUNK = 64


@dataclass(frozen=True)
class BacktestResult:
    """Trades plus a record of every setup the engine refused and why."""

    trades: tuple[Trade, ...]
    skipped: dict[str, int] = field(default_factory=dict)
    n_bars: int = 0
    costs: Costs = field(default_factory=Costs)
    max_bars: int | None = None
    entry_delay: int = 0
    same_bar: SameBar = "stop"

    @property
    def r(self) -> NDArray[np.float64]:
        """Net R of every trade, in trade order."""
        return np.array([t.r for t in self.trades], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.trades)

    def summary(self) -> Any:
        """Shortcut for :func:`backtest_skeptic.metrics.summarize`."""
        from .metrics import summarize

        return summarize(self.trades)


def _first_index(
    values: NDArray[np.float64], start: int, end: int, level: float, below: bool
) -> int:
    """First ``j`` in ``[start, end)`` with ``values[j] <= level`` (or ``>=``), else -1.

    Searches in growing chunks so a trade that resolves quickly costs little
    even on a long series.
    """
    size = _CHUNK
    a = start
    while a < end:
        b = min(end, a + size)
        seg = values[a:b]
        hit = seg <= level if below else seg >= level
        if hit.any():
            return a + int(np.argmax(hit))
        a = b
        size *= 2
    return -1


def _validate_options(
    max_bars: int | None, entry_delay: int, same_bar: str, risk_band: tuple[float, float] | None
) -> None:
    if max_bars is not None and (not isinstance(max_bars, (int, np.integer)) or max_bars < 1):
        raise ValueError(f"max_bars must be a positive integer or None, got {max_bars!r}")
    if not isinstance(entry_delay, (int, np.integer)) or entry_delay < 0:
        raise ValueError(f"entry_delay must be a non-negative integer, got {entry_delay!r}")
    if same_bar not in ("stop", "target"):
        raise ValueError(f"same_bar must be 'stop' or 'target', got {same_bar!r}")
    if risk_band is not None:
        lo, hi = risk_band
        if not (0 <= lo <= hi):
            raise ValueError(f"risk_band must satisfy 0 <= low <= high, got {risk_band!r}")


def _resolve(
    bars: Bars,
    signal_index: int,
    setup: Setup,
    costs: Costs,
    max_bars: int | None,
    entry_delay: int,
    same_bar: SameBar,
    risk_band: tuple[float, float] | None,
) -> Trade | str:
    """Open and close one bracket trade. Returns the trade or a skip reason."""
    n = len(bars)
    e = signal_index + 1 + entry_delay
    if e >= n:
        return "no_bar_to_fill"
    side, stop, target = setup.side, setup.stop, setup.target
    hs, slip = costs.half_spread, costs.slippage

    mid = float(bars.open[e])
    entry = mid + side * (hs + slip)
    risk = side * (entry - stop)
    reward = side * (target - entry)
    if risk <= 0:
        return "stop_beyond_entry"
    if reward <= 0:
        return "target_beyond_entry"
    if risk_band is not None:
        intended = abs(float(bars.close[signal_index]) - stop)
        lo, hi = risk_band
        if intended > 0 and not (lo * intended <= risk <= hi * intended):
            return "risk_band"

    last = n - 1 if max_bars is None else min(n - 1, e + max_bars - 1)
    end = last + 1
    if side > 0:
        js = _first_index(bars.low, e, end, stop, below=True)
        jt = _first_index(bars.high, e, js + 1 if js >= 0 else end, target, below=False)
    else:
        js = _first_index(bars.high, e, end, stop, below=False)
        jt = _first_index(bars.low, e, js + 1 if js >= 0 else end, target, below=True)

    ambiguous = js >= 0 and js == jt
    reason: ExitReason
    if js >= 0 and (jt < 0 or js < jt or (ambiguous and same_bar == "stop")):
        j = js
        reason = "stop"
        o = float(bars.open[j])
        exit_mid = min(stop, o) if side > 0 else max(stop, o)
        exit_fill = exit_mid - side * (hs + slip)
    elif jt >= 0:
        j = jt
        reason = "target"
        exit_fill = target - side * hs
    else:
        j = last
        reached_limit = max_bars is not None and last == e + max_bars - 1
        reason = "time" if reached_limit else "end"
        exit_fill = float(bars.close[j]) - side * (hs + slip)

    fees = costs.fee_rate * (abs(entry) + abs(exit_fill))
    r = (side * (exit_fill - entry) - fees) / risk
    return Trade(
        side=side,
        signal_index=signal_index,
        entry_index=e,
        exit_index=j,
        mid_entry=mid,
        entry_price=entry,
        exit_price=exit_fill,
        stop=stop,
        target=target,
        reason=reason,
        r=r,
        risk=risk,
        reward=reward,
        fees=fees,
        ambiguous=ambiguous,
    )


def resolve_bracket(
    bars: Bars,
    signal_index: int,
    setup: Setup | tuple[Any, Any, Any],
    *,
    costs: Costs | None = None,
    max_bars: int | None = None,
    entry_delay: int = 0,
    same_bar: SameBar = "stop",
    risk_band: tuple[float, float] | None = None,
) -> Trade | None:
    """Simulate a single bracket trade decided at the close of ``signal_index``.

    Returns ``None`` when the setup cannot be filled (no next bar, or the next
    open is already beyond the stop or the target). Ignores any other open
    position, which is what random-entry controls need.
    """
    _validate_options(max_bars, entry_delay, same_bar, risk_band)
    parsed = parse_setup(setup)
    if parsed is None:
        return None
    out = _resolve(
        bars,
        int(signal_index),
        parsed,
        costs or Costs(),
        max_bars,
        entry_delay,
        same_bar,
        risk_band,
    )
    return out if isinstance(out, Trade) else None


def run(
    bars: Bars,
    decide: Decide,
    *,
    costs: Costs | None = None,
    max_bars: int | None = None,
    warmup: int = 0,
    entry_delay: int = 0,
    same_bar: SameBar = "stop",
    risk_band: tuple[float, float] | None = None,
) -> BacktestResult:
    """Walk the bars, asking ``decide(i, history)`` for a setup whenever flat.

    ``history`` is ``bars.prefix(i + 1)``. ``decide`` may return ``None``, a
    :class:`~backtest_skeptic.Setup`, a ``(side, stop, target)`` tuple or a
    mapping (see :func:`~backtest_skeptic.trades.parse_setup`).

    ``risk_band=(low, high)`` skips a setup when the risk measured from the
    actual fill falls outside ``low..high`` times the risk measured from the
    signal bar's close. It is off by default because it filters trades on
    the fill bar's open, which biases the no-skill baseline.
    """
    _validate_options(max_bars, entry_delay, same_bar, risk_band)
    costs = costs or Costs()
    n = len(bars)
    trades: list[Trade] = []
    skipped: Counter[str] = Counter()
    i = max(0, int(warmup))
    last_signal = n - 2 - entry_delay
    while i <= last_signal:
        try:
            setup = parse_setup(decide(i, bars.prefix(i + 1)))
        except ValueError:
            # Non-finite or non-numeric levels, or an unknown side string.
            skipped["invalid_levels"] += 1
            i += 1
            continue
        if setup is None:
            i += 1
            continue
        out = _resolve(bars, i, setup, costs, max_bars, entry_delay, same_bar, risk_band)
        if isinstance(out, Trade):
            trades.append(out)
            i = out.exit_index  # flat after the exit bar closes; may decide again there
        else:
            skipped[out] += 1
            i += 1
    return BacktestResult(
        trades=tuple(trades),
        skipped=dict(skipped),
        n_bars=n,
        costs=costs,
        max_bars=max_bars,
        entry_delay=entry_delay,
        same_bar=same_bar,
    )


def run_signals(
    bars: Bars,
    side: ArrayLike,
    stop: ArrayLike,
    target: ArrayLike,
    *,
    costs: Costs | None = None,
    max_bars: int | None = None,
    warmup: int = 0,
    entry_delay: int = 0,
    same_bar: SameBar = "stop",
    risk_band: tuple[float, float] | None = None,
) -> BacktestResult:
    """Same engine for precomputed signal arrays.

    ``side[i]`` is ``+1``, ``-1`` or ``0`` (no signal) for a decision made at
    the close of bar ``i``; ``stop[i]`` and ``target[i]`` are the bracket
    levels for that decision. Results are identical to :func:`run` with a
    ``decide`` that reads these arrays, only faster.

    Precomputed arrays are where lookahead bugs hide. Check the function that
    builds them with :mod:`backtest_skeptic.lookahead`.
    """
    _validate_options(max_bars, entry_delay, same_bar, risk_band)
    costs = costs or Costs()
    n = len(bars)
    sides = np.asarray(side, dtype=np.float64)
    stops = np.asarray(stop, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    for name, arr in (("side", sides), ("stop", stops), ("target", targets)):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have one value per bar ({n}), got shape {arr.shape}")
    bad = ~np.isin(sides, (-1.0, 0.0, 1.0))
    if bad.any():
        raise ValueError(f"side must be -1, 0 or +1; bad value at bar {int(np.argmax(bad))}")

    trades: list[Trade] = []
    skipped: Counter[str] = Counter()
    free_from = max(0, int(warmup))
    last_signal = n - 2 - entry_delay
    for raw in np.flatnonzero(sides):
        i = int(raw)
        if i < free_from:
            continue
        if i > last_signal:
            break
        if not (np.isfinite(stops[i]) and np.isfinite(targets[i])):
            skipped["invalid_levels"] += 1
            continue
        setup = Setup(int(sides[i]), float(stops[i]), float(targets[i]))
        out = _resolve(bars, i, setup, costs, max_bars, entry_delay, same_bar, risk_band)
        if isinstance(out, Trade):
            trades.append(out)
            free_from = out.exit_index
        else:
            skipped[out] += 1
    return BacktestResult(
        trades=tuple(trades),
        skipped=dict(skipped),
        n_bars=n,
        costs=costs,
        max_bars=max_bars,
        entry_delay=entry_delay,
        same_bar=same_bar,
    )
