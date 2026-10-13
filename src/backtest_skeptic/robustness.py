"""Robustness heuristics: is the edge spread out, stable and cost-tolerant?

A backtest can pass a significance test and still rest on a handful of
trades, one lucky month, a cost assumption that is slightly too kind, or a
fill one bar earlier than anyone could get. The checks in this module look
for those weaknesses directly.

Most of them are **heuristics**. Concentration, fragility, sub-period
stability, cost stress and entry-delay decay have no canonical reference and
no significance level, and their thresholds (the "more than half the total"
flag in :func:`subperiods`, the collapse ratio in :func:`entry_delay_test`)
are judgement calls stated in the code. Use them to find out where an edge
comes from, not as pass/fail tests. The two statistical tools,
:func:`block_bootstrap` and :func:`clustered_t`, follow published methods.

What each check catches
-----------------------
:func:`concentration` (heuristic)
    Share of the total contributed by the best item, the best five, the best
    10%, optionally after summing values by a group key such as the calendar
    day. Catches a result made by one outlier or one day. A share above 100%
    means everything else lost money. When the total is zero or negative the
    shares are undefined and reported as ``None``; dividing by a negative
    total would print a large negative share that reads as "not concentrated".
:func:`drop_best` (heuristic)
    Mean, total, t-statistic and win share before and after removing the best
    ``k`` values or groups. Catches an expectancy that only exists because of
    the right tail.
:func:`period_keys` and :func:`subperiods` (heuristic)
    Calendar labels (day, ISO week, month, quarter, year) and per-period
    statistics. Catch an edge that lived in one regime: the share of periods
    that made money, the worst and best period, and a flag when one period
    produced more than half the total.
:func:`block_bootstrap`
    Circular moving-block bootstrap of any statistic. Resampling whole blocks
    keeps short-range dependence (streaks, volatility clusters) that an
    ordinary bootstrap destroys, so the interval is honest for serially
    dependent trades. Blocks wrap around the end of the series, so every
    observation is equally likely to be drawn; clamping blocks at the end
    instead would draw the last observations several times as often.
:func:`clustered_t`
    Cluster-robust (CR1) standard error of the mean. Trades that share a day
    or a symbol are not independent; treating them as independent overstates
    the t-statistic, sometimes by a factor of several.
:func:`cost_stress` (heuristic)
    Re-runs the engine with every cost component scaled by each multiplier
    and finds where expectancy crosses zero. It re-simulates rather than
    subtracting a cost from each trade afterwards: costs move the fills, the
    risk that defines 1R, and which setups can be filled at all, and a
    post-hoc adjustment easily charges the spread more than once.
:func:`entry_delay_test` (heuristic)
    Re-runs the engine filling ``d`` bars later. A real edge usually decays
    smoothly; a collapse at the first delay means the result needs an
    immediate fill at the next open, or is using information from the signal
    bar itself.

Units
-----
``values`` are per-trade (or per-period) results, usually net R from the
engine. A :class:`~backtest_skeptic.engine.BacktestResult` or a sequence of
:class:`~backtest_skeptic.Trade` is accepted and read as its R values. Every
function rejects empty input and non-finite values.

Limitations
-----------
* Concentration, drop-best and sub-period statistics describe the sample.
  They say nothing about significance, and a strategy that trades rarely
  is concentrated by construction.
* t-statistics in :func:`drop_best` and :func:`subperiods` treat values as
  independent. Removing the best values after looking at them is a
  selection, so the "after" t-statistic is a stress figure, not a test.
* The block bootstrap assumes the series is stationary and that dependence
  dies out within a block. The default block length ``ceil(n ** (1/3))`` is
  the usual growth rate for variance estimation, not a length tuned to your
  data; try a few lengths. Percentile intervals undercover in small samples.
* CR1 needs a reasonably large number of clusters; with fewer than about 30
  its standard error is itself noisy (Cameron & Miller 2015).
* Cost stress scales every component (spread, fees, slippage) by the same
  factor. If one component dominates your uncertainty, scale it yourself.

References
----------
* Kunsch, H. R. (1989). The jackknife and the bootstrap for general
  stationary observations. *Annals of Statistics* 17(3).
* Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap.
  *Journal of the American Statistical Association* 89(428). The circular
  wrap of the series used here is the one used there.
* Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
  cluster-robust inference. *Journal of Human Resources* 50(2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import Bars
from .engine import BacktestResult, Decide, run, run_signals
from .metrics import t_statistic
from .trades import Costs, Trade

__all__ = [
    "BootstrapResult",
    "ClusteredT",
    "ConcentrationResult",
    "CostRow",
    "CostStressResult",
    "DelayRow",
    "DropBestResult",
    "EntryDelayResult",
    "PeriodStats",
    "SampleStats",
    "Strategy",
    "SubperiodResult",
    "TopShare",
    "block_bootstrap",
    "clustered_t",
    "concentration",
    "cost_stress",
    "drop_best",
    "entry_delay_test",
    "period_keys",
    "subperiods",
]

FloatArray = NDArray[np.float64]

#: A ``decide(i, history)`` callable for :func:`~backtest_skeptic.engine.run`, or a
#: ``(side, stop, target)`` tuple of arrays for :func:`~backtest_skeptic.engine.run_signals`.
Strategy = Decide | tuple[ArrayLike, ArrayLike, ArrayLike]

Freq = Literal["D", "W", "M", "Q", "Y"]
CostStatus = Literal["crossing", "unprofitable_at_lowest", "profitable_at_all", "no_trades"]

_FREQS = ("D", "W", "M", "Q", "Y")
_ENGINE_KEYS = frozenset({"max_bars", "warmup", "entry_delay", "same_bar", "risk_band"})
# Elements per vectorised bootstrap chunk (about 32 MB of float64).
_CHUNK_ELEMENTS = 4_000_000
# Largest number of top items listed in a ConcentrationResult.
_MAX_LISTED = 10
# Largest per-period table printed by SubperiodResult.__str__.
_MAX_TABLE_ROWS = 40
# Heuristic thresholds; none has a canonical source.
_DOMINANT_SHARE = 0.5  # one period produced more than half the total
_EDGE_T = 2.0  # entry_delay_test only looks for decay when the first delay has t >= this
_COLLAPSE_RETENTION = 0.25  # "collapse": expectancy falls below this fraction of the first


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _fmt(x: float | None, spec: str = "+.4f") -> str:
    return "n/a" if x is None else format(x, spec)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _values(values: Any, name: str = "values") -> FloatArray:
    """Per-item results as a finite 1-D float64 array.

    Accepts numbers, a sequence of :class:`Trade` (read as ``t.r``) or a
    :class:`BacktestResult`.
    """
    if isinstance(values, BacktestResult):
        values = values.trades
    items: Any
    if isinstance(values, np.ndarray):
        items = values
    else:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of numbers, got a string")
        try:
            items = list(values)
        except TypeError:
            raise ValueError(
                f"{name} must be a one-dimensional sequence of numbers, got {type(values).__name__}"
            ) from None
        n_trades = sum(isinstance(t, Trade) for t in items)
        if n_trades and n_trades != len(items):
            raise ValueError(f"{name}: pass either Trade objects or plain numbers, not a mix")
        if n_trades:
            items = [t.r for t in items]
    try:
        arr = np.array(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numbers or Trade objects ({exc})") from None
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"{name} is empty")
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(
            f"{name} must be finite; first bad value at position {int(np.argmax(bad))}"
        )
    return arr


def _check_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _is_missing(key: Any) -> bool:
    if key is None:
        return True
    if isinstance(key, (float, np.floating)) and math.isnan(key):
        return True
    return isinstance(key, (np.datetime64, np.timedelta64)) and bool(np.isnat(key))


@dataclass(frozen=True)
class _Groups:
    """Values summed by group key; ``keys is None`` means one item per value."""

    sums: FloatArray
    counts: NDArray[np.int64]
    inverse: NDArray[np.intp]
    keys: NDArray[Any] | None

    def label(self, i: int) -> str:
        return f"#{i}" if self.keys is None else str(self.keys[i])

    @property
    def noun(self) -> str:
        return "value" if self.keys is None else "group"


def _group(values: FloatArray, groups: ArrayLike | None, name: str = "groups") -> _Groups:
    n = len(values)
    if groups is None:
        return _Groups(
            sums=values.copy(),
            counts=np.ones(n, dtype=np.int64),
            inverse=np.arange(n, dtype=np.intp),
            keys=None,
        )
    if isinstance(groups, (str, bytes)):
        raise ValueError(f"{name} must be a sequence with one key per value, got a string")
    keys = np.asarray(groups)
    if keys.ndim != 1 or len(keys) != n:
        raise ValueError(
            f"{name} must be one-dimensional with one key per value ({n}), got shape {keys.shape}"
        )
    if keys.dtype.kind in "fMmO":
        missing = [i for i, k in enumerate(keys) if _is_missing(k)]
        if missing:
            raise ValueError(f"{name} has missing keys (None, NaN or NaT) at position {missing[0]}")
    try:
        unique, inverse = np.unique(keys, return_inverse=True)
    except TypeError:
        raise ValueError(
            f"{name}: keys must be mutually comparable (for example all strings)"
        ) from None
    inv = np.asarray(inverse, dtype=np.intp).ravel()
    g = len(unique)
    return _Groups(
        sums=np.bincount(inv, weights=values, minlength=g).astype(np.float64),
        counts=np.bincount(inv, minlength=g).astype(np.int64),
        inverse=inv,
        keys=unique,
    )


def _parse_count(spec: Any, n_items: int, name: str) -> tuple[int, str]:
    """``(k, label)`` for an int count ``>= 1`` or a fraction in ``(0, 1)``."""
    if isinstance(spec, bool):
        raise ValueError(f"{name} must be an int count or a fraction in (0, 1), got {spec!r}")
    if isinstance(spec, (int, np.integer)):
        k = int(spec)
        if k < 1:
            raise ValueError(f"{name}: a count must be >= 1, got {k}")
        return k, str(k)
    if isinstance(spec, (float, np.floating)):
        frac = float(spec)
        if not 0.0 < frac < 1.0:
            raise ValueError(
                f"{name}: a fraction must be in (0, 1), got {frac}; pass an int for a count"
            )
        # round() first: 0.07 * 100 is 7.000000000000001 and must count as 7, not 8.
        k = max(1, math.ceil(round(frac * n_items, 9)))
        return k, f"{100.0 * frac:g}%"
    raise ValueError(f"{name} must be an int count or a fraction in (0, 1), got {spec!r}")


def _plural(k: int, noun: str) -> str:
    return f"{k} {noun}" + ("" if k == 1 else "s")


# ---------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopShare:
    """One row of :class:`ConcentrationResult`.

    Attributes
    ----------
    label: what was requested, e.g. ``"top 5"`` or ``"top 10%"``.
    k: number of items (values, or groups when grouped) in the top set.
    share: top-``k`` sum divided by the overall total. Above 1 means the
        other items lost money in aggregate. ``None`` when the total is
        ``<= 0``, where a share of the total has no meaning.
    even_share: ``k / n_items``, the share if every item contributed equally.
    total_after: total after dropping the top ``k`` items.
    mean_after: ``total_after`` per remaining value (per trade, also when
        grouped); ``None`` when nothing is left.
    n_after: number of values left.
    """

    label: str
    k: int
    share: float | None
    even_share: float
    total_after: float
    mean_after: float | None
    n_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "k": self.k,
            "share": self.share,
            "even_share": self.even_share,
            "total_after": self.total_after,
            "mean_after": self.mean_after,
            "n_after": self.n_after,
        }


@dataclass(frozen=True)
class ConcentrationResult:
    """How much of the total the best items contributed (a heuristic).

    Attributes
    ----------
    n_values: values passed in.
    n_items: items ranked: ``n_values``, or the number of groups.
    grouped: whether values were summed by a group key first.
    total: sum of all values.
    mean: ``total / n_values``.
    rows: one :class:`TopShare` per requested top-``k``.
    top_items: ``(label, sum)`` of the best items, best first (at most ten,
        and no more than the largest ``k`` requested). Labels are the group
        keys, or ``#i`` for the value at position ``i``.
    notes: plain-language findings.
    """

    n_values: int
    n_items: int
    grouped: bool
    total: float
    mean: float
    rows: tuple[TopShare, ...]
    top_items: tuple[tuple[str, float], ...]
    notes: tuple[str, ...] = ()

    @property
    def defined(self) -> bool:
        """Whether shares of the total are defined (the total is positive)."""
        return self.total > 0

    @property
    def fragile(self) -> bool:
        """The total is positive but dropping some requested top-``k`` leaves it ``<= 0``."""
        return self.total > 0 and any(row.total_after <= 0 for row in self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_values": self.n_values,
            "n_items": self.n_items,
            "grouped": self.grouped,
            "total": self.total,
            "mean": self.mean,
            "defined": self.defined,
            "fragile": self.fragile,
            "rows": [row.as_dict() for row in self.rows],
            "top_items": [list(item) for item in self.top_items],
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        noun = "group" if self.grouped else "value"
        head = f"Concentration (heuristic) of {self.n_values} values"
        if self.grouped:
            head += f" in {self.n_items} groups"
        lines = [head + f": total {self.total:+.4f}, mean {self.mean:+.4f} per value"]
        for row in self.rows:
            if row.label.endswith("%"):
                what = f"top {row.label} ({_plural(row.k, noun)})"
            else:
                what = f"top {_plural(row.k, noun)}"
            share = (
                "share undefined"
                if row.share is None
                else f"share {_pct(row.share)} (even split {_pct(row.even_share)})"
            )
            lines.append(
                f"  {what}: {share}; without {'it' if row.k == 1 else 'them'} total "
                f"{row.total_after:+.4f}, mean {_fmt(row.mean_after)} over "
                f"{_plural(row.n_after, 'value')}"
            )
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def concentration(
    values: Any,
    *,
    groups: ArrayLike | None = None,
    top: Sequence[int | float] | int | float = (1, 5, 0.1),
) -> ConcentrationResult:
    """Share of the total contributed by the best items (a heuristic).

    Parameters
    ----------
    values:
        Per-trade results (R values, a list of :class:`Trade` or a
        :class:`BacktestResult`).
    groups:
        Optional key per value, for example ``period_keys(times, "D")`` to
        rank calendar days instead of trades. Values with the same key are
        summed and the sums are ranked.
    top:
        What to report: an ``int`` ``k`` is the best ``k`` items; a ``float``
        in ``(0, 1)`` is that fraction of the items, rounded up. A ``k``
        larger than the number of items is capped, with a note.

    What it catches: a total made by a few outliers or a few days. Compare
    each ``share`` with ``even_share``; a share above 100% means the rest
    lost money, which :attr:`ConcentrationResult.fragile` flags.

    When the total is ``<= 0`` there is no profit to share out, so every
    ``share`` is ``None`` and a note says so. (Dividing by a negative total
    gives a large negative "share" that looks like no concentration at all.)
    ``total_after`` and ``mean_after`` are still reported.

    Limitations: descriptive only, no significance level. A strategy with a
    positively skewed payoff (small losses, rare large wins) is concentrated
    by design; judge the share against how many large wins you expect.
    """
    x = _values(values)
    grp = _group(x, groups)
    n_items = len(grp.sums)
    specs = [top] if isinstance(top, (int, float, np.integer, np.floating)) else list(top)
    if not specs:
        raise ValueError("top must name at least one count or fraction")
    total = float(x.sum())
    order = np.argsort(-grp.sums, kind="stable")
    ranked = grp.sums[order]
    cum_sum = np.cumsum(ranked)
    cum_count = np.cumsum(grp.counts[order])
    notes: list[str] = []
    rows: list[TopShare] = []
    for spec in specs:
        k, label = _parse_count(spec, n_items, "top")
        if k > n_items:
            notes.append(f"top {label} asks for more than the {n_items} items available; capped")
            k = n_items
        top_sum = float(cum_sum[k - 1])
        n_after = len(x) - int(cum_count[k - 1])
        total_after = total - top_sum
        rows.append(
            TopShare(
                label=label,
                k=k,
                share=top_sum / total if total > 0 else None,
                even_share=k / n_items,
                total_after=total_after,
                mean_after=total_after / n_after if n_after > 0 else None,
                n_after=n_after,
            )
        )
    noun = grp.noun
    if total <= 0:
        notes.append(
            f"the total is {total:+.4f} (not positive), so shares of the total are undefined; "
            "there is no profit for the best items to concentrate"
        )
    else:
        for row in rows:
            if row.total_after <= 0 and row.n_after > 0:
                notes.append(
                    f"without the top {_plural(row.k, noun)} the total is {row.total_after:+.4f}: "
                    "everything else lost money in aggregate"
                )
                break
    n_listed = min(_MAX_LISTED, max(row.k for row in rows))
    top_items = tuple((grp.label(int(order[i])), float(ranked[i])) for i in range(n_listed))
    return ConcentrationResult(
        n_values=len(x),
        n_items=n_items,
        grouped=grp.keys is not None,
        total=total,
        mean=total / len(x),
        rows=tuple(rows),
        top_items=top_items,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# drop the best
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleStats:
    """Basic statistics of a set of values.

    ``t_stat`` is the one-sample t-statistic of the mean against zero
    (:func:`~backtest_skeptic.metrics.t_statistic`), ``None`` for fewer than
    two values; ``win_share`` is the fraction of values above zero.
    """

    n: int
    total: float
    mean: float | None
    t_stat: float | None
    win_share: float | None

    @classmethod
    def of(cls, x: FloatArray) -> SampleStats:
        n = len(x)
        return cls(
            n=n,
            total=float(x.sum()),
            mean=float(x.mean()) if n else None,
            t_stat=t_statistic(x),
            win_share=float((x > 0).mean()) if n else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "total": self.total,
            "mean": self.mean,
            "t_stat": self.t_stat,
            "win_share": self.win_share,
        }

    def __str__(self) -> str:
        return (
            f"n {self.n}, total {self.total:+.4f}, mean {_fmt(self.mean)}, "
            f"t {_fmt(self.t_stat, '+.2f')}, win share {_pct(self.win_share)}"
        )


@dataclass(frozen=True)
class DropBestResult:
    """Statistics before and after removing the best values or groups (a heuristic).

    Attributes
    ----------
    k: number of items removed (values, or groups when grouped).
    grouped: whether the best groups (by group sum) were removed.
    dropped: labels of the removed items, best first (group keys, or ``#i``
        for the value at position ``i``).
    dropped_total: sum of the removed values.
    before, after: :class:`SampleStats` over the individual values, before
        and after the removal.
    notes: plain-language findings.
    """

    k: int
    grouped: bool
    dropped: tuple[str, ...]
    dropped_total: float
    before: SampleStats
    after: SampleStats
    notes: tuple[str, ...] = ()

    @property
    def sign_flips(self) -> bool:
        """The mean was positive and is ``<= 0`` without the best items."""
        b, a = self.before.mean, self.after.mean
        return b is not None and a is not None and b > 0 and a <= 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "grouped": self.grouped,
            "dropped": list(self.dropped),
            "dropped_total": self.dropped_total,
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "sign_flips": self.sign_flips,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        noun = "group" if self.grouped else "value"
        lines = [
            f"Drop the best {_plural(self.k, noun)} (heuristic), "
            f"removing {self.dropped_total:+.4f} in total",
            f"  before: {self.before}",
            f"  after:  {self.after}",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def drop_best(values: Any, k: int | float, *, groups: ArrayLike | None = None) -> DropBestResult:
    """Mean, total, t-statistic and win share without the best ``k`` items (a heuristic).

    Parameters
    ----------
    values:
        Per-trade results (R values, a list of :class:`Trade` or a
        :class:`BacktestResult`).
    k:
        An ``int`` ``>= 1`` removes that many items; a ``float`` in ``(0, 1)``
        removes that fraction of the items, rounded up. At least one item must
        remain.
    groups:
        Optional key per value. The best groups by group sum are removed
        (for example the best calendar days), together with all their values.

    Statistics are always computed over the individual values that remain,
    so ``mean`` is still an expectancy per trade.

    What it catches: an edge carried by its right tail. If the mean turns
    negative after removing the best 1% of trades, the strategy is a lottery
    ticket whose payoff depends on rare events repeating.

    Limitations: removing the best values after seeing them is a selection,
    so the "after" t-statistic is a stress figure, not a test. It treats
    values as independent; use :func:`clustered_t` for grouped data.
    """
    x = _values(values)
    grp = _group(x, groups)
    n_items = len(grp.sums)
    kk, _ = _parse_count(k, n_items, "k")
    if kk >= n_items:
        raise ValueError(f"k removes {kk} of {n_items} {grp.noun}s; at least one must remain")
    order = np.argsort(-grp.sums, kind="stable")
    best = order[:kk]
    removed = np.zeros(n_items, dtype=bool)
    removed[best] = True
    keep = ~removed[grp.inverse]
    before = SampleStats.of(x)
    after = SampleStats.of(x[keep])
    notes: list[str] = []
    noun = grp.noun
    if before.mean is not None and after.mean is not None:
        if before.mean > 0 >= after.mean:
            notes.append(
                f"the mean turns from {before.mean:+.4f} to {after.mean:+.4f} without the best "
                f"{_plural(kk, noun)}: the edge depends on a few outcomes"
            )
        elif before.mean > 0:
            notes.append(
                f"without the best {_plural(kk, noun)} the mean keeps "
                f"{100.0 * after.mean / before.mean:.0f}% of its value"
            )
    return DropBestResult(
        k=kk,
        grouped=grp.keys is not None,
        dropped=tuple(grp.label(int(i)) for i in best),
        dropped_total=float(x[~keep].sum()),
        before=before,
        after=after,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# calendar periods
# ---------------------------------------------------------------------------


def _to_datetime64(times: Any) -> NDArray[np.datetime64]:
    """Convert datetime64 values, datetime/date objects or ISO-8601 strings.

    Timezone-aware datetimes and strings with an offset are converted to UTC
    wall-clock time.
    """
    if isinstance(times, (str, bytes)) or np.ndim(times) != 1:
        raise ValueError("times must be a one-dimensional sequence")
    arr = np.asarray(times)
    if len(arr) == 0:
        return np.array([], dtype="datetime64[us]")
    out: NDArray[np.datetime64]
    if arr.dtype.kind == "M":
        out = arr
    elif arr.dtype.kind in "US":
        out = _parse_strings(arr)
    elif arr.dtype.kind == "O":
        converted: list[np.datetime64] = []
        for i, value in enumerate(arr):
            if isinstance(value, datetime):
                if value.tzinfo is not None and value.utcoffset() is not None:
                    value = value.astimezone(timezone.utc).replace(tzinfo=None)
                converted.append(np.datetime64(value.replace(tzinfo=None), "us"))
            elif isinstance(value, date):
                converted.append(np.datetime64(value, "D").astype("datetime64[us]"))
            elif isinstance(value, np.datetime64):
                converted.append(value.astype("datetime64[us]"))
            elif isinstance(value, str):
                converted.append(_parse_strings(np.array([value]))[0])
            else:
                raise ValueError(
                    f"times: unsupported value {value!r} at position {i}; use datetime64 values, "
                    "datetime/date objects or ISO-8601 strings"
                )
        out = np.array(converted, dtype="datetime64[us]")
    else:
        raise ValueError(
            f"times must be datetime64 values, datetime/date objects or ISO-8601 strings, got "
            f"dtype {arr.dtype}; convert epoch seconds with np.asarray(t, dtype='datetime64[s]')"
        )
    nat = np.isnat(out)
    if nat.any():
        raise ValueError(f"times has a missing value (NaT) at position {int(np.argmax(nat))}")
    return out


_OFFSET = re.compile(r"(?:[Zz]|[+-]\d{2}:?\d{2})$")


def _parse_strings(arr: NDArray[Any]) -> NDArray[np.datetime64]:
    """Parse ISO-8601 strings, converting any UTC offset to UTC.

    Offsets are handled here rather than by numpy, whose support for them is
    deprecated (a warning on numpy 1.x, a different one on 2.x).
    """
    flat = [str(v).strip() for v in arr.ravel()]
    try:
        if not any(_OFFSET.search(v) and "T" in v.upper() for v in flat):
            out: NDArray[np.datetime64] = np.array(flat).astype("datetime64[us]")
            return out.reshape(arr.shape)
        parsed = []
        for v in flat:
            if _OFFSET.search(v) and "T" in v.upper():
                text = v[:-1] + "+00:00" if v[-1] in "Zz" else v
                if text[-3] != ":":  # "+0500" -> "+05:00" for Python 3.10
                    text = text[:-2] + ":" + text[-2:]
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                parsed.append(np.datetime64(dt, "us"))
            else:
                parsed.append(np.datetime64(v, "us"))
        out = np.array(parsed, dtype="datetime64[us]")
    except ValueError as exc:
        raise ValueError(f"times: not an ISO-8601 date or datetime ({exc})") from None
    return out.reshape(arr.shape)


def period_keys(times: Any, freq: Freq) -> NDArray[np.str_]:
    """Calendar label of each timestamp, for grouping trades by period.

    Parameters
    ----------
    times:
        numpy ``datetime64`` values, ``datetime``/``date`` objects or
        ISO-8601 strings. Timezone-aware datetimes and strings with an offset
        (``Z``, ``+02:00``) are converted to UTC first; to group by local
        calendar days, pass local wall-clock times without an offset.
    freq:
        ``"D"`` day (``2024-03-15``), ``"W"`` ISO-8601 week (``2024-W11``,
        weeks start on Monday and week 1 contains the year's first Thursday,
        so 2021-01-03 is in ``2020-W53``), ``"M"`` month (``2024-03``),
        ``"Q"`` quarter (``2024-Q1``) or ``"Y"`` year (``2024``).

    Labels sort chronologically as strings (for years 1000 to 9999). Times
    before 1970 are floored correctly. Raises ``ValueError`` for an unknown
    ``freq``, numeric input, unparseable strings and missing values (NaT).
    """
    if freq not in _FREQS:
        raise ValueError(f"freq must be one of {', '.join(_FREQS)}, got {freq!r}")
    t = _to_datetime64(times)
    if len(t) == 0:
        return np.array([], dtype=np.str_)
    labels: Any
    if freq == "D":
        labels = np.datetime_as_string(t.astype("datetime64[D]"), unit="D")
    elif freq == "M":
        labels = np.datetime_as_string(t.astype("datetime64[M]"), unit="M")
    elif freq == "Y":
        labels = np.datetime_as_string(t.astype("datetime64[Y]"), unit="Y")
    elif freq == "Q":
        months = t.astype("datetime64[M]").astype(np.int64)  # months since 1970-01
        years = months // 12 + 1970
        quarters = (months % 12) // 3 + 1
        labels = [f"{y:04d}-Q{q}" for y, q in zip(years.tolist(), quarters.tolist(), strict=True)]
    else:
        days = t.astype("datetime64[D]").astype(np.int64)  # days since 1970-01-01, a Thursday
        weekday = (days + 3) % 7  # Monday = 0
        thursday = days - weekday + 3  # the ISO week belongs to the year of its Thursday
        iso_year = thursday.astype("datetime64[D]").astype("datetime64[Y]")
        jan1 = iso_year.astype("datetime64[D]").astype(np.int64)
        weeks = (thursday - jan1) // 7 + 1
        years = iso_year.astype(np.int64) + 1970
        labels = [f"{y:04d}-W{w:02d}" for y, w in zip(years.tolist(), weeks.tolist(), strict=True)]
    return np.asarray(labels, dtype=np.str_)


@dataclass(frozen=True)
class PeriodStats:
    """Statistics of one period in :class:`SubperiodResult`.

    ``t_stat`` is ``None`` for a period with one value; ``win_share`` is the
    fraction of the period's values above zero.
    """

    key: str
    n: int
    total: float
    mean: float
    t_stat: float | None
    win_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n": self.n,
            "total": self.total,
            "mean": self.mean,
            "t_stat": self.t_stat,
            "win_share": self.win_share,
        }


@dataclass(frozen=True)
class SubperiodResult:
    """Per-period statistics and how evenly the total is spread (a heuristic).

    Attributes
    ----------
    periods: one :class:`PeriodStats` per key, in sorted key order (which is
        chronological for :func:`period_keys` labels).
    n_values: values passed in.
    total: sum of all values.
    share_positive: fraction of periods whose total is above zero.
    worst, best: periods with the lowest and highest total (first in key
        order on ties).
    best_share: ``best.total / total``; ``None`` when the total is ``<= 0``.
    concentrated: the best period produced more than half of a positive total
        (a heuristic threshold).
    notes: plain-language findings.
    """

    periods: tuple[PeriodStats, ...]
    n_values: int
    total: float
    share_positive: float
    worst: PeriodStats
    best: PeriodStats
    best_share: float | None
    concentrated: bool
    notes: tuple[str, ...] = ()

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    def as_dict(self) -> dict[str, Any]:
        return {
            "periods": [p.as_dict() for p in self.periods],
            "n_values": self.n_values,
            "n_periods": self.n_periods,
            "total": self.total,
            "share_positive": self.share_positive,
            "worst": self.worst.as_dict(),
            "best": self.best.as_dict(),
            "best_share": self.best_share,
            "concentrated": self.concentrated,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        width = max(6, *(len(p.key) for p in self.periods))
        lines = [
            f"Sub-periods (heuristic): {self.n_values} values in {self.n_periods} periods, "
            f"total {self.total:+.4f}, {_pct(self.share_positive)} of periods positive",
            f"  {'period':<{width}}  {'n':>6}  {'total':>10}  {'mean':>9}  {'t':>6}  {'win':>6}",
        ]
        shown = self.periods if self.n_periods <= _MAX_TABLE_ROWS else ()
        for p in shown:
            lines.append(
                f"  {p.key:<{width}}  {p.n:>6}  {p.total:>+10.4f}  {p.mean:>+9.4f}  "
                f"{_fmt(p.t_stat, '+.2f'):>6}  {_pct(p.win_share):>6}"
            )
        if not shown:
            lines[-1] = f"  ({self.n_periods} periods; the table is in .periods)"
        lines.append(
            f"  worst {self.worst.key} ({self.worst.total:+.4f}), "
            f"best {self.best.key} ({self.best.total:+.4f}, {_pct(self.best_share)} of the total)"
        )
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def subperiods(values: Any, keys: ArrayLike) -> SubperiodResult:
    """Statistics per period and how evenly the result is spread (a heuristic).

    Parameters
    ----------
    values:
        Per-trade results (R values, a list of :class:`Trade` or a
        :class:`BacktestResult`).
    keys:
        One period key per value, usually ``period_keys(times, "M")`` with
        the entry or exit time of each trade.

    What it catches: an edge that lived in one regime. A robust edge makes
    money in most periods; one that made more than half its total in a single
    period (``concentrated``) or lost in most periods deserves suspicion.

    Limitations: periods with few values have noisy statistics, and the share
    of positive periods depends on the period length. The 50% flag is a
    heuristic, not a test.
    """
    x = _values(values)
    grp = _group(x, keys, name="keys")
    assert grp.keys is not None
    periods: list[PeriodStats] = []
    for g in range(len(grp.sums)):
        xs = x[grp.inverse == g]
        periods.append(
            PeriodStats(
                key=grp.label(g),
                n=len(xs),
                total=float(xs.sum()),
                mean=float(xs.mean()),
                t_stat=t_statistic(xs),
                win_share=float((xs > 0).mean()),
            )
        )
    totals = np.array([p.total for p in periods])
    total = float(x.sum())
    worst = periods[int(np.argmin(totals))]
    best = periods[int(np.argmax(totals))]
    best_share = best.total / total if total > 0 else None
    concentrated = best_share is not None and best_share > _DOMINANT_SHARE
    share_positive = float((totals > 0).mean())
    notes: list[str] = []
    if len(periods) == 1:
        notes.append("only one period: use a shorter period to see how the result is spread")
    if total <= 0:
        notes.append(
            f"the total is {total:+.4f} (not positive), so no period's share of it is defined"
        )
    elif concentrated:
        notes.append(
            f"period {best.key} produced {_pct(best_share)} of the total; "
            f"without it the rest total {total - best.total:+.4f}"
        )
    if len(periods) > 1 and share_positive < 0.5:
        notes.append(f"only {_pct(share_positive)} of periods made money")
    return SubperiodResult(
        periods=tuple(periods),
        n_values=len(x),
        total=total,
        share_positive=share_positive,
        worst=worst,
        best=best,
        best_share=best_share,
        concentrated=concentrated,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Circular block-bootstrap distribution of a statistic.

    Attributes
    ----------
    statistic: name of the statistic (``"mean"``, ``"sum"``, ``"median"`` or
        the callable's ``__name__``).
    estimate: the statistic on the original values.
    ci_low, ci_high: percentile interval of the bootstrap samples.
    ci: coverage of that interval (0.95 is 95%).
    prob_positive: fraction of bootstrap samples above zero. It describes the
        sampling distribution; it is not a p-value.
    std_error: standard deviation of the bootstrap samples.
    block: block length used.
    n_boot: number of resamples.
    n: number of values.
    seed: the integer seed, if one was given.
    samples: the bootstrap samples (read-only), left out of :meth:`as_dict`
        unless asked for.
    """

    statistic: str
    estimate: float
    ci_low: float
    ci_high: float
    ci: float
    prob_positive: float
    std_error: float
    block: int
    n_boot: int
    n: int
    seed: int | None
    samples: FloatArray = field(repr=False, compare=False)

    def as_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "statistic": self.statistic,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci": self.ci,
            "prob_positive": self.prob_positive,
            "std_error": self.std_error,
            "block": self.block,
            "n_boot": self.n_boot,
            "n": self.n,
            "seed": self.seed,
        }
        if include_samples:
            out["samples"] = self.samples.tolist()
        return out

    def __str__(self) -> str:
        return (
            f"Circular block bootstrap of the {self.statistic} over {self.n} values "
            f"(block {self.block}, {self.n_boot} resamples): estimate {self.estimate:+.4f}, "
            f"{100.0 * self.ci:g}% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"SE {self.std_error:.4f}, {_pct(self.prob_positive)} of resamples above zero"
        )


def _default_block(n: int) -> int:
    """``ceil(n ** (1/3))``, corrected in integer arithmetic.

    The floating-point cube root can land on the wrong side of an integer
    (``(10**18 + 1) ** (1/3)`` rounds to exactly ``1e6``), so the result is
    checked against ``b**3 >= n > (b - 1)**3``.
    """
    b = max(1, round(math.pow(n, 1.0 / 3.0)))
    while b**3 < n:
        b += 1
    while b > 1 and (b - 1) ** 3 >= n:
        b -= 1
    return b


def _scalar(value: Any, name: str) -> float:
    if np.ndim(value) != 0:
        raise ValueError(
            f"statistic {name!r} must return a single number, got shape {np.shape(value)}"
        )
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"statistic {name!r} must return a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"statistic {name!r} returned a non-finite value ({out})")
    return out


def _statistic(
    statistic: str | Callable[[FloatArray], Any],
) -> tuple[str, Callable[[FloatArray], FloatArray]]:
    """Name and a function from a 2-D array of resamples (one per row) to values."""
    if isinstance(statistic, str):

        def mean(d: FloatArray) -> FloatArray:
            return np.asarray(d.mean(axis=1), dtype=np.float64)

        def total(d: FloatArray) -> FloatArray:
            return np.asarray(d.sum(axis=1), dtype=np.float64)

        def median(d: FloatArray) -> FloatArray:
            return np.asarray(np.median(d, axis=1), dtype=np.float64)

        table: dict[str, Callable[[FloatArray], FloatArray]] = {
            "mean": mean,
            "sum": total,
            "median": median,
        }
        if statistic not in table:
            raise ValueError(
                f"statistic must be 'mean', 'sum', 'median' or a callable, got {statistic!r}"
            )
        return statistic, table[statistic]
    if callable(statistic):
        func = statistic
        name = str(getattr(func, "__name__", "custom statistic"))

        def apply(d: FloatArray) -> FloatArray:
            return np.array([_scalar(func(row), name) for row in d], dtype=np.float64)

        return name, apply
    raise ValueError(f"statistic must be 'mean', 'sum', 'median' or a callable, got {statistic!r}")


def block_bootstrap(
    values: Any,
    *,
    block: int | None = None,
    n_boot: int = 10_000,
    seed: int | np.random.Generator | None = 0,
    statistic: str | Callable[[FloatArray], Any] = "mean",
    ci: float = 0.95,
) -> BootstrapResult:
    """Circular moving-block bootstrap of a statistic of ``values``.

    Each resample is built from ``ceil(n / block)`` blocks of ``block``
    consecutive values, each starting at a uniformly random position, and is
    cut to length ``n``. Blocks that run past the end wrap around to the
    start (indices modulo ``n``), as in Politis & Romano (1994), so every
    value is equally likely to appear at every position and the bootstrap
    mean of the mean equals the sample mean. Clamping blocks at the end
    instead gives the last values several times the weight of the others.
    Moving blocks preserve dependence within a block (Kunsch 1989).

    Parameters
    ----------
    values:
        Per-trade or per-period results in time order (R values, a list of
        :class:`Trade` or a :class:`BacktestResult`). At least two.
    block:
        Block length, ``1 <= block <= n``. Default ``ceil(n ** (1/3))``.
        ``block=1`` is the ordinary i.i.d. bootstrap.
    n_boot:
        Number of resamples, at least 2.
    seed:
        Seed or :class:`numpy.random.Generator`; ``None`` is not reproducible.
    statistic:
        ``"mean"`` (default), ``"sum"`` or ``"median"``, computed vectorised;
        or a callable that takes one resample (a 1-D float array) and returns
        a finite number, called once per resample.
    ci:
        Coverage of the percentile interval, in ``(0, 1)``.

    What it catches: a confidence interval that is too narrow because trades
    are serially dependent (winning and losing streaks, volatility regimes).
    Compare with ``block=1``.

    Limitations: assumes stationarity and dependence shorter than a block;
    the default block length is a rule of thumb; percentile intervals
    undercover in small samples and for skewed statistics.
    """
    x = _values(values)
    n = len(x)
    if n < 2:
        raise ValueError("block_bootstrap needs at least two values")
    b = _default_block(n) if block is None else _check_int(block, "block", 1)
    if b > n:
        raise ValueError(f"block must be <= the number of values ({n}), got {b}")
    n_boot = _check_int(n_boot, "n_boot", 2)
    if isinstance(ci, bool) or not isinstance(ci, (int, float, np.floating)) or not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    name, func = _statistic(statistic)
    if isinstance(seed, np.random.Generator):
        rng = seed
        seed_out: int | None = None
    elif seed is None:
        rng = np.random.default_rng()
        seed_out = None
    else:
        seed_out = _check_int(seed, "seed", 0)
        rng = np.random.default_rng(seed_out)

    n_blocks = -(-n // b)
    offsets = np.arange(b)
    samples = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, _CHUNK_ELEMENTS // (n_blocks * b))
    for a in range(0, n_boot, chunk):
        m = min(chunk, n_boot - a)
        starts = rng.integers(0, n, size=(m, n_blocks))
        idx = ((starts[:, :, None] + offsets) % n).reshape(m, n_blocks * b)[:, :n]
        samples[a : a + m] = func(x[idx])
    estimate = float(func(x[None, :])[0])
    lo, hi = np.quantile(samples, [(1.0 - ci) / 2.0, (1.0 + ci) / 2.0])
    samples.setflags(write=False)
    return BootstrapResult(
        statistic=name,
        estimate=estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        ci=float(ci),
        prob_positive=float((samples > 0).mean()),
        std_error=float(samples.std(ddof=1)),
        block=b,
        n_boot=n_boot,
        n=n,
        seed=seed_out,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# cluster-robust t
# ---------------------------------------------------------------------------


class ClusteredT(NamedTuple):
    """``(mean, se, t, n_clusters)`` from :func:`clustered_t`; unpacks like a tuple."""

    mean: float
    se: float
    t: float
    n_clusters: int

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "se": self.se, "t": self.t, "n_clusters": self.n_clusters}

    def __str__(self) -> str:
        return (
            f"mean {self.mean:+.4f}, cluster-robust (CR1) SE {self.se:.4f}, "
            f"t {self.t:+.2f} over {self.n_clusters} clusters"
        )


def clustered_t(values: Any, clusters: ArrayLike) -> ClusteredT:
    """Mean with a cluster-robust (CR1) standard error and t-statistic.

    For the mean ``m`` of ``N`` values in ``G`` clusters, with residuals
    ``e_i = x_i - m`` and cluster sums ``S_g = sum of e_i in cluster g``::

        se = sqrt( G / (G - 1) * sum_g S_g ** 2 ) / N
        t  = m / se

    This is the CR1 estimator (the usual small-sample factor
    ``G/(G-1) * (N-1)/(N-K)`` with ``K = 1`` regressor, the constant); see
    Cameron & Miller (2015). With one value per cluster it equals the
    ordinary standard error ``s / sqrt(N)``. A zero standard error gives
    ``t = +-inf`` for a non-zero mean and ``0`` for a zero mean.

    Parameters
    ----------
    values:
        Per-trade results (R values, a list of :class:`Trade` or a
        :class:`BacktestResult`).
    clusters:
        One key per value: the calendar day, the symbol, the signal that
        opened the trade.

    What it catches: an inflated t-statistic from trades that are not
    independent because they share a day, a symbol or a market event.

    Limitations: requires at least two clusters (raises otherwise) and is
    only reliable with many (roughly 30 or more); with few clusters the
    standard error is noisy and t-tables are too optimistic.
    """
    x = _values(values)
    grp = _group(x, clusters, name="clusters")
    g = len(grp.sums)
    if g < 2:
        raise ValueError(f"clustered_t needs at least 2 clusters, got {g}")
    n = len(x)
    mean = float(x.mean())
    scores = np.bincount(grp.inverse, weights=x - mean, minlength=g)
    se = math.sqrt(g / (g - 1) * float(np.square(scores).sum())) / n
    scale = max(1.0, float(np.abs(x).max()))
    if se <= 1e-12 * scale:
        t = 0.0 if abs(mean) <= 1e-12 * scale else math.copysign(math.inf, mean)
    else:
        t = mean / se
    return ClusteredT(mean=mean, se=se, t=t, n_clusters=g)


# ---------------------------------------------------------------------------
# engine re-runs: cost stress and entry delay
# ---------------------------------------------------------------------------


def _runner(bars: Bars, strategy: Any) -> Callable[[Costs, dict[str, Any]], BacktestResult]:
    if not isinstance(bars, Bars):
        raise ValueError(f"bars must be a Bars instance, got {type(bars).__name__}")
    if callable(strategy):
        decide = strategy

        def go_decide(costs: Costs, kwargs: dict[str, Any]) -> BacktestResult:
            return run(bars, decide, costs=costs, **kwargs)

        return go_decide
    if isinstance(strategy, (tuple, list)) and len(strategy) == 3:
        side, stop, target = strategy

        def go_signals(costs: Costs, kwargs: dict[str, Any]) -> BacktestResult:
            return run_signals(bars, side, stop, target, costs=costs, **kwargs)

        return go_signals
    raise ValueError(
        "strategy must be a decide(i, history) callable or a (side, stop, target) tuple of "
        f"arrays, got {type(strategy).__name__}"
    )


def _engine_kwargs(kwargs: dict[str, Any], forbidden: str | None) -> dict[str, Any]:
    if "costs" in kwargs:
        raise ValueError("pass costs as the costs= argument, not among the engine options")
    if forbidden is not None and forbidden in kwargs:
        raise ValueError(f"{forbidden} is set by this test; do not pass it")
    unknown = sorted(set(kwargs) - _ENGINE_KEYS)
    if unknown:
        allowed = ", ".join(sorted(_ENGINE_KEYS))
        raise ValueError(f"unknown engine option(s) {', '.join(unknown)}; allowed: {allowed}")
    return dict(kwargs)


@dataclass(frozen=True)
class CostRow:
    """One engine run in :class:`CostStressResult`.

    ``expectancy_r`` is mean net R per trade (``None`` with no trades),
    ``total_r`` the sum, ``t_stat`` the t-statistic of the R values and
    ``skipped`` the number of setups the engine could not fill.
    """

    multiplier: float
    n_trades: int
    expectancy_r: float | None
    total_r: float
    t_stat: float | None
    skipped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "n_trades": self.n_trades,
            "expectancy_r": self.expectancy_r,
            "total_r": self.total_r,
            "t_stat": self.t_stat,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class CostStressResult:
    """Expectancy as costs are scaled up and down (a heuristic stress test).

    Attributes
    ----------
    rows: one :class:`CostRow` per multiplier, ascending.
    base_costs: the costs at multiplier 1.
    status: ``"crossing"`` (expectancy crosses zero between two tested
        multipliers), ``"unprofitable_at_lowest"``, ``"profitable_at_all"``
        or ``"no_trades"``.
    breakeven_multiplier: the zero crossing by linear interpolation between
        the bracketing multipliers; ``None`` unless ``status == "crossing"``.
        Never extrapolated beyond the tested range.
    message: one sentence stating the outcome.
    notes: other plain-language findings.
    """

    rows: tuple[CostRow, ...]
    base_costs: Costs
    status: CostStatus
    breakeven_multiplier: float | None
    message: str
    notes: tuple[str, ...] = ()

    def at(self, multiplier: float) -> CostRow:
        """The row for a tested multiplier."""
        for row in self.rows:
            if row.multiplier == multiplier:
                return row
        raise KeyError(f"multiplier {multiplier!r} was not tested")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "base_costs": {
                "half_spread": self.base_costs.half_spread,
                "fee_rate": self.base_costs.fee_rate,
                "slippage": self.base_costs.slippage,
            },
            "status": self.status,
            "breakeven_multiplier": self.breakeven_multiplier,
            "message": self.message,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        c = self.base_costs
        lines = [
            "Cost stress (heuristic): engine re-run with costs x multiplier "
            f"(base: half-spread {c.half_spread:g}, fee rate {c.fee_rate:g}, "
            f"slippage {c.slippage:g})",
            f"  {'mult':>6}  {'trades':>7}  {'expectancy':>11}  {'total R':>10}  {'t':>6}",
        ]
        for row in self.rows:
            lines.append(
                f"  {row.multiplier:>6g}  {row.n_trades:>7}  {_fmt(row.expectancy_r):>11}  "
                f"{row.total_r:>+10.2f}  {_fmt(row.t_stat, '+.2f'):>6}"
            )
        lines.append(f"  {self.message}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _breakeven(rows: Sequence[CostRow]) -> tuple[CostStatus, float | None, str, list[str]]:
    points = [(r.multiplier, r.expectancy_r) for r in rows if r.expectancy_r is not None]
    notes: list[str] = []
    if not points:
        return "no_trades", None, "no trades at any multiplier tested", notes
    m0, e0 = points[0]
    if e0 <= 0:
        return (
            "unprofitable_at_lowest",
            None,
            f"unprofitable at the lowest multiplier tested ({m0:g}x costs, expectancy "
            f"{e0:+.4f}R); no break-even crossing was observed",
            notes,
        )
    for j in range(1, len(points)):
        m1, e1 = points[j - 1]
        m2, e2 = points[j]
        if e2 <= 0:
            m_star = m1 + (m2 - m1) * e1 / (e1 - e2)
            if any(e > 0 for _, e in points[j + 1 :]):
                notes.append(
                    "expectancy is not monotonic in costs (positive again at a higher "
                    "multiplier); the crossing reported is the first one"
                )
            return (
                "crossing",
                m_star,
                f"break-even at about {m_star:.3g}x costs (linear interpolation between "
                f"{m1:g}x and {m2:g}x)",
                notes,
            )
    return (
        "profitable_at_all",
        None,
        f"profitable at every multiplier tested (up to {points[-1][0]:g}x costs); "
        "the break-even multiplier is above that",
        notes,
    )


def cost_stress(
    bars: Bars,
    strategy: Strategy,
    *,
    costs: Costs,
    multipliers: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    **engine_kwargs: Any,
) -> CostStressResult:
    """Re-run the engine with costs scaled by each multiplier (a heuristic stress test).

    Parameters
    ----------
    bars:
        The bars the strategy trades.
    strategy:
        A ``decide(i, history)`` callable (run with
        :func:`~backtest_skeptic.engine.run`) or a ``(side, stop, target)``
        tuple of per-bar arrays (run with
        :func:`~backtest_skeptic.engine.run_signals`).
    costs:
        The costs you expect (multiplier 1). Every component is scaled with
        :meth:`Costs.scaled`.
    multipliers:
        Non-negative, distinct; reported in ascending order.
    **engine_kwargs:
        ``max_bars``, ``warmup``, ``entry_delay``, ``same_bar``,
        ``risk_band``, passed to the engine unchanged.

    Every multiplier is a full engine run. Costs change the fill prices, the
    risk that defines 1R and which setups can be filled, so adjusting the R
    of finished trades afterwards is wrong (and easily charges the spread
    more than once).

    The break-even multiplier is found by linear interpolation between the
    two tested multipliers whose expectancies bracket zero. If expectancy is
    already ``<= 0`` at the lowest multiplier, or still positive at the
    highest, the result says so (``status``, ``message``) instead of
    reporting a crossing that was never observed.

    What it catches: an edge that disappears under slightly worse execution
    than assumed. A break-even below about 1.5x leaves little room for error
    (a heuristic guide, not a rule).
    """
    go = _runner(bars, strategy)
    if not isinstance(costs, Costs):
        raise ValueError(f"costs must be a Costs instance, got {type(costs).__name__}")
    kwargs = _engine_kwargs(engine_kwargs, None)
    mults = _sorted_unique_numbers(multipliers, "multipliers")
    rows: list[CostRow] = []
    for m in mults:
        res = go(costs.scaled(m), kwargs)
        r = res.r
        rows.append(
            CostRow(
                multiplier=m,
                n_trades=len(r),
                expectancy_r=float(r.mean()) if len(r) else None,
                total_r=float(r.sum()),
                t_stat=t_statistic(r),
                skipped=sum(res.skipped.values()),
            )
        )
    status, m_star, message, notes = _breakeven(rows)
    if costs.half_spread == 0 and costs.fee_rate == 0 and costs.slippage == 0:
        notes.append(
            "every cost component is zero, so scaling changes nothing; pass the costs you expect"
        )
    counts = [row.n_trades for row in rows]
    if min(counts) != max(counts):
        notes.append(
            f"the trade count changes with costs ({min(counts)} to {max(counts)}): costs move "
            "fills beyond the stop or target and change which signals arrive while flat"
        )
    no_trade = [row.multiplier for row in rows if row.n_trades == 0]
    if no_trade and len(no_trade) < len(rows):
        notes.append(
            "no trades at multiplier(s) " + ", ".join(f"{m:g}" for m in no_trade) + "; skipped"
        )
    return CostStressResult(
        rows=tuple(rows),
        base_costs=costs,
        status=status,
        breakeven_multiplier=m_star,
        message=message,
        notes=tuple(notes),
    )


def _sorted_unique_numbers(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numbers")
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be numbers, got {v!r}")
        f = float(v)
        if not math.isfinite(f) or f < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
        out.append(f)
    if not out:
        raise ValueError(f"{name} is empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} has duplicates: {list(values)!r}")
    return tuple(sorted(out))


@dataclass(frozen=True)
class DelayRow:
    """One engine run in :class:`EntryDelayResult`.

    ``retention`` is this delay's expectancy divided by the expectancy at the
    smallest delay tested, reported only when that one shows an edge
    (:attr:`EntryDelayResult.edge_at_first`); a ratio to a mean near zero is
    noise.
    """

    delay: int
    n_trades: int
    expectancy_r: float | None
    total_r: float
    t_stat: float | None
    retention: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay": self.delay,
            "n_trades": self.n_trades,
            "expectancy_r": self.expectancy_r,
            "total_r": self.total_r,
            "t_stat": self.t_stat,
            "retention": self.retention,
        }


@dataclass(frozen=True)
class EntryDelayResult:
    """Expectancy and trade count per entry delay (a heuristic).

    Attributes
    ----------
    rows: one :class:`DelayRow` per delay, ascending.
    costs: the costs every run used.
    edge_at_first: the smallest delay shows an edge worth tracking
        (expectancy > 0 and t-statistic >= 2, a heuristic threshold).
    collapse_delay: the first later delay whose expectancy falls below 25% of
        the first one (a heuristic threshold) or has no trades; ``None`` when
        there is no edge to lose or it never collapses.
    notes: plain-language findings.
    """

    rows: tuple[DelayRow, ...]
    costs: Costs
    edge_at_first: bool
    collapse_delay: int | None
    notes: tuple[str, ...] = ()

    def at(self, delay: int) -> DelayRow:
        """The row for a tested delay."""
        for row in self.rows:
            if row.delay == delay:
                return row
        raise KeyError(f"delay {delay!r} was not tested")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "costs": {
                "half_spread": self.costs.half_spread,
                "fee_rate": self.costs.fee_rate,
                "slippage": self.costs.slippage,
            },
            "edge_at_first": self.edge_at_first,
            "collapse_delay": self.collapse_delay,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        lines = [
            "Entry delay (heuristic): engine re-run filling d bars after the next open",
            f"  {'delay':>5}  {'trades':>7}  {'expectancy':>11}  {'total R':>10}  "
            f"{'t':>6}  {'kept':>6}",
        ]
        for row in self.rows:
            lines.append(
                f"  {row.delay:>5}  {row.n_trades:>7}  {_fmt(row.expectancy_r):>11}  "
                f"{row.total_r:>+10.2f}  {_fmt(row.t_stat, '+.2f'):>6}  {_pct(row.retention):>6}"
            )
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def entry_delay_test(
    bars: Bars,
    strategy: Strategy,
    *,
    delays: Sequence[int] = (0, 1, 2, 3),
    costs: Costs | None = None,
    **engine_kwargs: Any,
) -> EntryDelayResult:
    """Re-run the engine with each entry delay (a heuristic).

    Parameters
    ----------
    bars:
        The bars the strategy trades.
    strategy:
        A ``decide(i, history)`` callable or a ``(side, stop, target)``
        tuple of per-bar arrays, as in :func:`cost_stress`.
    delays:
        Distinct non-negative integers. Delay ``d`` fills at the open of bar
        ``i + 1 + d`` for a decision at the close of bar ``i``; the stop and
        target levels stay where the strategy put them.
    costs:
        Costs for every run (default: none).
    **engine_kwargs:
        ``max_bars``, ``warmup``, ``same_bar``, ``risk_band``.

    What it catches: a result that needs an immediate fill. A real edge with
    a horizon of several bars usually decays smoothly as the entry is
    delayed. A collapse at the first delay means the edge lives entirely in
    the first bar after the signal: either the strategy really needs to fill
    at the next open (check that you can), or the signal uses information
    from the signal bar that was not available at its close.

    Limitations: the collapse flag is a heuristic (expectancy below 25% of
    the first delay's, looked for only when the first delay has t >= 2). A
    genuine one-bar edge collapses too; this test cannot tell it from a
    leak, which is what :mod:`backtest_skeptic.lookahead` is for. Delayed
    entries are skipped when the open has already passed the stop or target,
    so the trade count can fall.
    """
    go = _runner(bars, strategy)
    if costs is None:
        costs = Costs()
    elif not isinstance(costs, Costs):
        raise ValueError(f"costs must be a Costs instance or None, got {type(costs).__name__}")
    kwargs = _engine_kwargs(engine_kwargs, "entry_delay")
    if isinstance(delays, (str, bytes)):
        raise ValueError("delays must be a sequence of integers")
    ds = sorted(_check_int(d, "delays", 0) for d in delays)
    if not ds:
        raise ValueError("delays is empty")
    if len(set(ds)) != len(ds):
        raise ValueError(f"delays has duplicates: {list(delays)!r}")

    raw: list[tuple[int, FloatArray]] = []
    for d in ds:
        raw.append((d, go(costs, {**kwargs, "entry_delay": d}).r))
    first_r = raw[0][1]
    first_mean = float(first_r.mean()) if len(first_r) else None
    first_t = t_statistic(first_r)
    edge = bool(first_mean and first_mean > 0 and first_t is not None and first_t >= _EDGE_T)
    rows = tuple(
        DelayRow(
            delay=d,
            n_trades=len(r),
            expectancy_r=float(r.mean()) if len(r) else None,
            total_r=float(r.sum()),
            t_stat=t_statistic(r),
            retention=(float(r.mean()) / first_mean if edge and first_mean and len(r) else None),
        )
        for d, r in raw
    )
    collapse: int | None = None
    notes: list[str] = []
    first = rows[0]
    if not edge:
        notes.append(
            f"no edge at delay {first.delay} to decay (expectancy {_fmt(first.expectancy_r)}R, "
            f"t {_fmt(first_t, '+.2f')}); nothing to conclude about timing"
        )
    else:
        for row in rows[1:]:
            if row.retention is None or row.retention < _COLLAPSE_RETENTION:
                collapse = row.delay
                break
        if collapse is None:
            last = rows[-1]
            notes.append(
                f"expectancy decays gradually: delay {last.delay} keeps {_pct(last.retention)} "
                f"of the delay-{first.delay} expectancy"
            )
        else:
            row = next(r for r in rows if r.delay == collapse)
            change = (
                f"{_fmt(first.expectancy_r)}R at delay {first.delay}, "
                f"{_fmt(row.expectancy_r)}R at delay {collapse}"
            )
            if collapse == first.delay + 1:
                notes.append(
                    f"expectancy collapses at delay {collapse} ({change}): the result depends on "
                    "an immediate fill, or on information from the signal bar itself"
                )
            else:
                notes.append(
                    f"expectancy falls below {_COLLAPSE_RETENTION:.0%} of its first value by "
                    f"delay {collapse} ({change}): a short-horizon edge; check that you can fill "
                    "that fast"
                )
    return EntryDelayResult(
        rows=rows,
        costs=costs,
        edge_at_first=edge,
        collapse_delay=collapse,
        notes=tuple(notes),
    )
