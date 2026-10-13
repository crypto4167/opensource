"""Behavioural lookahead tests: does a signal read bars it could not have seen?

A value computed for bar ``i`` (a signal, an indicator, a feature) may use
bars ``0..i`` and nothing later. Breaking that rule is the commonest way a
backtest lies, and it is easy to do by accident: a centred rolling window, a
``shift(-1)``, a backward fill, a z-score built from the whole series' mean
and standard deviation, a model fitted on all the data, or an off-by-one
window that includes bar ``i + 1``. Each produces a signal that looks
prescient in a backtest and cannot be computed live.

The tests in this module never read the code they test. They treat it as a
black box and check the defining property of a causal computation directly:

* :func:`truncation_test` recomputes the output on prefixes of the data. A
  causal function returns the same value for bar ``k - 1`` whether or not bars
  ``k, k + 1, ...`` exist.
* :func:`perturbation_test` replaces every bar after a cut with scrambled but
  valid prices. A causal function's output up to the cut cannot change.
* :func:`check_factory` applies the perturbation idea to engine-style
  strategies, whose ``decide`` callback only receives history but can still
  leak through arrays its factory precomputed from the full data.

They work with any Python callable (numpy, pandas, a hand-rolled loop, a
wrapper around another backtesting framework) because they only call it.

These are **heuristics** with no canonical citation. They are one-sided: a
failure is hard evidence of a leak (the output demonstrably depended on later
bars), while a pass is evidence, not proof. Only the prefixes and cuts that
were probed are checked, a leak through data the probes leave unchanged
(volume and timestamps are kept by the perturbation) is invisible, and the
function under test must be deterministic. :func:`self_test` runs the
harness on built-in leaky and honest probes so you can confirm it works in
your environment.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .data import Bars
from .trades import Setup, parse_setup

__all__ = [
    "DecideFactory",
    "LookaheadReport",
    "Mismatch",
    "SignalFunction",
    "check_factory",
    "perturb_future",
    "perturbation_test",
    "self_test",
    "truncation_test",
]

ReportKind = Literal["truncation", "perturbation", "factory"]
Check = Literal["last", "prefix"]
SignalFunction = Callable[[Bars], Any]
DecideFactory = Callable[[Bars], Callable[[int, Bars], Any]]

Cell = float | str | None

_MAX_STORED = 50  # mismatches kept in a report (all are counted)
_PER_PROBE = 5  # earliest mismatches kept per checkpoint, cut or trial
_SCALE_LOW = 1.5
_SCALE_HIGH = 3.0
_INVALID = "invalid levels"


# --------------------------------------------------------------------- results


@dataclass(frozen=True)
class Mismatch:
    """One output value that changed when later bars were removed or altered.

    Attributes
    ----------
    checkpoint:
        For a truncation test, the prefix length ``k`` (bars ``0..k-1`` were
        passed to the function). For a perturbation or factory test, the cut
        index ``c`` (bars after ``c`` were scrambled).
    row:
        Bar index of the output row that changed.
    column:
        Column of a 2-D output, or ``None`` for a 1-D output or a decision.
    expected:
        The value on the original, complete data (what the backtest used).
    got:
        The value with the later bars removed or scrambled (what was knowable).
    trial:
        Perturbation trial number, ``None`` for a truncation test.
    """

    checkpoint: int
    row: int
    column: int | None
    expected: Cell
    got: Cell
    trial: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "row": self.row,
            "column": self.column,
            "expected": self.expected,
            "got": self.got,
            "trial": self.trial,
        }

    def __str__(self) -> str:
        col = "" if self.column is None else f", column {self.column}"
        trial = "" if self.trial is None else f", trial {self.trial}"
        return (
            f"row {self.row}{col} (checkpoint {self.checkpoint}{trial}): "
            f"expected {_fmt(self.expected)}, got {_fmt(self.got)}"
        )


@dataclass(frozen=True)
class LookaheadReport:
    """Outcome of a lookahead test.

    Attributes
    ----------
    kind:
        ``"truncation"``, ``"perturbation"`` or ``"factory"``.
    passed:
        True only when at least one probe ran, no value changed and the
        function never raised. A report with nothing checked does not pass.
    n_checked:
        Probes evaluated: prefixes for a truncation test, perturbed copies of
        the data (cuts times trials) for the other two.
    n_values:
        Individual output values (or decisions) compared.
    n_mismatches:
        Values that differed. Only the earliest are kept in ``mismatches``.
    mismatches:
        Up to 50 mismatches, earliest row first.
    errors:
        One message per probe on which the function raised.
    first_bad_index:
        Earliest output row seen to change, or ``None``. It is the earliest
        among the rows compared, not necessarily the earliest leaking row in
        the data; scan every row with ``check="prefix"`` or explicit
        checkpoints to pin it down.
    checkpoints:
        Prefix lengths (truncation) or cut indices (perturbation, factory)
        that were probed.
    skipped:
        Prefix lengths below ``min_history`` on which the function raised and
        which were therefore not compared.
    column_names:
        Column labels when the function returned a DataFrame-like object.
    notes:
        Warnings about how much the test could see.
    """

    kind: ReportKind
    passed: bool
    n_checked: int
    n_values: int
    n_mismatches: int
    mismatches: tuple[Mismatch, ...]
    errors: tuple[str, ...]
    first_bad_index: int | None
    checkpoints: tuple[int, ...]
    skipped: tuple[int, ...] = ()
    column_names: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()

    @property
    def leaks(self) -> bool:
        """True when some output value demonstrably depended on later bars."""
        return self.n_mismatches > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "n_checked": self.n_checked,
            "n_values": self.n_values,
            "n_mismatches": self.n_mismatches,
            "mismatches": [m.as_dict() for m in self.mismatches],
            "errors": list(self.errors),
            "first_bad_index": self.first_bad_index,
            "checkpoints": list(self.checkpoints),
            "skipped": list(self.skipped),
            "column_names": None if self.column_names is None else list(self.column_names),
            "notes": list(self.notes),
        }

    def _column(self, column: int | None) -> str:
        if column is None:
            return ""
        if self.column_names is not None and column < len(self.column_names):
            return f", column {self.column_names[column]!r}"
        return f", column {column}"

    def _explain(self, m: Mismatch) -> str:
        col = self._column(m.column)
        exp, got = _fmt(m.expected), _fmt(m.got)
        if self.kind == "truncation":
            return (
                f"row {m.row}{col} is {exp} on the full data but {got} when only bars "
                f"0..{m.checkpoint - 1} exist"
            )
        trial = "" if m.trial is None else f", trial {m.trial}"
        if self.kind == "perturbation":
            return (
                f"row {m.row}{col} changes from {exp} to {got} when bars after "
                f"{m.checkpoint} are scrambled{trial}"
            )
        return (
            f"decide({m.row}) returns {exp} on the real data but {got} when bars after "
            f"{m.checkpoint} are scrambled{trial}"
        )

    def __str__(self) -> str:
        noun = "prefix(es)" if self.kind == "truncation" else "perturbed run(s)"
        head = f"{self.kind} test"
        lines: list[str] = []
        if self.passed:
            lines.append(
                f"{head} PASSED: {self.n_checked} {noun}, {self.n_values} values compared, "
                "nothing changed when later bars were "
                + ("removed." if self.kind == "truncation" else "scrambled.")
            )
        elif self.n_mismatches:
            first = self.mismatches[0]
            lines.append(
                f"{head} FAILED: LOOKAHEAD - earliest leaking row is {self.first_bad_index}. "
                f"{self.n_mismatches} of {self.n_values} values changed "
                f"across {self.n_checked} {noun}."
            )
            lines.append(f"  first: {self._explain(first)}")
            for m in self.mismatches[1:6]:
                lines.append(f"  also:  {self._explain(m)}")
        elif self.errors:
            lines.append(f"{head} FAILED: the function raised on {len(self.errors)} probe(s).")
        else:
            lines.append(f"{head} INCONCLUSIVE: nothing was checked.")
        for err in self.errors[:5]:
            lines.append(f"  error: {err}")
        if len(self.errors) > 5:
            lines.append(f"  ... and {len(self.errors) - 5} more errors")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ------------------------------------------------------------------ helpers


def _fmt(value: Cell) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return format(value, ".6g")
    return value


def _cell(value: Any) -> Cell:
    """A JSON-friendly scalar for a mismatch record."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def _check_bars(bars: Any, minimum: int) -> int:
    if not isinstance(bars, Bars):
        raise ValueError(f"bars must be a Bars instance, got {type(bars).__name__}")
    n = len(bars)
    if n < minimum:
        raise ValueError(f"need at least {minimum} bar(s), got {n}")
    return n


def _int_option(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_tolerance(rtol: Any, atol: Any) -> tuple[float, float]:
    out = []
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number, got {value!r}")
        val = float(value)
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        out.append(val)
    return out[0], out[1]


def _given_points(points: Iterable[Any], lo: int, hi: int, name: str) -> list[int]:
    chosen: set[int] = set()
    for p in points:
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise ValueError(f"{name} values must be integers, got {p!r}")
        if not lo <= p <= hi:
            raise ValueError(f"{name} {p} is outside the allowed range {lo}..{hi}")
        chosen.add(int(p))
    if not chosen:
        raise ValueError(f"no {name} values given")
    return sorted(chosen)


def _auto_points(
    lo: int, hi: int, count: int, active: NDArray[np.intp], rng: np.random.Generator
) -> list[int]:
    """``count`` probe points in ``lo..hi``: evenly spaced, at active rows, random.

    Every point is used when the range holds no more than ``count`` of them.
    """
    if hi - lo + 1 <= count:
        return list(range(lo, hi + 1))
    n_even = max(1, count // 2)
    n_active = (count + 2) // 4
    chosen: set[int] = set(np.linspace(lo, hi, n_even + 1)[1:].round().astype(int).tolist())
    act = active[(active >= lo) & (active <= hi)]
    if n_active and len(act):
        picks = rng.choice(act, size=min(n_active, len(act)), replace=False)
        chosen.update(int(x) for x in picks)
        if n_active > 1:
            chosen.add(int(act[0]))  # the earliest active row: the edge of any warm-up
    for _ in range(20):
        need = count - len(chosen)
        if need <= 0:
            break
        for d in rng.integers(lo, hi + 1, size=2 * need).tolist():
            if len(chosen) >= count:
                break
            chosen.add(int(d))
    return sorted(chosen)


def _coerce(
    raw: Any, n: int, where: str, ncols: int | None = None
) -> tuple[NDArray[Any], bool, tuple[str, ...] | None]:
    """Turn a function's output into a 2-D array with one row per bar.

    Numeric, boolean and None-holding outputs become float64 (None becomes
    NaN). Anything else (for example ``"BUY"``/``"SELL"`` strings) is kept as
    is and compared by exact equality.
    """
    names: tuple[str, ...] | None = None
    columns = getattr(raw, "columns", None)
    try:
        arr = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} did not return an array-like ({exc})") from exc
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"{where} returned an array of shape {arr.shape}; expected a 1-D or 2-D "
            "array with one row per bar"
        )
    if arr.shape[0] != n:
        raise ValueError(
            f"{where} returned {arr.shape[0]} rows for {n} bars; it must return exactly one "
            "row per bar (pad warm-up rows with NaN instead of dropping them)"
        )
    one_d = arr.ndim == 1
    if one_d:
        arr = arr.reshape(n, 1)
    elif columns is not None:
        names = tuple(str(c) for c in columns)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(
            f"{where} returned {arr.shape[1]} column(s) but fn(bars) returned {ncols}; "
            "the number of columns must not depend on how many bars there are"
        )
    kind = arr.dtype.kind
    if kind in "biuf":
        arr = arr.astype(np.float64)
    elif kind == "O":
        # Numbers and None become float64; genuinely non-numeric objects are
        # left as they are and compared by equality.
        with contextlib.suppress(TypeError, ValueError):
            arr = arr.astype(np.float64)
    elif kind == "c":
        raise ValueError(f"{where} returned complex numbers, which are not supported")
    return arr, one_d, names


def _is_nan(x: Any) -> bool:
    return isinstance(x, (float, np.floating)) and math.isnan(x)


def _same_object(a: Any, b: Any) -> bool:
    if a is b or (_is_nan(a) and _is_nan(b)):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _bad_cells(
    expected: NDArray[Any], got: NDArray[Any], rtol: float, atol: float
) -> NDArray[np.bool_]:
    """Mask of cells that differ.

    NaN equals NaN; NaN against a number is a difference (the signature of a
    backward fill or a centred window that has no future to read). Numbers
    match when ``|a - b| <= atol + rtol * max(|a|, |b|)``, and an infinity
    only matches the same infinity.
    """
    if expected.dtype == np.float64 and got.dtype == np.float64:
        with np.errstate(invalid="ignore", over="ignore"):
            same = (expected == got) | (np.isnan(expected) & np.isnan(got))
            finite = np.isfinite(expected) & np.isfinite(got)
            tol = atol + rtol * np.maximum(np.abs(expected), np.abs(got))
            close = finite & (np.abs(expected - got) <= tol)
        bad: NDArray[np.bool_] = ~(same | close)
        return bad
    eo = expected.astype(object)
    go = got.astype(object)
    out = np.empty(eo.shape, dtype=bool)
    for idx in np.ndindex(eo.shape):
        out[idx] = not _same_object(eo[idx], go[idx])
    return out


def _active_rows(out: NDArray[Any]) -> NDArray[np.intp]:
    """Rows where some column is set: finite and non-zero, or a non-empty value."""
    if out.dtype == np.float64:
        mask = (np.isfinite(out) & (out != 0)).any(axis=1)
    else:
        flat = out.astype(object)
        mask = np.zeros(out.shape[0], dtype=bool)
        for i in range(out.shape[0]):
            mask[i] = any(
                not (v is None or _is_nan(v) or _same_object(v, 0) or _same_object(v, ""))
                for v in flat[i]
            )
    return np.flatnonzero(mask)


class _Collector:
    """Accumulates mismatches, errors and counts while a test runs."""

    def __init__(self) -> None:
        self.kept: list[Mismatch] = []
        self.n_mismatches = 0
        self.first_bad_index: int | None = None
        self.errors: list[str] = []
        self.n_checked = 0
        self.n_values = 0

    def record(self, found: Sequence[Mismatch], total: int) -> None:
        """Add one probe's mismatches; ``found`` is in row order, ``total`` its full count."""
        if total == 0:
            return
        self.n_mismatches += total
        self.kept.extend(found[:_PER_PROBE])
        row = found[0].row
        if self.first_bad_index is None or row < self.first_bad_index:
            self.first_bad_index = row

    def compare(
        self,
        checkpoint: int,
        expected: NDArray[Any],
        got: NDArray[Any],
        row_offset: int,
        one_d: bool,
        rtol: float,
        atol: float,
        trial: int | None = None,
    ) -> None:
        bad = _bad_cells(expected, got, rtol, atol)
        self.n_values += int(bad.size)
        rows, cols = np.nonzero(bad)  # row-major, so earliest rows first
        found = [
            Mismatch(
                checkpoint=checkpoint,
                row=int(r) + row_offset,
                column=None if one_d else int(c),
                expected=_cell(expected[r, c]),
                got=_cell(got[r, c]),
                trial=trial,
            )
            for r, c in zip(rows[:_PER_PROBE], cols[:_PER_PROBE], strict=True)
        ]
        self.record(found, len(rows))

    def report(
        self,
        kind: ReportKind,
        checkpoints: Sequence[int],
        skipped: Sequence[int] = (),
        column_names: tuple[str, ...] | None = None,
        notes: Sequence[str] = (),
    ) -> LookaheadReport:
        all_notes = list(notes)
        if self.n_checked == 0 and not self.errors:
            all_notes.append("no probe could be evaluated, so nothing was checked")
        kept = sorted(
            self.kept,
            key=lambda m: (m.row, m.checkpoint, m.trial or 0, -1 if m.column is None else m.column),
        )
        return LookaheadReport(
            kind=kind,
            passed=self.n_checked > 0 and self.n_mismatches == 0 and not self.errors,
            n_checked=self.n_checked,
            n_values=self.n_values,
            n_mismatches=self.n_mismatches,
            mismatches=tuple(kept[:_MAX_STORED]),
            errors=tuple(self.errors),
            first_bad_index=self.first_bad_index,
            checkpoints=tuple(checkpoints),
            skipped=tuple(skipped),
            column_names=column_names,
            notes=tuple(all_notes),
        )


def _evaluate_full(
    fn: SignalFunction, bars: Bars, rtol: float, atol: float
) -> tuple[NDArray[Any], bool, tuple[str, ...] | None]:
    """Call ``fn(bars)`` twice and insist on identical answers."""
    if not callable(fn):
        raise ValueError(f"fn must be callable, got {type(fn).__name__}")
    n = len(bars)
    full, one_d, names = _coerce(fn(bars), n, "fn(bars)")
    again, _, _ = _coerce(fn(bars), n, "fn(bars)", ncols=full.shape[1])
    if _bad_cells(full, again, rtol, atol).any():
        raise ValueError(
            "fn(bars) returned different values on two calls with the same bars; a lookahead "
            "test needs a deterministic function (seed any random number generator inside it)"
        )
    return full, one_d, names


def _activity_notes(full: NDArray[Any]) -> list[str]:
    if len(_active_rows(full)) == 0:
        return ["fn(bars) is zero, empty or NaN on every row, so there was little to compare"]
    return []


# ------------------------------------------------------------- perturbation


def _perturb(bars: Bars, cut: int, rng: np.random.Generator, invert: bool) -> Bars:
    n = len(bars)
    tail = slice(cut + 1, n)
    m = n - cut - 1
    factors = rng.uniform(_SCALE_LOW, _SCALE_HIGH, size=(4, m))
    if invert:
        factors = 1.0 / factors
    new_open = bars.open[tail] * factors[0]
    new_close = bars.close[tail] * factors[1]
    stacked = np.vstack(
        [new_open, new_close, bars.high[tail] * factors[2], bars.low[tail] * factors[3]]
    )
    o, h, lo, c = bars.open.copy(), bars.high.copy(), bars.low.copy(), bars.close.copy()
    o[tail] = new_open
    c[tail] = new_close
    h[tail] = stacked.max(axis=0)
    lo[tail] = stacked.min(axis=0)

    changed = (
        (o[tail] != bars.open[tail])
        | (h[tail] != bars.high[tail])
        | (lo[tail] != bars.low[tail])
        | (c[tail] != bars.close[tail])
    )
    if not changed.all():
        bad = cut + 1 + int(np.argmin(changed))
        raise ValueError(
            f"cannot perturb bar {bad}: all its prices are zero, so scaling leaves it unchanged"
        )
    out = bars.with_prices(o, h, lo, c)
    keep = slice(0, cut + 1)
    for name in ("open", "high", "low", "close"):
        if not np.array_equal(getattr(out, name)[keep], getattr(bars, name)[keep]):
            raise RuntimeError(f"internal error: perturbation changed {name} at or before the cut")
    return out


def perturb_future(bars: Bars, cut: int, *, seed: int | None = 0, invert: bool = False) -> Bars:
    """Copy of ``bars`` with every bar after ``cut`` replaced by scrambled OHLC.

    Bars ``0..cut`` are untouched. Each later bar's open, high, low and close
    are multiplied by independent random factors in ``[1.5, 3]`` (divided by
    them when ``invert`` is true), then the high and low are rebuilt as the
    maximum and minimum of the four scaled prices so every bar stays valid.
    Volume and timestamps are kept. The result is a plausible-looking but
    completely different future: anything computed for bars ``0..cut`` that
    changes on it was reading bars after the cut.
    """
    n = _check_bars(bars, 2)
    cut = _int_option(cut, "cut", 0)
    if cut > n - 2:
        raise ValueError(f"cut must leave at least one bar after it (cut <= {n - 2}), got {cut}")
    return _perturb(bars, cut, np.random.default_rng(seed), bool(invert))


# -------------------------------------------------------------------- tests


def truncation_test(
    fn: SignalFunction,
    bars: Bars,
    *,
    checkpoints: Iterable[int] | None = None,
    n_checkpoints: int = 40,
    min_history: int = 1,
    seed: int | None = 0,
    rtol: float = 1e-9,
    atol: float = 0.0,
    check: Check = "last",
) -> LookaheadReport:
    """Recompute ``fn`` on prefixes of the data and compare with the full run.

    What it catches
    ---------------
    Any output row that depends on later bars: centred windows, ``shift(-k)``,
    backward fills, whole-series normalisation, models fitted on all the
    data, windows that run one bar too far. A causal function returns the
    same value for bar ``k - 1`` from ``bars.prefix(k)`` as from all bars.

    Parameters
    ----------
    fn:
        ``fn(bars)`` returns an array-like whose first axis has one row per
        bar: 1-D, or 2-D for several columns. pandas Series and DataFrames
        work through :func:`numpy.asarray`. Row ``i`` must describe what is
        known at the close of bar ``i``. Non-numeric outputs (``"BUY"`` and
        ``"SELL"`` strings) are compared by exact equality.
    bars:
        The data the backtest used.
    checkpoints:
        Prefix lengths ``k`` (numbers of bars, ``1..len(bars)``) to test;
        ``k = len(bars)`` is the full data and checks nothing. By
        default ``n_checkpoints`` of them are chosen between ``min_history``
        and ``len(bars) - 1``: half evenly spaced, a quarter just after rows
        where the full output is non-zero and non-NaN (where a sparse signal
        actually says something), the rest at random from ``seed``. Every
        prefix is tested when there are no more than ``n_checkpoints``.
    min_history:
        Shortest prefix ``fn`` is expected to handle. If ``fn`` raises on a
        shorter prefix, that prefix is skipped; an exception at or above
        ``min_history`` is reported as an error and the test fails.
    rtol, atol:
        Values match when ``|a - b| <= atol + rtol * max(|a|, |b|)``. NaN
        matches NaN, but NaN against a number is a mismatch: that is exactly
        what a backward fill or centred window looks like on a prefix.
    check:
        ``"last"`` compares row ``k - 1`` of the prefix output with row
        ``k - 1`` of the full output. ``"prefix"`` compares every row below
        ``k``, which also catches outputs that are revised (repainted) as
        more data arrives, and pins down the earliest leaking row exactly.

    Raises
    ------
    ValueError
        On bad options, when ``fn`` returns the wrong number of rows or
        columns, or when ``fn(bars)`` is not deterministic. An exception
        raised by ``fn(bars)`` itself propagates unchanged.

    Limitations
    -----------
    Only the probed prefixes are tested. A leak that only shows in some
    stretches of data can be missed by a sparse set of checkpoints; pass
    ``checkpoints=range(1, len(bars) + 1)`` to test every one. Numerical
    noise that depends on the series length (FFT convolutions, pairwise
    summation over the whole array) can exceed ``rtol`` and show up as a
    small mismatch; inspect the size of the differences before concluding.
    """
    n = _check_bars(bars, 1)
    n_checkpoints = _int_option(n_checkpoints, "n_checkpoints", 1)
    min_history = _int_option(min_history, "min_history", 1)
    if min_history > n:
        raise ValueError(f"min_history ({min_history}) exceeds the number of bars ({n})")
    rtol, atol = _check_tolerance(rtol, atol)
    if check not in ("last", "prefix"):
        raise ValueError(f"check must be 'last' or 'prefix', got {check!r}")

    full, one_d, names = _evaluate_full(fn, bars, rtol, atol)
    rng = np.random.default_rng(seed)
    if checkpoints is None:
        hi = max(min_history, n - 1)
        ks = _auto_points(min_history, hi, n_checkpoints, _active_rows(full) + 1, rng)
    else:
        ks = _given_points(checkpoints, 1, n, "checkpoint")

    col = _Collector()
    skipped: list[int] = []
    ncols = full.shape[1]
    for k in ks:
        if k == n:
            continue  # the full data as its own prefix cannot differ: nothing to check
        try:
            raw = fn(bars.prefix(k))
        except Exception as exc:
            if k < min_history:
                skipped.append(k)
            else:
                col.errors.append(
                    f"prefix {k}: {type(exc).__name__}: {exc} (if fn needs a minimum amount "
                    "of history, pass min_history)"
                )
            continue
        out, _, _ = _coerce(raw, k, f"fn(bars.prefix({k}))", ncols=ncols)
        col.n_checked += 1
        if check == "last":
            col.compare(k, full[k - 1 : k], out[k - 1 : k], k - 1, one_d, rtol, atol)
        else:
            col.compare(k, full[:k], out, 0, one_d, rtol, atol)
    return col.report("truncation", ks, skipped, names, _activity_notes(full))


def perturbation_test(
    fn: SignalFunction,
    bars: Bars,
    *,
    cuts: Iterable[int] | None = None,
    n_cuts: int = 5,
    n_trials: int = 3,
    seed: int | None = 0,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> LookaheadReport:
    """Scramble every bar after a cut and check that the past output is unchanged.

    What it catches
    ---------------
    Centred windows, backward fills, whole-series normalisation (a z-score
    with the full series' mean and standard deviation), ``shift(-k)``, and
    anything else that reads bars after the one it describes. Unlike
    :func:`truncation_test` it keeps the series length fixed, so it also
    catches code that behaves the same on short data but reads ahead when the
    future exists, and it never runs ``fn`` on a short prefix.

    Parameters
    ----------
    fn:
        As for :func:`truncation_test`.
    bars:
        The data the backtest used; at least two bars.
    cuts:
        Cut indices ``c`` in ``0..len(bars) - 2``. For each, bars ``c + 1``
        onwards are replaced (see :func:`perturb_future`) and output rows
        ``0..c`` must not change. By default ``n_cuts`` cuts are chosen: half
        evenly spaced up to ``len(bars) - 2``, some at rows where the full
        output is non-zero and non-NaN, the rest at random from ``seed``.
    n_trials:
        Perturbations per cut, each with fresh random factors. Even-numbered
        trials scale the future up and odd-numbered ones scale it down, so a
        signal that compares a future price with the current one flips at
        least once whichever way it pointed.
    rtol, atol:
        As for :func:`truncation_test`.

    Raises
    ------
    ValueError
        On bad options, a wrong output shape or a non-deterministic ``fn``.
        Exceptions raised by ``fn`` on perturbed data are reported as errors.

    Limitations
    -----------
    Volume and timestamps are not perturbed, so a leak through future volume
    or time alone is invisible. The test cannot see a leak that leaves the
    output unchanged on the scrambled future (for example a signal that
    clips to the same value either way); more cuts and trials reduce that
    chance but never to zero.
    """
    n = _check_bars(bars, 2)
    n_cuts = _int_option(n_cuts, "n_cuts", 1)
    n_trials = _int_option(n_trials, "n_trials", 1)
    rtol, atol = _check_tolerance(rtol, atol)

    full, one_d, names = _evaluate_full(fn, bars, rtol, atol)
    rng = np.random.default_rng(seed)
    if cuts is None:
        cs = _auto_points(0, n - 2, n_cuts, _active_rows(full), rng)
    else:
        cs = _given_points(cuts, 0, n - 2, "cut")

    col = _Collector()
    ncols = full.shape[1]
    for c in cs:
        for trial in range(n_trials):
            perturbed = _perturb(bars, c, rng, invert=trial % 2 == 1)
            try:
                raw = fn(perturbed)
            except Exception as exc:
                col.errors.append(f"cut {c}, trial {trial}: {type(exc).__name__}: {exc}")
                continue
            out, _, _ = _coerce(raw, n, f"fn(perturbed bars, cut {c})", ncols=ncols)
            col.n_checked += 1
            col.compare(c, full[: c + 1], out[: c + 1], 0, one_d, rtol, atol, trial=trial)
    return col.report("perturbation", cs, (), names, _activity_notes(full))


def _decision(decide: Callable[[int, Bars], Any], i: int, bars: Bars) -> Setup | str | None:
    """What the engine would make of ``decide(i, history)``.

    Like :func:`backtest_skeptic.run`, a ``ValueError`` (non-finite levels, an
    unknown side) counts as an invalid setup rather than a crash.
    """
    try:
        return parse_setup(decide(i, bars.prefix(i + 1)))
    except ValueError:
        return _INVALID


def _describe_decision(d: Setup | str | None) -> str:
    if d is None:
        return "no trade"
    if isinstance(d, str):
        return d
    side = "long" if d.side > 0 else "short"
    return f"{side} stop={d.stop:.6g} target={d.target:.6g}"


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    return a == b or abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _same_decision(a: Setup | str | None, b: Setup | str | None, rtol: float, atol: float) -> bool:
    if isinstance(a, Setup) and isinstance(b, Setup):
        return (
            a.side == b.side
            and _close(a.stop, b.stop, rtol, atol)
            and _close(a.target, b.target, rtol, atol)
        )
    return a == b


def check_factory(
    factory: DecideFactory,
    bars: Bars,
    *,
    cuts: Iterable[int] | None = None,
    n_cuts: int = 5,
    n_trials: int = 2,
    warmup: int = 0,
    seed: int | None = 0,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> LookaheadReport:
    """Lookahead test for engine-style strategies built by a factory.

    What it catches
    ---------------
    :func:`backtest_skeptic.run` passes ``decide(i, history)`` only bars
    ``0..i``, but a strategy can still leak: a factory that precomputes an
    indicator from the full data and lets ``decide`` read it through a
    closure sees the future. This test builds the strategy twice, once from
    the real bars and once from bars whose future after a cut has been
    scrambled (:func:`perturb_future`), and requires every decision at bars
    ``warmup..c`` to match.

    Parameters
    ----------
    factory:
        ``factory(bars)`` returns a ``decide(i, history)`` callable, as
        accepted by :func:`backtest_skeptic.run`.
    bars:
        The data the backtest used; at least two bars.
    cuts:
        Cut indices in ``warmup..len(bars) - 2``. By default ``n_cuts`` are
        chosen: half evenly spaced, some at bars where the real strategy
        proposes a trade, the rest at random from ``seed``.
    n_trials:
        Perturbations per cut; even trials scale the future up, odd trials
        down.
    warmup:
        First bar at which ``decide`` is called, as in
        :func:`backtest_skeptic.run`.
    rtol, atol:
        Tolerance for comparing stop and target levels.

    Decisions are compared after :func:`~backtest_skeptic.trades.parse_setup`,
    so ``(1, 99, 102)`` and ``Setup(1, 99, 102)`` are the same decision. A
    ``ValueError`` from ``decide`` counts as an invalid setup, as in the
    engine. Every bar up to each cut is evaluated, in order, so a stateful
    ``decide`` sees the same call sequence on both copies.

    Raises
    ------
    ValueError
        On bad options or when ``factory(bars)`` does not return a callable.
        Exceptions from the real strategy propagate; exceptions on the
        perturbed copy are reported as errors.

    Limitations
    -----------
    The cost is about ``n_cuts * n_trials * len(bars) / 2`` calls to
    ``decide``. ``decide`` must be deterministic. Leaks through volume or
    timestamps are not perturbed and so not seen.
    """
    n = _check_bars(bars, 2)
    n_cuts = _int_option(n_cuts, "n_cuts", 1)
    n_trials = _int_option(n_trials, "n_trials", 1)
    warmup = _int_option(warmup, "warmup", 0)
    if warmup > n - 2:
        raise ValueError(f"warmup must be <= {n - 2} so at least one decision can be tested")
    rtol, atol = _check_tolerance(rtol, atol)
    if not callable(factory):
        raise ValueError(f"factory must be callable, got {type(factory).__name__}")

    decide = factory(bars)
    if not callable(decide):
        raise ValueError(
            f"factory(bars) must return a decide(i, history) callable, got {type(decide).__name__}"
        )
    rng = np.random.default_rng(seed)
    given = None if cuts is None else _given_points(cuts, warmup, n - 2, "cut")
    last = n - 2 if given is None else given[-1]
    original = [_decision(decide, i, bars) for i in range(warmup, last + 1)]
    if given is None:
        active = np.flatnonzero([isinstance(d, Setup) for d in original]) + warmup
        cs = _auto_points(warmup, n - 2, n_cuts, active, rng)
    else:
        cs = given

    col = _Collector()
    for c in cs:
        for trial in range(n_trials):
            perturbed = _perturb(bars, c, rng, invert=trial % 2 == 1)
            try:
                p_decide = factory(perturbed)
                got = [_decision(p_decide, i, perturbed) for i in range(warmup, c + 1)]
            except Exception as exc:
                col.errors.append(f"cut {c}, trial {trial}: {type(exc).__name__}: {exc}")
                continue
            col.n_checked += 1
            col.n_values += len(got)
            found: list[Mismatch] = []
            total = 0
            for offset, (exp, out) in enumerate(zip(original, got, strict=False)):
                if _same_decision(exp, out, rtol, atol):
                    continue
                total += 1
                if len(found) < _PER_PROBE:
                    found.append(
                        Mismatch(
                            checkpoint=c,
                            row=warmup + offset,
                            column=None,
                            expected=_describe_decision(exp),
                            got=_describe_decision(out),
                            trial=trial,
                        )
                    )
            col.record(found, total)
    notes = []
    if not any(isinstance(d, Setup) for d in original):
        notes.append(
            "the strategy proposed no trade on the real data up to the last cut, "
            "so there was little to compare"
        )
    return col.report("factory", cs, (), None, notes)


# ---------------------------------------------------------------- self-test


def _rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Causal rolling mean (NaN until ``window`` bars exist), via a running sum."""
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        cs = np.cumsum(np.concatenate([[0.0], x]))
        out[window - 1 :] = (cs[window:] - cs[:-window]) / window
    return out


def _probe_causal_mean(bars: Bars) -> NDArray[np.float64]:
    return _rolling_mean(bars.close, 5)


def _probe_ewma(bars: Bars) -> NDArray[np.float64]:
    alpha = 0.1
    closes = bars.close.tolist()
    out = np.empty(len(closes))
    level = closes[0]
    for i, value in enumerate(closes):
        level += alpha * (value - level)
        out[i] = level
    return out


def _probe_centred_mean(bars: Bars) -> NDArray[np.float64]:
    half = 2
    x = bars.close
    out = np.full(len(x), np.nan)
    if len(x) > half:
        out[: len(x) - half] = _rolling_mean(x, 2 * half + 1)[half:]
    return out


def _probe_zscore(bars: Bars) -> NDArray[np.float64]:
    x = bars.close
    sd = float(x.std())
    return np.zeros(len(x)) if sd == 0 else (x - x.mean()) / sd


def _probe_shift_back(bars: Bars) -> NDArray[np.float64]:
    out = np.full(len(bars), np.nan)
    out[:-1] = bars.close[1:]
    return out


def _probe_backfill(bars: Bars) -> NDArray[np.float64]:
    """A causal sparse series (closes at 5-bar highs), then backward filled."""
    x = bars.close
    n = len(x)
    sparse = np.full(n, np.nan)
    if n >= 5:
        highs = np.lib.stride_tricks.sliding_window_view(x, 5).max(axis=1)
        event = np.flatnonzero(x[4:] >= highs) + 4
        sparse[event] = x[event]
    idx = np.where(np.isnan(sparse), n, np.arange(n))
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    out = np.full(n, np.nan)
    ok = nxt < n
    out[ok] = sparse[nxt[ok]]
    return out


def _probe_window_too_far(bars: Bars) -> NDArray[np.float64]:
    out = np.full(len(bars), np.nan)
    out[:-1] = _rolling_mean(bars.close, 5)[1:]
    return out


def _factory_honest(bars: Bars) -> Callable[[int, Bars], Any]:
    ma = _rolling_mean(bars.close, 10)  # causal, so precomputing it is harmless

    def decide(i: int, history: Bars) -> tuple[int, float, float] | None:
        c = float(history.close[-1])
        if np.isnan(ma[i]) or c <= ma[i]:
            return None
        return (1, c * 0.99, c * 1.02)

    return decide


def _factory_leaky(bars: Bars) -> Callable[[int, Bars], Any]:
    centred = _probe_centred_mean(bars)  # reads two bars ahead

    def decide(i: int, history: Bars) -> tuple[int, float, float] | None:
        c = float(history.close[-1])
        if np.isnan(centred[i]) or c >= centred[i]:
            return None
        return (1, c * 0.99, c * 1.02)

    return decide


_SIGNAL_PROBES: tuple[tuple[str, SignalFunction, bool], ...] = (
    ("causal_rolling_mean", _probe_causal_mean, True),
    ("ewma", _probe_ewma, True),
    ("centred_rolling_mean", _probe_centred_mean, False),
    ("full_sample_zscore", _probe_zscore, False),
    ("shift_minus_1", _probe_shift_back, False),
    ("backfill_sparse", _probe_backfill, False),
    ("window_includes_next_bar", _probe_window_too_far, False),
)


def self_test(*, n_bars: int = 300, seed: int = 0) -> dict[str, bool]:
    """Run the harness on built-in probes and report whether it judged each correctly.

    On a driftless random walk of ``n_bars`` bars, both
    :func:`truncation_test` and :func:`perturbation_test` must pass an honest
    causal rolling mean and an EWMA, and both must catch a centred rolling
    mean, a full-sample z-score, a ``shift(-1)``, a backward fill of a sparse
    series and a rolling window that includes bar ``i + 1``.
    :func:`check_factory` must pass a factory that precomputes a causal
    indicator and catch one that precomputes a centred one. Each entry is
    ``True`` when the harness got that probe right. All probes are pure numpy.
    """
    from .synthetic import random_walk_bars

    n_bars = _int_option(n_bars, "n_bars", 20)
    bars = random_walk_bars(n_bars, seed=seed)
    results: dict[str, bool] = {}
    for name, fn, honest in _SIGNAL_PROBES:
        trunc = truncation_test(fn, bars, seed=seed)
        pert = perturbation_test(fn, bars, seed=seed)
        if honest:
            results[name] = trunc.passed and pert.passed
        else:
            results[name] = trunc.leaks and pert.leaks
    results["factory_honest"] = check_factory(_factory_honest, bars, seed=seed).passed
    results["factory_precomputed_leak"] = check_factory(_factory_leaky, bars, seed=seed).leaks
    return results
