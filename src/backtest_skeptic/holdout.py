"""A read-once holdout ledger and a registry of every configuration tried.

Two pieces of bookkeeping decide whether an out-of-sample result means
anything, and both are usually kept in someone's head:

1. **How often the holdout was looked at.** A holdout is clean only until
   its result first influences a decision. Evaluate, tweak, re-evaluate, and
   the holdout has become training data: Dwork et al. (2015) show that
   adaptively re-using a holdout lets an analyst overfit it as surely as
   the training set, unless access to it is deliberately limited. This
   module takes the strictest form of that limit: each sealed holdout may be
   read once, and every read is written to an append-only file.
   :class:`HoldoutLedger` refuses a second read unless it is explicitly
   allowed, and then records it as a re-read for anyone auditing the result.

2. **How many configurations were tried.** The best of many backtests looks
   good by luck alone. The deflated Sharpe ratio (Bailey & Lopez de Prado,
   2014) corrects for this, but needs the number of trials ``N`` and the
   variance of their Sharpe ratios ``V[SR]``, which nobody remembers after
   the fact. :class:`TrialRegistry` records every trial as it is run, so
   both numbers can be read off at the end
   (``deflated_sharpe_ratio(..., n_trials=N, var_trials_sharpe=V)`` in
   :mod:`backtest_skeptic.stats`).

:func:`split_holdout` produces a train/holdout split with an optional purge
gap, and :func:`fingerprint` hashes bars so that a holdout cannot be swapped
or edited between sealing and reading.

The files are JSON lines, one event per line, and each line carries the
SHA-256 of the line before it, so editing, reordering or deleting a line in
the middle is detected when the file is loaded. This is an audit trail, not
access control: nothing stops someone from evaluating the holdout without
telling the ledger, or from deleting the whole file. Its value is that an
honest workflow leaves evidence, and that a re-read cannot happen by
accident.

References
----------
* Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. & Roth, A.
  (2015). The reusable holdout: Preserving validity in adaptive data
  analysis. *Science* 349(6248).
* Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  *Journal of Portfolio Management* 40(5).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import ArrayLike

from .data import Bars

__all__ = [
    "HoldoutAlreadyRead",
    "HoldoutError",
    "HoldoutLedger",
    "HoldoutMismatch",
    "HoldoutSplit",
    "HoldoutStatus",
    "LedgerCorrupted",
    "LedgerStatus",
    "ReadRecord",
    "SealRecord",
    "TrialRecord",
    "TrialRegistry",
    "TrialSummary",
    "fingerprint",
    "split_holdout",
]

_GENESIS = "0" * 64
_FORMAT_VERSION = 1
_REL_TOL = 1e-12
_FINGERPRINT_TAG = b"backtest-skeptic/bars/v1"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class HoldoutError(ValueError):
    """Base class for holdout-ledger and trial-registry errors."""


class HoldoutAlreadyRead(HoldoutError):
    """The holdout was already read once and a re-read was not allowed."""


class HoldoutMismatch(HoldoutError):
    """The bars presented do not match the fingerprint recorded when sealing."""


class LedgerCorrupted(HoldoutError):
    """The ledger file cannot be parsed or its hash chain is broken."""


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutSplit:
    """Disjoint train, purged and holdout index ranges over ``n`` bars.

    ``train``, ``purged`` and ``holdout`` are contiguous slices that follow
    each other in that order and together cover ``0 .. n-1`` exactly once. Use them directly on
    :class:`~backtest_skeptic.Bars` (``bars[split.train]``) or call
    :meth:`apply`.
    """

    train: slice
    holdout: slice
    purged: slice
    n: int

    @property
    def n_train(self) -> int:
        return int(self.train.stop - self.train.start)

    @property
    def n_purged(self) -> int:
        return int(self.purged.stop - self.purged.start)

    @property
    def n_holdout(self) -> int:
        return int(self.holdout.stop - self.holdout.start)

    def apply(self, bars: Bars) -> tuple[Bars, Bars]:
        """``(train_bars, holdout_bars)``; raises if ``bars`` does not have ``n`` bars."""
        if not isinstance(bars, Bars):
            raise TypeError(f"expected Bars, got {type(bars).__name__}")
        if len(bars) != self.n:
            raise ValueError(f"this split is for {self.n} bars, got {len(bars)}")
        return bars[self.train], bars[self.holdout]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "train": [self.train.start, self.train.stop],
            "purged": [self.purged.start, self.purged.stop],
            "holdout": [self.holdout.start, self.holdout.stop],
            "n_train": self.n_train,
            "n_purged": self.n_purged,
            "n_holdout": self.n_holdout,
        }

    def __str__(self) -> str:
        def span(s: slice) -> str:
            return f"{s.start}..{s.stop - 1}" if s.stop > s.start else "none"

        return (
            f"train bars {span(self.train)} ({self.n_train}), "
            f"purged {span(self.purged)} ({self.n_purged}), "
            f"holdout {span(self.holdout)} ({self.n_holdout}) of {self.n}"
        )


def _int_arg(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def split_holdout(
    n: int,
    *,
    holdout_fraction: float = 0.2,
    holdout_start: int | None = None,
    purge: int = 0,
) -> HoldoutSplit:
    """Split ``n`` bars into train and a final holdout that never share a bar.

    Parameters
    ----------
    n:
        Number of bars.
    holdout_fraction:
        Fraction of the bars, at the end, that form the holdout. The holdout
        gets ``ceil(holdout_fraction * n)`` bars, so it is never smaller than
        asked. Ignored when ``holdout_start`` is given.
    holdout_start:
        Index of the first holdout bar, instead of a fraction. The holdout is
        ``holdout_start .. n-1``.
    purge:
        Number of bars at the end of the training range to drop. A label or
        trade that looks up to ``h`` bars ahead of its decision bar (a
        forward return over ``h`` bars, or an engine trade with
        ``max_bars + entry_delay = h``) uses only pre-holdout prices when
        ``purge >= h``. Without the purge, the last training labels are
        computed from holdout prices and the holdout leaks into training.

    The boundary is half-open: train is ``[0, holdout_start - purge)``, the
    purged gap is ``[holdout_start - purge, holdout_start)`` and the holdout
    is ``[holdout_start, n)``, so the boundary bar belongs to the holdout
    only. (Using ``<=`` on both sides of the boundary puts that bar in both
    sets, which is the bug this function exists to prevent.)

    Raises ``ValueError`` when the split leaves no training bar or no
    holdout bar.
    """
    n = _int_arg(n, "n")
    purge = _int_arg(purge, "purge")
    if n < 2:
        raise ValueError(f"n must be at least 2 to split, got {n}")
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}")
    if holdout_start is None:
        if isinstance(holdout_fraction, bool):
            raise ValueError(f"holdout_fraction must be a number, got {holdout_fraction!r}")
        try:
            frac = float(holdout_fraction)
        except (TypeError, ValueError):
            raise ValueError(
                f"holdout_fraction must be a number, got {holdout_fraction!r}"
            ) from None
        if not (math.isfinite(frac) and 0.0 < frac < 1.0):
            raise ValueError(f"holdout_fraction must be strictly between 0 and 1, got {frac!r}")
        n_holdout = math.ceil(frac * n - 1e-9)
        start = n - n_holdout
    else:
        start = _int_arg(holdout_start, "holdout_start")
    if not 1 <= start <= n - 1:
        raise ValueError(
            f"the holdout must start in 1..{n - 1} so that both sides are non-empty; "
            f"got holdout start {start} for n={n}"
        )
    train_stop = start - purge
    if train_stop < 1:
        raise ValueError(
            f"purge={purge} removes every training bar (the holdout starts at bar {start})"
        )
    return HoldoutSplit(
        train=slice(0, train_stop),
        holdout=slice(start, n),
        purged=slice(train_stop, start),
        n=n,
    )


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def fingerprint(bars: Bars) -> str:
    """SHA-256 hex digest identifying a set of bars exactly.

    The digest covers, in order: a format tag, the number of bars, the
    little-endian float64 bytes of ``open``, ``high``, ``low`` and ``close``,
    the volume bytes when the bars have volume, and the timestamps (each
    converted with ``str``) when they have timestamps. Every section is
    labelled and length-prefixed, so bars with and without volume, or with
    shifted arrays, cannot collide.

    Any change to a single price, even in the last bit, a bar added or
    removed, or a timestamp shifted, changes the fingerprint. Bars that are
    equal value-for-value but built differently (sliced from a longer
    series, re-read from a CSV) have the same fingerprint. ``-0.0`` and
    ``0.0`` differ.
    """
    if not isinstance(bars, Bars):
        raise TypeError(f"expected Bars, got {type(bars).__name__}")
    n = len(bars)
    h = hashlib.sha256()
    h.update(_FINGERPRINT_TAG)

    def section(label: bytes, payload: bytes) -> None:
        h.update(label + b"\x00")
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)

    section(b"n", n.to_bytes(8, "little"))
    arrays = [("open", bars.open), ("high", bars.high), ("low", bars.low), ("close", bars.close)]
    if bars.volume is not None:
        arrays.append(("volume", bars.volume))
    for label, arr in arrays:
        section(label.encode("ascii"), np.ascontiguousarray(arr, dtype="<f8").tobytes())
    if bars.time is not None:
        encoded = [str(t).encode("utf-8") for t in bars.time]
        lengths = np.array([len(e) for e in encoded], dtype="<u8").tobytes()
        section(b"time", lengths + b"".join(encoded))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON helpers and the append-only file
# ---------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def _jsonable(value: Any, where: str = "value") -> Any:
    """Convert ``value`` to plain JSON types.

    numpy scalars and arrays, tuples, dataclasses and objects with an
    ``as_dict()`` method are converted. Non-finite floats become the strings
    ``"NaN"``, ``"Infinity"`` and ``"-Infinity"`` so the file stays strict
    JSON. Anything else raises ``ValueError`` before the file is touched.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{where}: dict keys must be strings, got {k!r}")
            out[k] = _jsonable(v, f"{where}[{k!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, np.ndarray):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(value.tolist())]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict(), where)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), where)
    raise ValueError(f"{where}: a {type(value).__name__} cannot be stored as JSON")


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


if sys.platform == "win32":  # pragma: no cover - exercised on Windows only

    def _lock(fh: IO[str]) -> None:
        """No advisory locking on Windows; concurrent writers are not serialised."""

    def _unlock(fh: IO[str]) -> None:
        pass

    def _lock_shared(fh: IO[str]) -> None:
        pass

else:
    import fcntl

    def _lock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _lock_shared(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)


Builder = Callable[[list[dict[str, Any]]], dict[str, Any] | None]


class _AppendOnlyLog:
    """JSON-lines file where each line stores the SHA-256 of the previous line.

    Every operation re-reads the whole file, so several objects or processes
    pointing at the same path always see the same history. Appends hold an
    exclusive ``flock`` (POSIX only) while they re-read, decide and write,
    so two writers cannot both record "read number 1"; plain reads hold a
    shared one, so they never see a half-written line.
    """

    def __init__(self, path: str | os.PathLike[str], events: frozenset[str], what: str) -> None:
        self.path = Path(path)
        self.events = events
        self.what = what

    def _parse(self, text: str) -> tuple[list[dict[str, Any]], str]:
        if not text:
            return [], _GENESIS
        if not text.endswith("\n"):
            raise LedgerCorrupted(
                f"{self.path}: the last line is incomplete (an interrupted write?). "
                "Inspect the file; the ledger will not guess which events are valid."
            )
        records: list[dict[str, Any]] = []
        prev = _GENESIS
        for lineno, line in enumerate(text[:-1].split("\n"), start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerCorrupted(
                    f"{self.path}, line {lineno}: not valid JSON ({exc})"
                ) from exc
            if not isinstance(rec, dict) or rec.get("event") not in self.events:
                raise LedgerCorrupted(f"{self.path}, line {lineno}: not a {self.what} event")
            if rec.get("prev") != prev:
                raise LedgerCorrupted(
                    f"{self.path}, line {lineno}: hash chain broken; a line was edited, "
                    "reordered, inserted or removed"
                )
            prev = _line_hash(line)
            records.append(rec)
        return records, prev

    def load(self) -> tuple[list[dict[str, Any]], str]:
        """All events and the hash of the last line (the chain head)."""
        try:
            with open(self.path, encoding="utf-8", newline="") as fh:
                # Shared lock: never read a line an append is half-way through.
                _lock_shared(fh)
                try:
                    text = fh.read()
                finally:
                    _unlock(fh)
        except FileNotFoundError:
            return [], _GENESIS
        return self._parse(text)

    def append(self, build: Builder) -> dict[str, Any] | None:
        """Re-read under a lock, let ``build`` decide, append its event (if any)."""
        with open(self.path, "a+", encoding="utf-8", newline="") as fh:
            _lock(fh)
            try:
                fh.seek(0)
                records, head = self._parse(fh.read())
                event = build(records)
                if event is None:
                    return None
                event = {**event, "prev": head}
                line = json.dumps(event, allow_nan=False)
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                _unlock(fh)
        return event


def _check_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _field(rec: Mapping[str, Any], key: str, path: Path) -> Any:
    try:
        return rec[key]
    except KeyError:
        raise LedgerCorrupted(f"{path}: a {rec.get('event')!r} event has no {key!r}") from None


# ---------------------------------------------------------------------------
# holdout ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealRecord:
    """What was recorded when a holdout was sealed."""

    name: str
    fingerprint: str
    n_bars: int
    first_time: str | None
    last_time: str | None
    created_at: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        span = ""
        if self.first_time is not None:
            span = f" {self.first_time}..{self.last_time}"
        note = f" ({self.note})" if self.note else ""
        return (
            f"holdout {self.name!r}: {self.n_bars} bars{span}, sealed {self.created_at}, "
            f"fingerprint {self.fingerprint[:12]}{note}"
        )


@dataclass(frozen=True)
class ReadRecord:
    """One recorded read of a sealed holdout."""

    name: str
    read_number: int
    strategy_id: str
    fingerprint: str
    read_at: str
    reread: bool
    result: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "read_number": self.read_number,
            "strategy_id": self.strategy_id,
            "fingerprint": self.fingerprint,
            "read_at": self.read_at,
            "reread": self.reread,
            "result": None if self.result is None else json.loads(json.dumps(self.result)),
        }

    def __str__(self) -> str:
        flag = " (RE-READ: this holdout is no longer clean)" if self.reread else ""
        return (
            f"read #{self.read_number} of holdout {self.name!r} by {self.strategy_id!r} "
            f"at {self.read_at}{flag}"
        )


@dataclass(frozen=True)
class HoldoutStatus:
    """A sealed holdout and how often it has been read.

    ``clean`` is True while the holdout has been read at most once.
    """

    name: str
    fingerprint: str
    n_bars: int
    first_time: str | None
    last_time: str | None
    sealed_at: str
    note: str
    n_reads: int
    strategies: tuple[str, ...]
    last_read_at: str | None

    @property
    def clean(self) -> bool:
        return self.n_reads <= 1

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["strategies"] = list(self.strategies)
        out["clean"] = self.clean
        return out

    def __str__(self) -> str:
        span = f" {self.first_time}..{self.last_time}" if self.first_time is not None else ""
        if self.n_reads == 0:
            reads = "never read"
        else:
            who = ", ".join(self.strategies)
            times = "time" if self.n_reads == 1 else "times"
            reads = f"read {self.n_reads} {times} ({who}), last {self.last_read_at}"
        warn = "" if self.clean else " -- read more than once, no longer a clean holdout"
        return f"{self.name}: {self.n_bars} bars{span}, sealed {self.sealed_at}; {reads}{warn}"


@dataclass(frozen=True)
class LedgerStatus:
    """Every sealed holdout in a ledger file.

    ``head`` is the SHA-256 of the last line. Publishing or committing it
    elsewhere lets an auditor detect later truncation of the file, which the
    hash chain alone cannot.
    """

    path: str
    n_events: int
    head: str
    holdouts: tuple[HoldoutStatus, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_events": self.n_events,
            "head": self.head,
            "holdouts": [h.as_dict() for h in self.holdouts],
        }

    def __str__(self) -> str:
        lines = [f"holdout ledger {self.path} ({self.n_events} events, head {self.head[:12]})"]
        if not self.holdouts:
            lines.append("  no holdouts sealed")
        lines.extend(f"  {h}" for h in self.holdouts)
        return "\n".join(lines)


class HoldoutLedger:
    """Append-only record that each holdout was sealed once and read once.

    Workflow::

        split = split_holdout(len(bars), holdout_fraction=0.2, purge=24)
        train, holdout = split.apply(bars)
        ledger = HoldoutLedger("holdout_ledger.jsonl")
        ledger.seal("btc-2024", holdout, note="final 20%, purge 24")
        ...  # research on ``train`` only, recording trials in a TrialRegistry
        result = evaluate(holdout)
        ledger.read("btc-2024", holdout, strategy_id="breakout-v7", result=result)

    A second :meth:`read` of the same holdout raises
    :class:`HoldoutAlreadyRead`. ``allow_reread=True`` records the read
    anyway, numbered and flagged, so the result can be reported honestly as
    coming from a holdout that was looked at before.

    The file is JSON lines with a SHA-256 hash chain (see the module
    docstring). Every call re-reads the file, so ledgers opened later, or in
    another process, see the same history.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._log = _AppendOnlyLog(path, frozenset({"seal", "read"}), "holdout ledger")

    @property
    def path(self) -> Path:
        return self._log.path

    def __repr__(self) -> str:
        return f"HoldoutLedger({str(self.path)!r})"

    # ------------------------------------------------------------- parsing

    def _seal_from(self, rec: Mapping[str, Any]) -> SealRecord:
        p = self.path
        return SealRecord(
            name=str(_field(rec, "name", p)),
            fingerprint=str(_field(rec, "fingerprint", p)),
            n_bars=int(_field(rec, "n_bars", p)),
            first_time=_field(rec, "first_time", p),
            last_time=_field(rec, "last_time", p),
            created_at=str(_field(rec, "created_at", p)),
            note=str(rec.get("note", "")),
        )

    def _read_from(self, rec: Mapping[str, Any]) -> ReadRecord:
        p = self.path
        result = _field(rec, "result", p)
        return ReadRecord(
            name=str(_field(rec, "name", p)),
            read_number=int(_field(rec, "read_number", p)),
            strategy_id=str(_field(rec, "strategy_id", p)),
            fingerprint=str(_field(rec, "fingerprint", p)),
            read_at=str(_field(rec, "read_at", p)),
            reread=bool(_field(rec, "reread", p)),
            result=None if result is None else dict(result),
        )

    def _seals(self, records: list[dict[str, Any]]) -> dict[str, SealRecord]:
        out: dict[str, SealRecord] = {}
        for rec in records:
            if rec["event"] == "seal":
                seal = self._seal_from(rec)
                if seal.name in out:
                    raise LedgerCorrupted(f"{self.path}: holdout {seal.name!r} is sealed twice")
                out[seal.name] = seal
        return out

    def _reads(self, records: list[dict[str, Any]], name: str) -> list[ReadRecord]:
        return [
            self._read_from(rec)
            for rec in records
            if rec["event"] == "read" and rec.get("name") == name
        ]

    # ----------------------------------------------------------------- API

    def seal(self, name: str, bars: Bars, *, note: str = "") -> SealRecord:
        """Record the fingerprint of a holdout before anyone looks at it.

        Stores the name, :func:`fingerprint`, number of bars, first and last
        timestamp (``str`` of the bar times, or ``None``), the UTC creation
        time in ISO-8601 and ``note``. Sealing a name again with the same
        bars returns the existing record and writes nothing; sealing it with
        different bars raises :class:`HoldoutMismatch`. Sealing bars that are
        already sealed under another name raises :class:`HoldoutError`, so a
        holdout cannot get a fresh read count by being renamed. (Bars that
        differ in a single value have a different fingerprint, so a slightly
        trimmed copy is not caught; that is a limit of any exact hash.)
        """
        name = _check_name(name, "name")
        if not isinstance(bars, Bars):
            raise TypeError(f"expected Bars, got {type(bars).__name__}")
        if len(bars) == 0:
            raise ValueError("cannot seal an empty holdout")
        if not isinstance(note, str):
            raise ValueError(f"note must be a string, got {note!r}")
        fp = fingerprint(bars)
        n_bars = len(bars)
        first = None if bars.time is None else str(bars.time[0])
        last = None if bars.time is None else str(bars.time[-1])
        existing: list[SealRecord] = []

        def build(records: list[dict[str, Any]]) -> dict[str, Any] | None:
            seals = self._seals(records)
            if name in seals:
                old = seals[name]
                if old.fingerprint != fp:
                    raise HoldoutMismatch(
                        f"holdout {name!r} is already sealed with different data "
                        f"({old.n_bars} bars, fingerprint {old.fingerprint[:12]}; "
                        f"got {n_bars} bars, fingerprint {fp[:12]}). Use a new name."
                    )
                existing.append(old)
                return None
            for other in seals.values():
                if other.fingerprint == fp:
                    raise HoldoutError(
                        f"these bars are already sealed as holdout {other.name!r}; sealing "
                        f"them again as {name!r} would hide that holdout's read history"
                    )
            return {
                "event": "seal",
                "v": _FORMAT_VERSION,
                "name": name,
                "fingerprint": fp,
                "n_bars": n_bars,
                "first_time": first,
                "last_time": last,
                "created_at": _now(),
                "note": note,
            }

        event = self._log.append(build)
        if event is None:
            return existing[0]
        return self._seal_from(event)

    def read(
        self,
        name: str,
        bars: Bars,
        *,
        strategy_id: str,
        result: Any = None,
        allow_reread: bool = False,
    ) -> ReadRecord:
        """Record that ``strategy_id`` was evaluated on the holdout ``name``.

        Checks, in order, that ``name`` was sealed, that ``bars`` has the
        sealed fingerprint (raises :class:`HoldoutMismatch` if a single value
        differs) and that the holdout has not been read before (raises
        :class:`HoldoutAlreadyRead` unless ``allow_reread=True``). Only then
        does it append a read event with the read number, ``strategy_id``,
        the UTC time and ``result``.

        ``result`` is ``None``, a dict, or an object with ``as_dict()`` such as
        :class:`~backtest_skeptic.Summary`. numpy values, tuples, dates and
        nested result objects are converted; non-finite floats are stored as the strings
        ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``. A failed check writes
        nothing.
        """
        name = _check_name(name, "name")
        strategy_id = _check_name(strategy_id, "strategy_id")
        if not isinstance(bars, Bars):
            raise TypeError(f"expected Bars, got {type(bars).__name__}")
        stored_result = None
        if result is not None:
            stored_result = _jsonable(result, "result")
            if not isinstance(stored_result, dict):
                raise ValueError(
                    "result must be a dict or an object with as_dict(), "
                    f"got {type(result).__name__}"
                )
        fp = fingerprint(bars)

        def build(records: list[dict[str, Any]]) -> dict[str, Any]:
            seals = self._seals(records)
            if name not in seals:
                known = ", ".join(repr(k) for k in seals) or "none"
                raise HoldoutError(
                    f"no holdout named {name!r} is sealed in {self.path} (sealed: {known})"
                )
            seal = seals[name]
            if seal.fingerprint != fp:
                raise HoldoutMismatch(
                    f"the bars passed for holdout {name!r} differ from the sealed data "
                    f"(sealed {seal.n_bars} bars, fingerprint {seal.fingerprint[:12]}; "
                    f"got {len(bars)} bars, fingerprint {fp[:12]})"
                )
            previous = self._reads(records, name)
            if previous and not allow_reread:
                last = previous[-1]
                raise HoldoutAlreadyRead(
                    f"holdout {name!r} was already read {len(previous)} time(s), last by "
                    f"{last.strategy_id!r} at {last.read_at}. A second look turns the "
                    "holdout into training data; pass allow_reread=True to record a "
                    "deliberate re-read, which the ledger flags."
                )
            return {
                "event": "read",
                "v": _FORMAT_VERSION,
                "name": name,
                "read_number": len(previous) + 1,
                "strategy_id": strategy_id,
                "fingerprint": fp,
                "read_at": _now(),
                "reread": bool(previous),
                "result": stored_result,
            }

        event = self._log.append(build)
        assert event is not None
        return self._read_from(event)

    def reads(self, name: str) -> tuple[ReadRecord, ...]:
        """Every recorded read of holdout ``name``, oldest first."""
        records, _ = self._log.load()
        return tuple(self._reads(records, name))

    def status(self) -> LedgerStatus:
        """Every sealed holdout with its read count."""
        records, head = self._log.load()
        seals = self._seals(records)
        holdouts = []
        for seal in seals.values():
            reads = self._reads(records, seal.name)
            strategies = tuple(dict.fromkeys(r.strategy_id for r in reads))
            holdouts.append(
                HoldoutStatus(
                    name=seal.name,
                    fingerprint=seal.fingerprint,
                    n_bars=seal.n_bars,
                    first_time=seal.first_time,
                    last_time=seal.last_time,
                    sealed_at=seal.created_at,
                    note=seal.note,
                    n_reads=len(reads),
                    strategies=strategies,
                    last_read_at=reads[-1].read_at if reads else None,
                )
            )
        return LedgerStatus(
            path=str(self.path), n_events=len(records), head=head, holdouts=tuple(holdouts)
        )


# ---------------------------------------------------------------------------
# trial registry
# ---------------------------------------------------------------------------


def _params_hash(params: Mapping[str, Any] | None, name: str) -> tuple[dict[str, Any] | None, str]:
    """Canonical-JSON SHA-256 of ``params``; the trial name stands in when there are none."""
    if params is None:
        payload: Any = {"__trial_name__": name}
        stored = None
    else:
        if not isinstance(params, Mapping):
            raise ValueError(
                f"params must be a mapping (dict) or None, got {type(params).__name__}"
            )
        stored = _jsonable(params, "params")
        payload = stored
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return stored, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialRecord:
    """One configuration tried, with its per-period Sharpe ratio."""

    record_number: int
    name: str
    params_hash: str
    params: dict[str, Any] | None
    sharpe: float
    n_obs: int | None
    source: str
    recorded_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_number": self.record_number,
            "name": self.name,
            "params_hash": self.params_hash,
            "params": None if self.params is None else json.loads(json.dumps(self.params)),
            "sharpe": self.sharpe,
            "n_obs": self.n_obs,
            "source": self.source,
            "recorded_at": self.recorded_at,
        }

    def __str__(self) -> str:
        n = "" if self.n_obs is None else f", n={self.n_obs}"
        return (
            f"trial #{self.record_number} {self.name!r}: Sharpe {self.sharpe:.4f} per period{n} "
            f"(params {self.params_hash[:12]})"
        )


@dataclass(frozen=True)
class TrialSummary:
    """The deflated-Sharpe inputs read off a trial registry.

    ``n_trials`` is the number of distinct configurations and
    ``sharpe_variance`` the ``ddof=1`` variance of their per-period Sharpe
    ratios (``None`` with fewer than two). ``best`` is the configuration
    with the highest Sharpe ratio.
    """

    path: str
    n_records: int
    n_trials: int
    sharpe_variance: float | None
    best: TrialRecord | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_records": self.n_records,
            "n_trials": self.n_trials,
            "sharpe_variance": self.sharpe_variance,
            "best": None if self.best is None else self.best.as_dict(),
        }

    def __str__(self) -> str:
        if self.n_trials == 0:
            return f"trial registry {self.path}: no trials recorded"
        var = "n/a (fewer than 2 trials)"
        if self.sharpe_variance is not None:
            var = f"{self.sharpe_variance:.6g}"
        lines = [
            f"trial registry {self.path}: {self.n_trials} distinct configurations "
            f"({self.n_records} records), variance of per-period Sharpe {var}",
        ]
        if self.best is not None:
            lines.append(f"  best: {self.best}")
        if self.sharpe_variance is not None:
            lines.append(
                f"  deflate the best with n_trials={self.n_trials}, "
                f"var_trials_sharpe={self.sharpe_variance:.6g}"
            )
        return "\n".join(lines)


class TrialRegistry:
    """Append-only record of every configuration tried during research.

    Call :meth:`record` for every backtest you run, including the ones you
    throw away. At the end, :attr:`n_trials` and :attr:`sharpe_variance`
    are exactly the ``N`` and ``V[SR]`` that the deflated Sharpe ratio
    needs (Bailey & Lopez de Prado, 2014)::

        reg = TrialRegistry("trials.jsonl")
        for params in grid:
            r = backtest(train, **params).r
            reg.record("breakout", r, params=params)
        deflated_sharpe_ratio(best_returns, n_trials=reg.n_trials,
                              var_trials_sharpe=reg.sharpe_variance)

    A configuration is identified by the SHA-256 of its ``params`` as
    canonical JSON (sorted keys), or by its name when ``params`` is None.
    Re-running a configuration does not add a trial; the latest record of
    each configuration is the one used for the variance. Note that
    ``{"n": 14}`` and ``{"n": 14.0}`` hash differently.

    Limitations: ``N`` counts distinct configurations, which overstates the
    number of *independent* trials when configurations are correlated (a
    lookback of 20 and of 21 are nearly the same strategy); the deflated
    Sharpe ratio is then conservative. Trials you ran but did not record
    are invisible, which makes it optimistic. All Sharpe ratios must be in
    the same units (per period, same sampling frequency).
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._log = _AppendOnlyLog(path, frozenset({"trial"}), "trial registry")

    @property
    def path(self) -> Path:
        return self._log.path

    def __repr__(self) -> str:
        return f"TrialRegistry({str(self.path)!r})"

    def _trial_from(self, rec: Mapping[str, Any], number: int) -> TrialRecord:
        p = self.path
        n_obs = _field(rec, "n_obs", p)
        params = _field(rec, "params", p)
        return TrialRecord(
            record_number=number,
            name=str(_field(rec, "name", p)),
            params_hash=str(_field(rec, "params_hash", p)),
            params=None if params is None else dict(params),
            sharpe=float(_field(rec, "sharpe", p)),
            n_obs=None if n_obs is None else int(n_obs),
            source=str(_field(rec, "source", p)),
            recorded_at=str(_field(rec, "recorded_at", p)),
        )

    def record(
        self,
        name: str,
        returns: ArrayLike | None = None,
        *,
        sharpe: float | None = None,
        n_obs: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> TrialRecord:
        """Append one trial.

        Pass either ``returns`` (per-period returns or per-trade R; the
        Sharpe ratio ``mean / std`` with ``ddof=1`` and ``n_obs`` are computed
        from them) or a precomputed per-period ``sharpe`` with an optional
        ``n_obs``. Passing both forms raises, as do non-finite returns, fewer
        than two returns and constant returns.
        """
        name = _check_name(name, "name")
        if returns is not None:
            if sharpe is not None or n_obs is not None:
                raise ValueError(
                    "pass returns, or sharpe (with an optional n_obs), not both; "
                    "the Sharpe ratio and n_obs are computed from returns"
                )
            x = np.asarray(returns, dtype=np.float64)
            if x.ndim != 1:
                raise ValueError(f"returns must be one-dimensional, got shape {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("returns must be finite (no NaN or inf)")
            if len(x) < 2:
                raise ValueError(f"at least 2 returns are required, got {len(x)}")
            sd = float(x.std(ddof=1))
            if sd <= _REL_TOL * float(np.abs(x).max()):
                raise ValueError("returns are constant; the Sharpe ratio is undefined")
            sr = float(x.mean()) / sd
            nobs: int | None = len(x)
            source = "returns"
        else:
            if sharpe is None:
                raise ValueError("pass returns or a per-period sharpe")
            if isinstance(sharpe, bool):
                raise ValueError(f"sharpe must be a finite number, got {sharpe!r}")
            sr = float(sharpe)
            if not math.isfinite(sr):
                raise ValueError(f"sharpe must be finite, got {sharpe!r}")
            nobs = None
            if n_obs is not None:
                nobs = _int_arg(n_obs, "n_obs")
                if nobs < 2:
                    raise ValueError(f"n_obs must be >= 2, got {nobs}")
            source = "sharpe"
        stored_params, digest = _params_hash(params, name)
        number: list[int] = []

        def build(records: list[dict[str, Any]]) -> dict[str, Any]:
            number.append(len(records) + 1)
            return {
                "event": "trial",
                "v": _FORMAT_VERSION,
                "name": name,
                "params_hash": digest,
                "params": stored_params,
                "sharpe": sr,
                "n_obs": nobs,
                "source": source,
                "recorded_at": _now(),
            }

        event = self._log.append(build)
        assert event is not None
        return self._trial_from(event, number[0])

    def trials(self) -> tuple[TrialRecord, ...]:
        """Every recorded trial, oldest first (re-runs included)."""
        records, _ = self._log.load()
        return tuple(self._trial_from(rec, i) for i, rec in enumerate(records, start=1))

    def _distinct(self) -> list[TrialRecord]:
        latest: dict[str, TrialRecord] = {}
        for trial in self.trials():
            latest.pop(trial.params_hash, None)
            latest[trial.params_hash] = trial
        return list(latest.values())

    @property
    def n_trials(self) -> int:
        """Number of distinct configurations (unique params hashes)."""
        return len(self._distinct())

    @property
    def sharpe_variance(self) -> float:
        """``ddof=1`` variance of the per-period Sharpe ratios across configurations.

        Uses the latest record of each configuration. Raises ``ValueError``
        with fewer than two configurations, where the variance is undefined.
        """
        distinct = self._distinct()
        if len(distinct) < 2:
            raise ValueError(
                f"the Sharpe variance needs at least 2 distinct trials, have {len(distinct)}"
            )
        return float(np.var([t.sharpe for t in distinct], ddof=1))

    def summary(self) -> TrialSummary:
        """Counts, variance and the best configuration in one result."""
        all_trials = self.trials()
        distinct = self._distinct()
        var = None
        if len(distinct) >= 2:
            var = float(np.var([t.sharpe for t in distinct], ddof=1))
        best = max(distinct, key=lambda t: t.sharpe) if distinct else None
        return TrialSummary(
            path=str(self.path),
            n_records=len(all_trials),
            n_trials=len(distinct),
            sharpe_variance=var,
            best=best,
        )
