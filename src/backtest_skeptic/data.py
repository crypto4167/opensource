"""OHLC bar container with validation.

Every check in this package reads prices through :class:`Bars`. The arrays are
float64, one-dimensional, read-only and checked on construction, so a bad cell
(NaN, inf, a high below the close) fails loudly instead of silently turning a
backtest's results into NaN or zero trades further down the line.

Bars are treated as **mid prices**. Stops and targets trigger on these prices;
the spread is charged separately through :class:`backtest_skeptic.Costs`. If
your bars are bid-only or ask-only, adjust the half-spread accordingly.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Bars", "DataError"]

FloatArray = NDArray[np.float64]

_TIME_COLUMNS = ("time", "datetime", "date", "timestamp")
_MAX_REPORTED = 10


class DataError(ValueError):
    """Raised when bar data fails validation."""


def _as_float_array(values: ArrayLike, name: str) -> FloatArray:
    try:
        # Copy, so freezing the array below never freezes the caller's data.
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{name}: values are not numeric ({exc})") from exc
    if arr.ndim != 1:
        raise DataError(f"{name}: expected a one-dimensional array, got shape {arr.shape}")
    return arr


def _describe(indices: NDArray[np.intp]) -> str:
    shown = ", ".join(str(int(i)) for i in indices[:_MAX_REPORTED])
    more = len(indices) - _MAX_REPORTED
    return shown + (f" and {more} more" if more > 0 else "")


class Bars:
    """Validated, immutable OHLC bars.

    Parameters
    ----------
    open, high, low, close:
        Prices, one value per bar, oldest first.
    volume:
        Optional volume per bar.
    time:
        Optional timestamps, oldest first. Any sortable values work (numbers,
        ``datetime64``, ISO-8601 strings). They must be strictly increasing.
    validate:
        Check finiteness and OHLC consistency (``low <= min(open, close)`` and
        ``high >= max(open, close)``). Leave this on unless you have already
        validated the data.
    """

    __slots__ = ("close", "high", "low", "open", "time", "volume")

    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray | None
    time: NDArray[Any] | None

    def __init__(
        self,
        open: ArrayLike,
        high: ArrayLike,
        low: ArrayLike,
        close: ArrayLike,
        volume: ArrayLike | None = None,
        time: ArrayLike | None = None,
        *,
        validate: bool = True,
    ) -> None:
        o = _as_float_array(open, "open")
        h = _as_float_array(high, "high")
        lo = _as_float_array(low, "low")
        c = _as_float_array(close, "close")
        n = len(o)
        for name, arr in (("high", h), ("low", lo), ("close", c)):
            if len(arr) != n:
                raise DataError(f"{name} has {len(arr)} values but open has {n}")
        v = None
        if volume is not None:
            v = _as_float_array(volume, "volume")
            if len(v) != n:
                raise DataError(f"volume has {len(v)} values but open has {n}")
        t = None
        if time is not None:
            t = np.array(time)
            if t.ndim != 1 or len(t) != n:
                raise DataError(f"time must be one-dimensional with {n} values")

        if validate:
            stacked = np.vstack([o, h, lo, c])
            bad = np.flatnonzero(~np.isfinite(stacked).all(axis=0))
            if len(bad):
                raise DataError(f"non-finite OHLC value at bar(s) {_describe(bad)}")
            if v is not None:
                bad = np.flatnonzero(~np.isfinite(v) | (v < 0))
                if len(bad):
                    raise DataError(f"negative or non-finite volume at bar(s) {_describe(bad)}")
            bad = np.flatnonzero((h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (lo > h))
            if len(bad):
                raise DataError(
                    "inconsistent OHLC (high below open/close or low above open/close) "
                    f"at bar(s) {_describe(bad)}"
                )
            if t is not None and n > 1:
                try:
                    increasing = t[1:] > t[:-1]
                except TypeError as exc:
                    raise DataError(f"time values are not comparable ({exc})") from exc
                bad = np.flatnonzero(~np.asarray(increasing, dtype=bool)) + 1
                if len(bad):
                    raise DataError(f"time is not strictly increasing at bar(s) {_describe(bad)}")

        frozen: list[NDArray[Any] | None] = [o, h, lo, c, v, t]
        for item in frozen:
            if item is not None:
                item.setflags(write=False)
        self._set(o, h, lo, c, v, t)

    def _set(
        self,
        o: FloatArray,
        h: FloatArray,
        lo: FloatArray,
        c: FloatArray,
        v: FloatArray | None,
        t: NDArray[Any] | None,
    ) -> None:
        object.__setattr__(self, "open", o)
        object.__setattr__(self, "high", h)
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "close", c)
        object.__setattr__(self, "volume", v)
        object.__setattr__(self, "time", t)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Bars is immutable")

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "Bars(n=0)"
        span = ""
        if self.time is not None:
            span = f", time={self.time[0]!s}..{self.time[-1]!s}"
        return f"Bars(n={n}{span}, last_close={self.close[-1]:g})"

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self.open)

    def __getitem__(self, key: slice) -> Bars:
        """Return a contiguous slice of bars (``bars[a:b]``) without copying."""
        if not isinstance(key, slice):
            raise TypeError("Bars only supports slicing, e.g. bars[:100]")
        if key.step not in (None, 1):
            raise ValueError("Bars slices must be contiguous (step 1)")
        return self._view(key)

    def prefix(self, stop: int) -> Bars:
        """Bars ``0 .. stop-1``: everything known after bar ``stop-1`` closed."""
        return self._view(slice(0, stop))

    def _view(self, key: slice) -> Bars:
        new = object.__new__(Bars)
        new._set(
            self.open[key],
            self.high[key],
            self.low[key],
            self.close[key],
            None if self.volume is None else self.volume[key],
            None if self.time is None else self.time[key],
        )
        return new

    def index_of(self, when: Any) -> int:
        """Index of the bar whose timestamp equals ``when``."""
        if self.time is None:
            raise DataError("these bars have no timestamps")
        pos = int(np.searchsorted(self.time, when, side="left"))
        if pos >= len(self) or self.time[pos] != when:
            raise KeyError(f"no bar with time {when!r}")
        return pos

    def with_prices(
        self,
        open: ArrayLike,
        high: ArrayLike,
        low: ArrayLike,
        close: ArrayLike,
    ) -> Bars:
        """Copy of these bars with new prices and the same volume and timestamps."""
        return Bars(open, high, low, close, volume=self.volume, time=self.time)

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        time_key: str | None = None,
    ) -> Bars:
        """Build bars from dicts with ``open``, ``high``, ``low``, ``close`` keys.

        ``volume`` and a time key (``time``, ``datetime``, ``date`` or
        ``timestamp``, or the one named by ``time_key``) are optional.
        """
        rows = list(records)
        if not rows:
            raise DataError("no records")
        first = rows[0]
        key = time_key or next((k for k in _TIME_COLUMNS if k in first), None)
        try:
            o = [float(r["open"]) for r in rows]
            h = [float(r["high"]) for r in rows]
            lo = [float(r["low"]) for r in rows]
            c = [float(r["close"]) for r in rows]
        except KeyError as exc:
            raise DataError(f"record is missing field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise DataError(f"record has a non-numeric price ({exc})") from exc
        v = [float(r.get("volume") or 0.0) for r in rows] if "volume" in first else None
        t = [r[key] for r in rows] if key is not None else None
        return cls(o, h, lo, c, volume=v, time=None if t is None else np.asarray(t))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        time_column: str | None = None,
        columns: Mapping[str, str] | None = None,
    ) -> Bars:
        """Load bars from a CSV file with a header row.

        Column names are matched case-insensitively. ``columns`` maps the
        standard names (``open``, ``high``, ``low``, ``close``, ``volume``,
        ``time``) to the names used in the file, e.g. ``{"close": "Adj Close"}``.
        Rows are never dropped silently: an unparseable row raises
        :class:`DataError` with its line number.
        """
        mapping = {k.lower(): v for k, v in (columns or {}).items()}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise DataError(f"{path}: empty file")
            lookup = {name.strip().lower(): name for name in reader.fieldnames}

            def column(std: str, required: bool) -> str | None:
                wanted = mapping.get(std, std).strip().lower()
                if wanted in lookup:
                    return lookup[wanted]
                if required:
                    raise DataError(f"{path}: no '{mapping.get(std, std)}' column")
                return None

            names = {std: column(std, True) for std in ("open", "high", "low", "close")}
            vol_col = column("volume", False)
            time_col: str | None
            if time_column is not None or "time" in mapping:
                time_col = (
                    column("time", True) if time_column is None else lookup.get(time_column.lower())
                )
                if time_col is None:
                    raise DataError(f"{path}: no '{time_column}' column")
            else:
                time_col = next((lookup[k] for k in _TIME_COLUMNS if k in lookup), None)

            cols: dict[str, list[float]] = {k: [] for k in ("open", "high", "low", "close")}
            vol: list[float] = []
            times: list[str] = []
            for line, row in enumerate(reader, start=2):
                try:
                    for std, name in names.items():
                        assert name is not None
                        cols[std].append(float(row[name]))
                    if vol_col is not None:
                        raw = row[vol_col]
                        vol.append(float(raw) if raw not in (None, "") else 0.0)
                except (TypeError, ValueError) as exc:
                    raise DataError(f"{path}, line {line}: {exc}") from exc
                if time_col is not None:
                    times.append(row[time_col])
        if not cols["open"]:
            raise DataError(f"{path}: no data rows")
        return cls(
            cols["open"],
            cols["high"],
            cols["low"],
            cols["close"],
            volume=vol if vol_col is not None else None,
            time=np.asarray(times) if time_col is not None else None,
        )

    @classmethod
    def from_dataframe(cls, df: Any, *, use_index_as_time: bool = True) -> Bars:
        """Build bars from a pandas-like DataFrame.

        Columns are matched case-insensitively (``Open``, ``open`` ...). The
        index is used as the timestamp unless ``use_index_as_time`` is False.
        pandas is not imported; any object with ``columns``, ``__getitem__`` and
        ``index`` works.
        """
        lookup = {str(c).strip().lower(): c for c in df.columns}

        def col(name: str) -> Any:
            if name not in lookup:
                raise DataError(f"DataFrame has no '{name}' column")
            return np.asarray(df[lookup[name]], dtype=np.float64)

        volume = col("volume") if "volume" in lookup else None
        time = None
        if use_index_as_time:
            time = np.asarray(df.index)
        else:
            key = next((k for k in _TIME_COLUMNS if k in lookup), None)
            if key is not None:
                time = np.asarray(df[lookup[key]])
        return cls(col("open"), col("high"), col("low"), col("close"), volume=volume, time=time)

    @classmethod
    def from_close(cls, close: Sequence[float] | ArrayLike) -> Bars:
        """Bars where open, high, low and close are all the closing price.

        Useful for tests and for strategies that only ever look at closes.
        Note that stops and targets can then only trigger on closes.
        """
        c = _as_float_array(close, "close")
        return cls(c, c, c, c)
