"""Synthetic bars for null worlds and positive controls.

A check that never fails proves nothing, so every check in this package is
tested two ways: on a driftless random walk, where there is nothing to find
and it must report no edge, and on data with a planted edge or a deliberate
lookahead, where it must find the problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import Bars

__all__ = ["oracle_signal", "planted_signal_bars", "random_walk_bars"]


def _times(n: int, start_time: Any, interval: Any) -> NDArray[Any] | None:
    if start_time is None:
        return None
    start = np.datetime64(start_time, "s")
    step = _parse_interval(interval) if isinstance(interval, str) else np.timedelta64(interval)
    out: NDArray[Any] = start + step * np.arange(n)
    return out


def _parse_interval(text: str) -> np.timedelta64:
    units = {"s": "s", "m": "m", "min": "m", "h": "h", "d": "D"}
    num = "".join(ch for ch in text if ch.isdigit()) or "1"
    unit = text[len(num) :] if text.startswith(num) else text.lstrip("0123456789")
    if unit not in units:
        raise ValueError(f"unknown interval {text!r}; use e.g. '15m', '1h', '1d'")
    seconds = {"s": 1, "m": 60, "h": 3600, "D": 86400}[units[unit]]
    return np.timedelta64(int(num) * seconds, "s")


def _bars_from_log_increments(
    increments: NDArray[np.float64],
    gaps: NDArray[np.float64],
    start: float,
    volume: NDArray[np.float64],
    time: NDArray[Any] | None,
    rng: np.random.Generator,
    step_sd: float,
) -> Bars:
    """Assemble bars from log-price increments.

    Highs and lows are drawn from the exact distribution of a Brownian
    bridge's maximum and minimum between consecutive simulated points, so the
    bars behave as if the price moved continuously inside each bar. A level is
    then touched exactly when the high or low reaches it, which is the
    assumption the engine's fills rest on. Without this, discrete sampling
    misses touches and creates a small spurious edge for wide targets.
    """
    n, steps = increments.shape
    path = np.cumsum(increments, axis=1)
    bar_move = path[:, -1]
    log_open = np.log(start) + np.concatenate([[0.0], np.cumsum(bar_move[:-1] + gaps[1:])])
    points = np.concatenate([log_open[:, None], log_open[:, None] + path], axis=1)
    x0, x1 = points[:, :-1], points[:, 1:]
    var = step_sd**2
    spread_hi = np.sqrt((x1 - x0) ** 2 - 2.0 * var * np.log(rng.random((n, steps))))
    spread_lo = np.sqrt((x1 - x0) ** 2 - 2.0 * var * np.log(rng.random((n, steps))))
    log_high = ((x0 + x1 + spread_hi) / 2.0).max(axis=1)
    log_low = ((x0 + x1 - spread_lo) / 2.0).min(axis=1)
    log_close = points[:, -1]
    return Bars(
        np.exp(log_open),
        np.exp(np.maximum(log_high, np.maximum(log_open, log_close))),
        np.exp(np.minimum(log_low, np.minimum(log_open, log_close))),
        np.exp(log_close),
        volume=volume,
        time=time,
    )


def random_walk_bars(
    n: int,
    *,
    start: float = 100.0,
    vol: float = 0.01,
    drift: float = 0.0,
    gap_vol: float = 0.0,
    substeps: int = 8,
    seed: int | None = None,
    start_time: Any = "2020-01-01T00:00",
    interval: Any = "1h",
) -> Bars:
    """Geometric random-walk bars with intrabar highs and lows.

    Parameters
    ----------
    n: number of bars.
    vol: standard deviation of each bar's log return.
    drift: expected log return per bar on top of the martingale correction.
        With ``drift=0`` the price is a martingale, so no entry rule has an
        edge: a bracket with stop distance ``a`` and target distance ``b``
        hits its target with probability ``a / (a + b)`` (up to the rare bar
        that touches both, which the engine books as a stop).
    gap_vol: standard deviation of the log gap between one bar's close and the
        next bar's open (0 means no gaps).
    substeps: price points simulated inside each bar. Highs and lows come from
        the continuous path between these points, so a few are enough.
    seed: seed for :func:`numpy.random.default_rng`.
    start_time, interval: timestamps for the bars (``None`` for no timestamps).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if vol < 0 or gap_vol < 0 or substeps < 1:
        raise ValueError("vol and gap_vol must be >= 0 and substeps >= 1")
    rng = np.random.default_rng(seed)
    step_sd = vol / np.sqrt(substeps)
    step_mean = (drift - 0.5 * vol**2) / substeps
    increments = rng.normal(step_mean, step_sd, size=(n, substeps))
    gaps = rng.normal(-0.5 * gap_vol**2, gap_vol, size=n) if gap_vol > 0 else np.zeros(n)
    volume = rng.uniform(1_000.0, 10_000.0, size=n).round()
    times = _times(n, start_time, interval)
    return _bars_from_log_increments(increments, gaps, start, volume, times, rng, step_sd)


def planted_signal_bars(
    n: int,
    *,
    edge: float = 0.2,
    signal_prob: float = 0.1,
    start: float = 100.0,
    vol: float = 0.01,
    substeps: int = 8,
    seed: int | None = None,
    start_time: Any = "2020-01-01T00:00",
    interval: Any = "1h",
) -> tuple[Bars, NDArray[np.int64]]:
    """Random-walk bars with a genuine, known edge.

    Returns ``(bars, signal)``. ``signal[i]`` is ``+1``, ``-1`` or ``0`` and is
    known at the close of bar ``i``. During the next bar (bar ``i + 1``) the
    price drifts in the signal's direction by ``edge`` times
    the bar volatility. Use it as a positive control: a check that cannot see
    this edge cannot be trusted to clear a real one.
    """
    if not 0 < signal_prob <= 1:
        raise ValueError("signal_prob must be in (0, 1]")
    rng = np.random.default_rng(seed)
    signal = np.zeros(n, dtype=np.int64)
    fires = rng.random(n) < signal_prob
    signal[fires] = rng.choice(np.array([-1, 1]), size=int(fires.sum()))
    signal[-1] = 0
    step_sd = vol / np.sqrt(substeps)
    bar_drift = np.zeros(n)
    bar_drift[1:] = edge * vol * signal[:-1]
    step_mean = (bar_drift[:, None] - 0.5 * vol**2) / substeps
    increments = rng.normal(0.0, step_sd, size=(n, substeps)) + step_mean
    volume = rng.uniform(1_000.0, 10_000.0, size=n).round()
    times = _times(n, start_time, interval)
    bars = _bars_from_log_increments(increments, np.zeros(n), start, volume, times, rng, step_sd)
    return bars, signal


def oracle_signal(bars: Bars, horizon: int = 1) -> NDArray[np.int64]:
    """A signal that cheats by reading the future.

    ``+1`` at bar ``i`` if ``close[i + horizon] > close[i]``, else ``-1``, and
    ``0`` where the future is not available. It exists only to prove that the
    lookahead tests and the engine can catch and exploit a leak; never trade it.
    """
    close = bars.close
    out = np.zeros(len(close), dtype=np.int64)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(close) > horizon:
        out[:-horizon] = np.where(close[horizon:] > close[:-horizon], 1, -1)
    return out
