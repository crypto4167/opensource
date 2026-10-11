"""Summary statistics for a list of trades measured in R (units of initial risk)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .trades import Trade

__all__ = ["Summary", "max_drawdown", "profit_factor", "summarize", "t_statistic"]


def t_statistic(values: ArrayLike) -> float | None:
    """One-sample t-statistic of the mean against zero.

    Returns ``None`` for fewer than two values. When every value is the same
    (up to floating-point noise) the t-statistic is ``+inf``/``-inf`` for a
    non-zero mean and ``0.0`` for a zero mean, instead of a huge number that
    depends on rounding.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n < 2:
        return None
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    scale = max(1.0, float(np.abs(x).max()))
    if sd <= 1e-12 * scale:
        if abs(mean) <= 1e-12 * scale:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / (sd / math.sqrt(n))


def profit_factor(values: ArrayLike) -> float | None:
    """Gross gains divided by gross losses.

    ``inf`` when there are gains and no losses, ``None`` when there are
    neither (an all-zero or empty list has no defined profit factor).
    """
    x = np.asarray(values, dtype=np.float64)
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses > 0:
        return gains / losses
    return math.inf if gains > 0 else None


def max_drawdown(values: ArrayLike) -> float:
    """Largest peak-to-trough fall of the cumulative sum, starting from 0.

    The starting point counts as a peak, so a loss on the very first trade is
    a drawdown. Returned as a non-negative number in the units of ``values``.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


@dataclass(frozen=True)
class Summary:
    """Trade-list statistics. All R figures are net of costs."""

    trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float | None
    expectancy_r: float | None
    total_r: float
    std_r: float | None
    t_stat: float | None
    profit_factor: float | None
    avg_win_r: float | None
    avg_loss_r: float | None
    max_drawdown_r: float
    long_share: float | None
    exits: dict[str, int]
    ambiguous: int
    avg_bars_held: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        if self.trades == 0:
            return "0 trades"

        def fmt(x: float | None, spec: str = ".3f") -> str:
            return "n/a" if x is None else format(x, spec)

        return (
            f"{self.trades} trades, win rate {fmt(self.win_rate, '.1%')}, "
            f"expectancy {fmt(self.expectancy_r, '+.3f')}R, total {self.total_r:+.2f}R, "
            f"t {fmt(self.t_stat, '.2f')}, PF {fmt(self.profit_factor, '.2f')}, "
            f"max DD {self.max_drawdown_r:.2f}R"
        )


def summarize(trades: Iterable[Trade] | ArrayLike) -> Summary:
    """Summarise trades (a sequence of :class:`Trade`) or a plain array of R values.

    A trade with ``r == 0`` counts as breakeven, neither win nor loss; the win
    rate is ``wins / trades``.
    """
    items: Sequence[Any] = (
        trades.ravel().tolist() if isinstance(trades, np.ndarray) else list(trades)  # type: ignore[arg-type]
    )
    trade_objs = [t for t in items if isinstance(t, Trade)]
    if trade_objs and len(trade_objs) != len(items):
        raise TypeError("pass either Trade objects or plain numbers, not a mix")
    r: NDArray[np.float64]
    if trade_objs:
        r = np.array([t.r for t in trade_objs], dtype=np.float64)
    else:
        r = np.asarray(items, dtype=np.float64).ravel()
    if not np.isfinite(r).all():
        raise ValueError("R values must be finite")

    n = len(r)
    wins = r > 0
    losses = r < 0
    exits: Counter[str] = Counter(t.reason for t in trade_objs)
    return Summary(
        trades=n,
        wins=int(wins.sum()),
        losses=int(losses.sum()),
        breakeven=int(n - wins.sum() - losses.sum()),
        win_rate=float(wins.mean()) if n else None,
        expectancy_r=float(r.mean()) if n else None,
        total_r=float(r.sum()),
        std_r=float(r.std(ddof=1)) if n > 1 else None,
        t_stat=t_statistic(r),
        profit_factor=profit_factor(r) if n else None,
        avg_win_r=float(r[wins].mean()) if wins.any() else None,
        avg_loss_r=float(r[losses].mean()) if losses.any() else None,
        max_drawdown_r=max_drawdown(r),
        long_share=(float(np.mean([t.side > 0 for t in trade_objs])) if trade_objs else None),
        exits=dict(exits),
        ambiguous=sum(1 for t in trade_objs if t.ambiguous),
        avg_bars_held=(float(np.mean([t.bars_held for t in trade_objs])) if trade_objs else None),
    )
