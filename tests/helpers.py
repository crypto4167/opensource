"""Small fixtures shared by the tests."""

from __future__ import annotations

from collections.abc import Sequence

from backtest_skeptic import Bars


def bars_from_rows(rows: Sequence[tuple[float, float, float, float]]) -> Bars:
    """Bars from (open, high, low, close) tuples."""
    o, h, lo, c = zip(*rows, strict=True)
    return Bars(o, h, lo, c)


def flat_then(rows: Sequence[tuple[float, float, float, float]], lead: int = 1) -> Bars:
    """``lead`` flat bars at 100 followed by ``rows``."""
    return bars_from_rows([(100.0, 100.0, 100.0, 100.0)] * lead + list(rows))
