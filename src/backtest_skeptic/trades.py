"""Trade records, bracket setups and the cost model shared by every check."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

__all__ = ["LONG", "SHORT", "Costs", "ExitReason", "Setup", "Trade", "parse_setup", "parse_side"]

LONG = 1
SHORT = -1

ExitReason = Literal["stop", "target", "time", "end"]

_SIDE_WORDS = {
    "buy": LONG,
    "long": LONG,
    "b": LONG,
    "sell": SHORT,
    "short": SHORT,
    "s": SHORT,
}


def _real(value: Any, name: str) -> float:
    """Convert to a finite float, rejecting bools, strings and None.

    numpy scalars of any width and ``Decimal`` are accepted.
    """
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def parse_side(value: Any) -> int:
    """Normalise a trade direction to ``+1`` (long) or ``-1`` (short).

    Accepts ``+1/-1``, ``"BUY"/"SELL"``, ``"long"/"short"`` (any case).
    """
    if isinstance(value, str):
        side = _SIDE_WORDS.get(value.strip().lower())
        if side is None:
            raise ValueError(f"unknown side {value!r}")
        return side
    if isinstance(value, bool):
        raise ValueError(f"unknown side {value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"unknown side {value!r}") from None
    if num == 1.0:
        return LONG
    if num == -1.0:
        return SHORT
    raise ValueError(f"side must be +1 or -1, got {value!r}")


@dataclass(frozen=True)
class Costs:
    """Trading costs, charged identically to a strategy and to every control.

    Parameters
    ----------
    half_spread:
        Half the bid/ask spread in price units. Paid on entry and on exit, so a
        round trip costs one full spread.
    fee_rate:
        Commission as a fraction of traded notional, per side (``0.0005`` is
        5 basis points per side).
    slippage:
        Extra adverse price movement, in price units, on market-style fills:
        entries, stop exits, time exits and end-of-data exits. Target exits
        are treated as resting limit orders and get no slippage.
    """

    half_spread: float = 0.0
    fee_rate: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        for name in ("half_spread", "fee_rate", "slippage"):
            value = _real(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)

    def scaled(self, factor: float) -> Costs:
        """Every cost component multiplied by ``factor`` (for cost stress tests)."""
        factor = _real(factor, "factor")
        if factor < 0:
            raise ValueError("factor must be >= 0")
        return replace(
            self,
            half_spread=self.half_spread * factor,
            fee_rate=self.fee_rate * factor,
            slippage=self.slippage * factor,
        )


@dataclass(frozen=True)
class Setup:
    """A bracket order: direction plus absolute stop and target price levels."""

    side: int
    stop: float
    target: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", parse_side(self.side))
        object.__setattr__(self, "stop", _real(self.stop, "stop"))
        object.__setattr__(self, "target", _real(self.target, "target"))


def parse_setup(value: Any) -> Setup | None:
    """Turn whatever a ``decide`` callback returned into a :class:`Setup`.

    Accepted forms: ``None`` or ``False`` (no trade), a :class:`Setup`, a
    ``(side, stop, target)`` tuple, or a mapping with a side key (``side`` or
    ``signal``), a stop key (``stop`` or ``stop_loss``) and a target key
    (``target`` or ``take_profit``). A mapping whose side is ``None``, ``0``
    or ``"HOLD"`` means no trade.
    """
    if value is None or value is False:
        return None
    if isinstance(value, Setup):
        return value
    if isinstance(value, Mapping):
        side = value.get("side", value.get("signal"))
        if side is None or side == 0 or (isinstance(side, str) and side.upper() == "HOLD"):
            return None
        stop = value.get("stop", value.get("stop_loss"))
        target = value.get("target", value.get("take_profit"))
        return Setup(side, stop, target)
    if isinstance(value, tuple) and len(value) == 3:
        return Setup(*value)
    raise TypeError(
        "decide() must return None, a Setup, a (side, stop, target) tuple or a mapping; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class Trade:
    """One completed trade.

    Prices are actual fills, so ``entry_price`` already includes the spread
    and slippage paid on entry and ``exit_price`` the spread and slippage paid
    on exit. ``r`` is the net result in units of initial risk, after every
    cost.

    Attributes
    ----------
    side: ``+1`` long, ``-1`` short.
    signal_index: bar whose close the decision was made on.
    entry_index: bar whose open the order filled at.
    exit_index: bar on which the trade closed.
    mid_entry: the entry bar's open, before spread and slippage.
    risk: ``side * (entry_price - stop)``, always positive.
    reward: ``side * (target - entry_price)``, always positive.
    fees: commission paid on both sides, in price units.
    ambiguous: the exit bar touched both the stop and the target, so OHLC
        data cannot say which came first. The engine books the stop.
    """

    side: int
    signal_index: int
    entry_index: int
    exit_index: int
    mid_entry: float
    entry_price: float
    exit_price: float
    stop: float
    target: float
    reason: ExitReason
    r: float
    risk: float
    reward: float
    fees: float
    ambiguous: bool = False

    @property
    def bars_held(self) -> int:
        """Number of bars the position was open, counting entry and exit bars."""
        return self.exit_index - self.entry_index + 1

    @property
    def is_win(self) -> bool:
        return self.r > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "signal_index": self.signal_index,
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "mid_entry": self.mid_entry,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop": self.stop,
            "target": self.target,
            "reason": self.reason,
            "r": self.r,
            "risk": self.risk,
            "reward": self.reward,
            "fees": self.fees,
            "ambiguous": self.ambiguous,
        }
