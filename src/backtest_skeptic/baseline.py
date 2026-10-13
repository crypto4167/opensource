"""No-skill baselines for bracket (stop/target) trades.

A 90% win rate means nothing on its own. A trade whose target sits one ninth
as far from the entry as its stop wins about 90% of the time with no skill at
all, and after costs it loses money. This module computes, trade by trade,
the win rate a skill-less trader would get with the same brackets, the win
rate needed to break even after costs, and an exact test of the observed wins
against the first of these.

The baseline
------------
Bars are mid prices. For a trade on side ``s`` (``+1`` long, ``-1`` short),
let ``m`` be the mid price it filled at, which is the fill bar's open before
costs (:attr:`Trade.mid_entry`), and let

* ``a = s * (m - stop)``, the distance from the fill to the stop, and
* ``b = s * (target - m)``, the distance from the fill to the target.

If the mid price is a driftless continuous martingale from the fill on, it
reaches the target before the stop with probability ``a / (a + b)``. This is
the fair-game ruin probability (Feller, vol. 1, ch. XIV), and for a
continuous martingale it follows from optional stopping: the stopped price
has mean ``m`` and can only end at the stop or the target.

Costs move the break-even point, not the no-skill rate. With half-spread
``h``, slippage ``sl`` and fee rate ``f`` (a :class:`~backtest_skeptic.Costs`),
entry fill ``e`` (:attr:`Trade.entry_price`) and risk ``R``
(:attr:`Trade.risk`), the engine books a target hit and a stop hit (without a
gap) as::

    r_win  = (s * (target - s*h - e)      - f * (|e| + |target - s*h|))      / R
    r_loss = (s * (stop - s*(h+sl) - e)   - f * (|e| + |stop - s*(h+sl)|))   / R

A trader who wins with probability ``p`` breaks even at
``p* = -r_loss / (r_win - r_loss)``. Without costs ``p* = a / (a + b)``; with
costs ``p*`` is higher. So there are two thresholds, and they answer
different questions:

* ``win_rate > no_skill_rate`` means the entries predict direction better
  than chance (skill), and
* ``win_rate > breakeven_rate`` means the trades make money after costs.

Every trade's expectancy splits exactly into the two:
``r = (w - p*) * (r_win - r_loss)`` with ``w`` = 1 for a target hit, and
``w - p* = (w - a/(a+b)) - (p* - a/(a+b))``. See
:func:`expectancy_decomposition`.

Three mistakes this module avoids
---------------------------------
1. **Anchoring at the signal bar's close.** The engine brackets from the fill
   bar's open, so the baseline must too. Anchoring at the signal close is not
   just noisier: whenever trades are selected on the fill-bar open (a
   ``risk_band`` filter, or the engine's own rule that skips a fill already
   beyond the stop or the target, which removes more trades on the side of
   the nearer level), the surviving trades have systematically different
   odds from fill, and a signal-close baseline turns that selection into fake
   skill. Selection on information available at the fill leaves the
   fill-anchored probability valid.
2. **Calling a no-skill trader breakeven.** At ``p = a / (a + b)`` a trader
   loses about one round-trip cost per trade (exactly
   ``-(2h + sl*(a + 2b)/(a + b)) / R`` without fees). ``no_skill_rate`` and
   ``breakeven_rate`` are reported separately, with ``cost_drag`` between
   them and ``no_skill_expectancy_r`` for the R a skill-less trader expects.
3. **Scoring time and end-of-data exits against the closed form.** A trade
   closed by ``max_bars`` or by the end of the data reached neither level.
   About half of them close in profit, so counting profitable ones as wins
   against ``a / (a + b)`` makes zero-skill entries with wide targets look
   skilled. The win/loss test uses only trades that exited at the stop or
   the target, and says how many it left out.

Leaving those trades out is itself a selection: a trade that resolves
quickly is more likely to have hit the nearer level. When many trades time
out, the resolved-only test is biased toward the nearer level (tight targets
look skilled, wide targets look unskilled). Two things guard against this:

* ``exclusion_bias`` estimates the shift from the excluded trades
  themselves. Under no skill, a trade that timed out with its mid price a
  fraction ``u`` of the way from the stop to the target would have gone on to
  hit the target with probability ``u``. Under no skill the resolved trades'
  excess wins are therefore expected to equal ``sum(a/(a+b) - u)`` over the
  excluded trades; divided by ``n_resolved``, that is ``exclusion_bias``.
* ``all_exits_edge`` scores every trade: 1 at the target, 0 at the stop and
  ``u`` for a time or end exit. By optional stopping its no-skill mean is
  ``a / (a + b)`` for every exit type, so it has no exclusion bias. Its test
  uses the bound ``Var(u) <= p(1 - p)`` for a score in ``[0, 1]`` with mean
  ``p``, which is conservative, and a normal approximation.

For a baseline that treats every exit the same way under the real price
process, use matched random entries
(:func:`backtest_skeptic.random_entry.random_entry_test`).

Limitations
-----------
* The baseline assumes a driftless mid price. A trend gives a long-only
  strategy a win rate above ``a / (a + b)`` with no timing skill (that is
  beta, not alpha); compare against random entries on the same data.
* A gap through the stop fills at the bar's open and loses more than
  ``r_loss``; such trades are counted (``gap_fills``) and they make the true
  break-even rate higher than reported. A gap through the target is filled
  at the target, which slightly favours the stop.
* A bar that opens between the levels and touches both is booked as a stop
  by default (``same_bar="stop"``), which biases the win rate down. Wins
  booked on such bars (``same_bar="target"``) are flagged, since OHLC data
  cannot tell which level came first.
* The exact p-value treats trade outcomes as independent Bernoulli trials
  with the computed probabilities. Non-overlapping trades on a random walk
  satisfy this when bracket sizes are fixed in advance; brackets that adapt
  to earlier outcomes, and volatility clustering in real data, make it an
  approximation.
* ``costs`` must be the costs the backtest used: the break-even rates depend
  on them. Passing a :class:`~backtest_skeptic.engine.BacktestResult` uses its
  costs; trades whose fills contradict the costs given raise ``ValueError``.

References
----------
* Feller, W. *An Introduction to Probability Theory and Its Applications*,
  vol. 1, ch. XIV (random walks and ruin problems).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .engine import BacktestResult
from .trades import Costs, Trade

__all__ = [
    "BaselineResult",
    "BracketOdds",
    "ExpectancyDecomposition",
    "bracket_odds",
    "breakeven_win_rate",
    "expectancy_decomposition",
    "no_skill_test",
    "poisson_binomial_pmf",
]

Method = Literal["exact", "normal"]

_NORMAL = NormalDist()
_PRICE_RTOL = 1e-9
# Heuristic thresholds for warnings; neither has a canonical source.
_UNRESOLVED_WARN_SHARE = 0.10
_BIAS_WARN_SE = 0.5
_REASONS = ("stop", "target", "time", "end")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _pts(x: float) -> str:
    return f"{100.0 * x:+.1f} pts"


def _pval(p: float) -> str:
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def _check_tolerance(tolerance: float) -> float:
    tol = _finite_float(tolerance, "tolerance")
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0, got {tol}")
    return tol


def poisson_binomial_pmf(probs: ArrayLike) -> NDArray[np.float64]:
    """Distribution of the number of successes in independent Bernoulli trials.

    ``probs[i]`` is the success probability of trial ``i``. Returns an array
    of length ``len(probs) + 1`` whose entry ``k`` is ``P(successes == k)``,
    computed exactly (up to rounding) by the dynamic program that adds one
    trial at a time: ``P_new(k) = P(k) (1 - p) + P(k - 1) p``. Every term is
    non-negative, so tail sums do not suffer from cancellation. Cost is
    ``O(n^2)``: about a quarter of a second for 20,000 trials.

    With equal probabilities this is the binomial distribution.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"probs must be one-dimensional, got shape {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probs must be finite and between 0 and 1")
    n = int(p.size)
    pmf = np.zeros(n + 1, dtype=np.float64)
    pmf[0] = 1.0
    for k in range(n):
        pk = float(p[k])
        qk = 1.0 - pk
        head = pmf[: k + 1] * pk
        seg = pmf[1 : k + 2]
        seg *= qk
        seg += head
        pmf[0] *= qk
    return pmf


# ---------------------------------------------------------------------------
# per-trade arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Book:
    """Per-trade arrays shared by the public functions."""

    n: int
    side: NDArray[np.float64]
    stop: NDArray[np.float64]
    target: NDArray[np.float64]
    r: NDArray[np.float64]
    is_stop: NDArray[np.bool_]
    is_target: NDArray[np.bool_]
    is_time: NDArray[np.bool_]
    is_end: NDArray[np.bool_]
    ambiguous: NDArray[np.bool_]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    valid: NDArray[np.bool_]
    p0: NDArray[np.float64]
    r_win: NDArray[np.float64]
    r_loss: NDArray[np.float64]
    pstar: NDArray[np.float64]
    gap: NDArray[np.bool_]
    u: NDArray[np.float64]


def _collect(
    trades: BacktestResult | Iterable[Trade], costs: Costs | None
) -> tuple[list[Trade], Costs]:
    if costs is not None and not isinstance(costs, Costs):
        raise TypeError(f"costs must be a Costs instance or None, got {type(costs).__name__}")
    if isinstance(trades, BacktestResult):
        if costs is not None and costs != trades.costs:
            raise ValueError(
                f"costs={costs!r} differ from the costs the backtest used ({trades.costs!r}); "
                "break-even rates are only meaningful with the backtest's own costs, so "
                "omit costs when passing a BacktestResult"
            )
        return list(trades.trades), trades.costs
    if isinstance(trades, Trade):
        raise TypeError("pass a sequence of trades or a BacktestResult, not a single Trade")
    items = list(trades)
    for i, t in enumerate(items):
        if not isinstance(t, Trade):
            raise TypeError(f"item {i} is a {type(t).__name__}, not a Trade")
    return items, (costs if costs is not None else Costs())


def _fail(i: int, what: str) -> ValueError:
    return ValueError(
        f"trade {i}: {what}. The trades do not match the costs given; pass the costs the "
        "backtest used (or the BacktestResult itself, which carries them)."
    )


def _build(trades: list[Trade], costs: Costs, tol: float) -> _Book:
    """Validate trades against ``costs`` and compute every per-trade quantity."""
    n = len(trades)
    if n == 0:
        raise ValueError("no trades to test")
    for i, t in enumerate(trades):
        if t.reason not in _REASONS:
            raise ValueError(f"trade {i}: unknown exit reason {t.reason!r}")
        if t.side not in (1, -1):
            raise ValueError(f"trade {i}: side must be +1 or -1, got {t.side!r}")

    def col(name: str) -> NDArray[np.float64]:
        return np.array([getattr(t, name) for t in trades], dtype=np.float64)

    side = col("side")
    mid = col("mid_entry")
    entry = col("entry_price")
    exit_ = col("exit_price")
    stop = col("stop")
    target = col("target")
    risk = col("risk")
    r = col("r")
    fees = col("fees")
    for name, arr in (
        ("mid_entry", mid),
        ("entry_price", entry),
        ("exit_price", exit_),
        ("stop", stop),
        ("target", target),
        ("risk", risk),
        ("r", r),
        ("fees", fees),
    ):
        bad = ~np.isfinite(arr)
        if bad.any():
            raise ValueError(f"trade {int(np.argmax(bad))}: {name} is not finite")
    if (risk <= 0).any():
        raise ValueError(f"trade {int(np.argmax(risk <= 0))}: risk must be positive")

    reasons = [t.reason for t in trades]
    is_stop = np.array([x == "stop" for x in reasons], dtype=np.bool_)
    is_target = np.array([x == "target" for x in reasons], dtype=np.bool_)
    is_time = np.array([x == "time" for x in reasons], dtype=np.bool_)
    is_end = np.array([x == "end" for x in reasons], dtype=np.bool_)
    ambiguous = np.array([bool(t.ambiguous) for t in trades], dtype=np.bool_)

    h, sl, f = costs.half_spread, costs.slippage, costs.fee_rate

    # Consistency of each trade with itself and with the costs.
    bad = np.abs(risk - side * (entry - stop)) > _PRICE_RTOL * (np.abs(entry) + np.abs(stop))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"trade {i}: risk {risk[i]!r} is not side * (entry_price - stop)")
    implied = side * (entry - mid)
    bad = np.abs(implied - (h + sl)) > _PRICE_RTOL * (np.abs(entry) + np.abs(mid))
    if bad.any():
        i = int(np.argmax(bad))
        raise _fail(
            i,
            f"entry_price - mid_entry implies an entry cost of {implied[i]:.10g}, but "
            f"half_spread + slippage = {h + sl:.10g}",
        )
    notional = np.abs(entry) + np.abs(exit_)
    bad = np.abs(fees - f * notional) > _PRICE_RTOL * notional
    if bad.any():
        i = int(np.argmax(bad))
        raise _fail(i, f"fees {fees[i]:.10g} are not fee_rate * notional = {f * notional[i]:.10g}")
    bad = np.abs(r - (side * (exit_ - entry) - fees) / risk) > tol
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"trade {i}: r {r[i]!r} does not match its prices, fees and risk")
    win_exit = target - side * h
    bad = is_target & (np.abs(exit_ - win_exit) > _PRICE_RTOL * (np.abs(exit_) + np.abs(target)))
    if bad.any():
        i = int(np.argmax(bad))
        raise _fail(
            i,
            f"a target exit filled at {exit_[i]:.10g}, but target - side * half_spread = "
            f"{win_exit[i]:.10g}",
        )

    # The closed form.
    loss_exit = stop - side * (h + sl)
    r_win = (side * (win_exit - entry) - f * (np.abs(entry) + np.abs(win_exit))) / risk
    r_loss = (side * (loss_exit - entry) - f * (np.abs(entry) + np.abs(loss_exit))) / risk
    better = is_stop & (r > r_loss + tol)
    if better.any():
        i = int(np.argmax(better))
        raise _fail(i, f"a stop exit booked r = {r[i]:.6g}, better than the stop allows")
    gap = is_stop & (r < r_loss - tol)

    a = side * (mid - stop)
    b = side * (target - mid)
    spread = r_win - r_loss
    valid = (a > 0) & (b > 0) & (spread > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = np.where(valid, a / (a + b), np.nan)
        pstar = np.where(valid, -r_loss / spread, np.nan)
        # Fraction of the way from the stop to the target at which the mid
        # price stood when the trade closed (1 at a target exit).
        exit_mid = np.where(is_target, target, exit_ + side * (h + sl))
        u = np.where(valid, (exit_mid - stop) / (target - stop), np.nan)
    return _Book(
        n=n,
        side=side,
        stop=stop,
        target=target,
        r=r,
        is_stop=is_stop,
        is_target=is_target,
        is_time=is_time,
        is_end=is_end,
        ambiguous=ambiguous,
        a=a,
        b=b,
        valid=valid,
        p0=p0,
        r_win=r_win,
        r_loss=r_loss,
        pstar=pstar,
        gap=gap,
        u=u,
    )


def _exclusions(book: _Book) -> dict[str, int]:
    return {
        "time": int((book.valid & book.is_time).sum()),
        "end": int((book.valid & book.is_end).sum()),
        "invalid_geometry": int((~book.valid).sum()),
    }


def _no_resolved_error(excluded: dict[str, int]) -> ValueError:
    detail = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in excluded.items())
    return ValueError(
        "no trade exited at its stop or target with valid geometry "
        f"(excluded: {detail}); the closed-form baseline needs resolved trades. "
        "Use backtest_skeptic.random_entry.random_entry_test for a baseline that scores "
        "time exits."
    )


# ---------------------------------------------------------------------------
# public: per-trade odds and the simple break-even helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketOdds:
    """No-skill odds and payoffs of one bracket trade.

    ``stop_distance`` and ``target_distance`` are ``a`` and ``b`` in price
    units, measured from the fill bar's open (mid price). ``r_win`` and
    ``r_loss`` are the trade's net R if it hits the target or the stop
    (without a gap). ``no_skill_prob`` is ``a / (a + b)`` and
    ``breakeven_prob`` is ``-r_loss / (r_win - r_loss)``; a break-even
    probability of 1 or more means costs eat the whole reward.
    """

    side: int
    stop_distance: float
    target_distance: float
    no_skill_prob: float
    r_win: float
    r_loss: float
    breakeven_prob: float

    @property
    def no_skill_expectancy_r(self) -> float:
        """Expected net R of the trade at the no-skill win probability."""
        p = self.no_skill_prob
        return p * self.r_win + (1.0 - p) * self.r_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "stop_distance": self.stop_distance,
            "target_distance": self.target_distance,
            "no_skill_prob": self.no_skill_prob,
            "r_win": self.r_win,
            "r_loss": self.r_loss,
            "breakeven_prob": self.breakeven_prob,
            "no_skill_expectancy_r": self.no_skill_expectancy_r,
        }

    def __str__(self) -> str:
        return (
            f"{'long' if self.side > 0 else 'short'} bracket: stop {self.stop_distance:.6g} / "
            f"target {self.target_distance:.6g} from fill; no-skill win rate "
            f"{_pct(self.no_skill_prob)}, break-even {_pct(self.breakeven_prob)}; "
            f"win {self.r_win:+.3f}R, loss {self.r_loss:+.3f}R; no-skill expectancy "
            f"{self.no_skill_expectancy_r:+.4f}R"
        )


def bracket_odds(trade: Trade, *, costs: Costs | None = None) -> BracketOdds:
    """No-skill win probability, payoffs and break-even rate of one trade.

    ``costs`` must be the costs the trade was filled with (default
    ``Costs()``, no costs); a trade whose fills contradict them raises
    ``ValueError``. Raises ``ValueError`` when the fill-bar open is already
    at or beyond the stop, where no no-skill probability is defined.
    """
    if not isinstance(trade, Trade):
        raise TypeError(f"trade must be a Trade, got {type(trade).__name__}")
    items, used = _collect([trade], costs)
    book = _build(items, used, 1e-9)
    if not book.valid[0]:
        raise ValueError(
            "the fill-bar open is at or beyond the stop or the target "
            f"(a = {book.a[0]:.6g}, b = {book.b[0]:.6g}); the no-skill probability is undefined"
        )
    return BracketOdds(
        side=int(book.side[0]),
        stop_distance=float(book.a[0]),
        target_distance=float(book.b[0]),
        no_skill_prob=float(book.p0[0]),
        r_win=float(book.r_win[0]),
        r_loss=float(book.r_loss[0]),
        breakeven_prob=float(book.pstar[0]),
    )


def breakeven_win_rate(risk: float, reward: float, *, cost_r: float = 0.0) -> float:
    """Win rate at which a bracket breaks even, for planning.

    ``risk`` and ``reward`` are the stop and target distances in the same
    price units (both positive). ``cost_r`` is the round-trip cost in units
    of ``risk``, charged on wins and losses alike, so a win pays
    ``reward / risk - cost_r`` and a loss ``-1 - cost_r``. The break-even
    win rate is then ``(1 + cost_r) * risk / (risk + reward)``; with zero
    costs it equals the no-skill rate ``risk / (risk + reward)``.

    A result of 1 or more means the costs eat the whole reward and no win
    rate breaks even. For trades from a backtest, :func:`no_skill_test`
    computes the exact per-trade figure from the fills (where target exits
    pay no slippage and fees depend on the exit price).
    """
    risk_f = _finite_float(risk, "risk")
    reward_f = _finite_float(reward, "reward")
    cost_f = _finite_float(cost_r, "cost_r")
    if risk_f <= 0 or reward_f <= 0:
        raise ValueError(f"risk and reward must be positive, got {risk_f} and {reward_f}")
    if cost_f < 0:
        raise ValueError(f"cost_r must be >= 0, got {cost_f}")
    return (1.0 + cost_f) * risk_f / (risk_f + reward_f)


# ---------------------------------------------------------------------------
# public: the no-skill test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Observed target hits against the no-skill and break-even win rates.

    Rates are fractions (0.55 is 55%) and edges are differences of fractions.
    R figures are net of costs, per trade.

    Attributes
    ----------
    n_trades: trades passed in.
    n_resolved: trades that exited at the stop or the target with valid
        geometry (``a > 0`` and ``b > 0``); the win/loss test uses only these.
    excluded: counts of trades left out, by reason: ``time`` (``max_bars``),
        ``end`` (end of data) and ``invalid_geometry`` (filled with the mid
        price at or beyond the stop).
    wins: resolved trades that hit the target (a target hit is a win even if
        costs made it a net loss).
    win_rate: ``wins / n_resolved``.
    no_skill_rate: mean of ``a / (a + b)``.
    breakeven_rate: mean of ``p*``. Exact for identical brackets; for mixed
        brackets :func:`expectancy_decomposition` does the exact accounting.
    skill_edge: ``win_rate - no_skill_rate``.
    cost_drag: ``breakeven_rate - no_skill_rate``.
    margin: ``win_rate - breakeven_rate``.
    expected_wins: ``sum(a / (a + b))``, the no-skill expected number of wins.
    z: ``(wins - expected_wins) / sqrt(sum p (1 - p))``.
    p_value: one-sided ``P(W >= wins)`` under no skill, ``W`` Poisson-binomial.
    p_value_lower: ``P(W <= wins)`` (small when the entries are worse than
        random, which usually means a bug or a strongly adverse filter).
    method: ``"exact"`` (dynamic programming) or ``"normal"`` (normal
        approximation with continuity correction, above ``exact_limit``).
    no_skill_expectancy_r: mean R a skill-less trader expects with these
        brackets and costs (negative with any costs).
    mean_r: mean R the resolved trades actually earned.
    gap_fills: resolved stop exits that filled beyond the stop.
    ambiguous: resolved trades whose exit bar touched both levels.
    ambiguous_wins: of those, the ones booked as target hits.
    unresolved_share: ``(time + end) / n_trades``.
    exclusion_bias: estimated shift in ``skill_edge`` caused by leaving out
        time and end exits, if the entries had no skill.
    all_exits_n, all_exits_edge, all_exits_z, all_exits_p_value: the
        exclusion-free score over every valid trade (1 at the target, 0 at
        the stop, the stop-to-target fraction ``u`` otherwise); conservative
        variance, normal approximation, one-sided.
    costs: the costs used for ``r_win``, ``r_loss`` and ``p*``.
    warnings: plain-language problems found.
    """

    n_trades: int
    n_resolved: int
    excluded: dict[str, int]
    wins: int
    win_rate: float
    no_skill_rate: float
    breakeven_rate: float
    skill_edge: float
    cost_drag: float
    margin: float
    expected_wins: float
    z: float
    p_value: float
    p_value_lower: float
    method: Method
    no_skill_expectancy_r: float
    mean_r: float
    gap_fills: int
    ambiguous: int
    ambiguous_wins: int
    unresolved_share: float
    exclusion_bias: float
    all_exits_n: int
    all_exits_edge: float
    all_exits_z: float
    all_exits_p_value: float
    costs: Costs
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "n_resolved": self.n_resolved,
            "excluded": dict(self.excluded),
            "wins": self.wins,
            "win_rate": self.win_rate,
            "no_skill_rate": self.no_skill_rate,
            "breakeven_rate": self.breakeven_rate,
            "skill_edge": self.skill_edge,
            "cost_drag": self.cost_drag,
            "margin": self.margin,
            "expected_wins": self.expected_wins,
            "z": self.z,
            "p_value": self.p_value,
            "p_value_lower": self.p_value_lower,
            "method": self.method,
            "no_skill_expectancy_r": self.no_skill_expectancy_r,
            "mean_r": self.mean_r,
            "gap_fills": self.gap_fills,
            "ambiguous": self.ambiguous,
            "ambiguous_wins": self.ambiguous_wins,
            "unresolved_share": self.unresolved_share,
            "exclusion_bias": self.exclusion_bias,
            "all_exits_n": self.all_exits_n,
            "all_exits_edge": self.all_exits_edge,
            "all_exits_z": self.all_exits_z,
            "all_exits_p_value": self.all_exits_p_value,
            "costs": {
                "half_spread": self.costs.half_spread,
                "fee_rate": self.costs.fee_rate,
                "slippage": self.costs.slippage,
            },
            "warnings": list(self.warnings),
        }

    def __str__(self) -> str:
        ex = self.excluded
        lines = [
            f"No-skill baseline over {self.n_resolved} resolved trades "
            f"(excluded: {ex['time']} time, {ex['end']} end, "
            f"{ex['invalid_geometry']} invalid geometry)",
            f"  win rate {_pct(self.win_rate)} vs no-skill {_pct(self.no_skill_rate)}: "
            f"skill edge {_pts(self.skill_edge)} (z {self.z:+.2f}, one-sided p "
            f"{_pval(self.p_value)}, {self.method})",
            f"  break-even win rate {_pct(self.breakeven_rate)}: cost drag "
            f"{_pts(self.cost_drag)}, margin over break-even {_pts(self.margin)}",
            f"  a no-skill trader with these brackets expects {self.no_skill_expectancy_r:+.4f}R "
            f"per trade; these trades averaged {self.mean_r:+.4f}R",
        ]
        if self.all_exits_n > self.n_resolved:
            lines.append(
                f"  all exits ({self.all_exits_n} trades, time/end scored by position): "
                f"edge {_pts(self.all_exits_edge)} (z {self.all_exits_z:+.2f}, p "
                f"{_pval(self.all_exits_p_value)}, normal approx.)"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def no_skill_test(
    trades: BacktestResult | Iterable[Trade],
    *,
    costs: Costs | None = None,
    tolerance: float = 1e-9,
    exact_limit: int = 20_000,
) -> BaselineResult:
    """Test whether bracket trades hit their targets more often than chance.

    What it catches: a high win rate that comes from bracket geometry (a
    target much closer than the stop) instead of from the entries, and a
    win rate that beats chance but not costs.

    Parameters
    ----------
    trades:
        A :class:`~backtest_skeptic.engine.BacktestResult` (its costs are
        used) or any iterable of :class:`~backtest_skeptic.Trade`.
    costs:
        The costs the trades were filled with. Defaults to the
        BacktestResult's costs, or ``Costs()`` (none) for a plain list.
        Different costs from the ones the backtest used would give wrong
        break-even rates, so trades whose entry, target-exit, stop-exit or
        fee amounts contradict ``costs`` raise ``ValueError``.
    tolerance:
        Slack, in R, for matching a trade's ``r`` to its prices and for
        detecting a stop that filled beyond the stop price (a gap).
    exact_limit:
        Largest number of resolved trades for the exact Poisson-binomial
        p-value; above it a normal approximation is used and flagged.

    Only trades that exited at the stop or the target are tested against
    ``a / (a + b)``: time and end-of-data exits reached neither level and are
    reported in ``excluded``, with a warning when they are more than 10% of
    the trades (a heuristic threshold) or when their exclusion is estimated
    to move ``skill_edge`` by more than half a standard error. The
    ``all_exits_*`` fields give an exclusion-free score; see the module
    docstring for the derivation and limitations.

    Raises ``ValueError`` when no trade resolved at a level.
    """
    tol = _check_tolerance(tolerance)
    if isinstance(exact_limit, bool) or not isinstance(exact_limit, (int, np.integer)):
        raise ValueError(f"exact_limit must be an integer, got {exact_limit!r}")
    if exact_limit < 0:
        raise ValueError(f"exact_limit must be >= 0, got {exact_limit}")
    items, used_costs = _collect(trades, costs)
    book = _build(items, used_costs, tol)

    excluded = _exclusions(book)
    resolved = book.valid & (book.is_stop | book.is_target)
    n_res = int(resolved.sum())
    if n_res == 0:
        raise _no_resolved_error(excluded)

    p = book.p0[resolved]
    w = book.is_target[resolved]
    wins = int(w.sum())
    mu = float(p.sum())
    var = float((p * (1.0 - p)).sum())
    sd = math.sqrt(var)
    win_rate = wins / n_res
    no_skill_rate = mu / n_res
    breakeven_rate = float(book.pstar[resolved].mean())
    z = (wins - mu) / sd if sd > 0 else 0.0

    method: Method
    if n_res <= exact_limit or sd == 0:
        # sd == 0 only if every probability rounded to exactly 0 or 1, where
        # the normal approximation is undefined and the count is deterministic.
        method = "exact"
        pmf = poisson_binomial_pmf(p)
        p_upper = min(1.0, float(pmf[wins:].sum()))
        p_lower = min(1.0, float(pmf[: wins + 1].sum()))
    else:
        method = "normal"
        p_upper = _NORMAL.cdf(-(wins - 0.5 - mu) / sd)
        p_lower = _NORMAL.cdf((wins + 0.5 - mu) / sd)

    r_win = book.r_win[resolved]
    r_loss = book.r_loss[resolved]
    no_skill_exp = float((p * r_win + (1.0 - p) * r_loss).mean())
    mean_r = float(book.r[resolved].mean())
    gap_fills = int((book.gap & resolved).sum())
    ambiguous = int((book.ambiguous & resolved).sum())
    ambiguous_wins = int((book.ambiguous & book.is_target & resolved).sum())
    n_unresolved = excluded["time"] + excluded["end"]
    unresolved_share = n_unresolved / book.n

    unresolved = book.valid & ~resolved
    exclusion_bias = float((book.p0[unresolved] - book.u[unresolved]).sum()) / n_res
    p_all = book.p0[book.valid]
    score = np.where(book.is_target, 1.0, np.where(book.is_stop, 0.0, book.u))[book.valid]
    all_n = int(p_all.size)
    all_edge = float((score - p_all).mean())
    sd_all = math.sqrt(float((p_all * (1.0 - p_all)).sum()))
    all_z = float((score - p_all).sum()) / sd_all if sd_all > 0 else 0.0
    all_p = _NORMAL.cdf(-all_z)

    warnings: list[str] = []
    se_edge = sd / n_res
    if unresolved_share > _UNRESOLVED_WARN_SHARE:
        warnings.append(
            f"{_pct(unresolved_share)} of trades ({n_unresolved}) exited on time or at the end "
            "of the data and are excluded. Excluding them biases skill_edge toward the nearer "
            f"level (estimated {_pts(exclusion_bias)} under no skill); see all_exits_edge, "
            "which scores every exit, or use random_entry.random_entry_test."
        )
    elif n_unresolved and se_edge > 0 and abs(exclusion_bias) > _BIAS_WARN_SE * se_edge:
        warnings.append(
            f"excluding {n_unresolved} time/end exits shifts skill_edge by an estimated "
            f"{_pts(exclusion_bias)} under no skill ({abs(exclusion_bias) / se_edge:.1f} "
            "standard errors); compare all_exits_edge, or use random_entry.random_entry_test."
        )
    if excluded["invalid_geometry"]:
        warnings.append(
            f"{excluded['invalid_geometry']} trades filled with the mid price at or beyond the "
            "stop and were excluded."
        )
    if gap_fills:
        warnings.append(
            f"{gap_fills} stop losses filled beyond the stop (gaps) and lost more than r_loss; "
            "breakeven_rate understates the win rate these trades need."
        )
    if ambiguous_wins:
        warnings.append(
            f"{ambiguous_wins} wins came from bars that touched both the stop and the target; "
            "OHLC bars cannot show which came first, so these wins may not be real "
            "(rerun with same_bar='stop')."
        )
    hopeless = int((book.pstar[resolved] >= 1.0).sum())
    if hopeless:
        warnings.append(
            f"{hopeless} trades had costs at least as large as their target distance: no win "
            "rate breaks even on them."
        )
    if method == "normal":
        warnings.append(
            f"{n_res} resolved trades exceed exact_limit={exact_limit}; p-values use a normal "
            "approximation with continuity correction."
        )

    return BaselineResult(
        n_trades=book.n,
        n_resolved=n_res,
        excluded=excluded,
        wins=wins,
        win_rate=win_rate,
        no_skill_rate=no_skill_rate,
        breakeven_rate=breakeven_rate,
        skill_edge=win_rate - no_skill_rate,
        cost_drag=breakeven_rate - no_skill_rate,
        margin=win_rate - breakeven_rate,
        expected_wins=mu,
        z=z,
        p_value=p_upper,
        p_value_lower=p_lower,
        method=method,
        no_skill_expectancy_r=no_skill_exp,
        mean_r=mean_r,
        gap_fills=gap_fills,
        ambiguous=ambiguous,
        ambiguous_wins=ambiguous_wins,
        unresolved_share=unresolved_share,
        exclusion_bias=exclusion_bias,
        all_exits_n=all_n,
        all_exits_edge=all_edge,
        all_exits_z=all_z,
        all_exits_p_value=all_p,
        costs=used_costs,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# public: where the expectancy comes from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectancyDecomposition:
    """Mean R split into a skill term and a cost term.

    For a trade that hit its target (``w = 1``) or its stop without a gap
    (``w = 0``), ``r = (w - p*) * (r_win - r_loss)`` exactly. Writing
    ``w - p* = (w - p0) - (p* - p0)`` with ``p0 = a / (a + b)`` gives

    * ``skill_r = mean((w - p0) * (r_win - r_loss))``: R earned by hitting
      targets more often than chance, each hit valued at its payoff spread;
      zero on average without skill.
    * ``cost_r = mean((p0 - p*) * (r_win - r_loss))``: what a skill-less
      trader with the same brackets expects, i.e. the cost of trading
      (negative whenever there are costs).

    ``mean_r = skill_r + cost_r + residual_r``, where ``residual_r`` is
    rounding error. ``excluded`` counts trades left out: ``time`` and
    ``end`` exits (no closed-form payoff), ``invalid_geometry`` and
    ``gap_fill`` (stop exits that filled beyond the stop).
    """

    n_trades: int
    n_used: int
    excluded: dict[str, int]
    mean_r: float
    skill_r: float
    cost_r: float
    residual_r: float
    win_rate: float
    no_skill_rate: float
    breakeven_rate: float
    costs: Costs
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "n_used": self.n_used,
            "excluded": dict(self.excluded),
            "mean_r": self.mean_r,
            "skill_r": self.skill_r,
            "cost_r": self.cost_r,
            "residual_r": self.residual_r,
            "win_rate": self.win_rate,
            "no_skill_rate": self.no_skill_rate,
            "breakeven_rate": self.breakeven_rate,
            "costs": {
                "half_spread": self.costs.half_spread,
                "fee_rate": self.costs.fee_rate,
                "slippage": self.costs.slippage,
            },
            "warnings": list(self.warnings),
        }

    def __str__(self) -> str:
        ex = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in self.excluded.items())
        lines = [
            f"Expectancy over {self.n_used} resolved trades (excluded: {ex}): "
            f"{self.mean_r:+.4f}R = skill {self.skill_r:+.4f}R + costs {self.cost_r:+.4f}R "
            f"(residual {self.residual_r:+.1e})",
            f"  win rate {_pct(self.win_rate)}, no-skill {_pct(self.no_skill_rate)}, "
            f"break-even {_pct(self.breakeven_rate)}",
        ]
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def expectancy_decomposition(
    trades: BacktestResult | Iterable[Trade],
    *,
    costs: Costs | None = None,
    tolerance: float = 1e-9,
) -> ExpectancyDecomposition:
    """Show how much of the mean R comes from skill and how much costs take.

    Uses the trades that exited at the stop (without a gap) or the target,
    where the payoff is known in closed form; ``costs`` and ``tolerance``
    work as in :func:`no_skill_test`. A strategy with positive ``mean_r`` but
    ``skill_r`` near zero is not making money from its entries (check for a
    cost model that is too kind); one with positive ``skill_r`` and negative
    ``mean_r`` predicts direction but not well enough to pay for trading.

    Raises ``ValueError`` when no trade qualifies.
    """
    tol = _check_tolerance(tolerance)
    items, used_costs = _collect(trades, costs)
    book = _build(items, used_costs, tol)
    excluded = _exclusions(book)
    resolved = book.valid & (book.is_stop | book.is_target)
    gap = book.gap & resolved
    excluded["gap_fill"] = int(gap.sum())
    used = resolved & ~gap
    n_used = int(used.sum())
    if n_used == 0:
        if int(resolved.sum()) == 0:
            raise _no_resolved_error(_exclusions(book))
        raise ValueError("every resolved trade was a gap fill; nothing to decompose")

    w = book.is_target[used].astype(np.float64)
    p0 = book.p0[used]
    pstar = book.pstar[used]
    spread = book.r_win[used] - book.r_loss[used]
    skill_r = float(((w - p0) * spread).mean())
    cost_r = float(((p0 - pstar) * spread).mean())
    mean_r = float(book.r[used].mean())

    warnings: list[str] = []
    n_unresolved = excluded["time"] + excluded["end"]
    if n_unresolved / book.n > _UNRESOLVED_WARN_SHARE:
        warnings.append(
            f"{_pct(n_unresolved / book.n)} of trades exited on time or at the end of the data "
            "and are not in the decomposition; it describes only trades that reached a level."
        )
    if excluded["gap_fill"]:
        warnings.append(
            f"{excluded['gap_fill']} gap-filled stop losses are left out; they cost more than "
            "the closed-form loss."
        )
    return ExpectancyDecomposition(
        n_trades=book.n,
        n_used=n_used,
        excluded=excluded,
        mean_r=mean_r,
        skill_r=skill_r,
        cost_r=cost_r,
        residual_r=mean_r - (skill_r + cost_r),
        win_rate=float(w.mean()),
        no_skill_rate=float(p0.mean()),
        breakeven_rate=float(pstar.mean()),
        costs=used_costs,
        warnings=tuple(warnings),
    )
