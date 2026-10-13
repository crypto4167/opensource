"""Matched random-entry null tests ("monkey tests").

The question these tests answer: **could random entries with the same number
of trades, the same direction mix, the same bracket geometry, the same costs
and the same exit rules have produced this result?** A bracket strategy's
result mixes several ingredients: the market's drift over the test period,
the payoff shape of the bracket (a far target and a near stop win rarely but
big), costs, the long/short mix and, last, the entry signal itself. Comparing
the strategy with zero, or with a buy-and-hold benchmark, credits the signal
with all of them. Comparing it with random entries that copy everything
except the timing of the entries isolates the one ingredient a signal is
supposed to contribute.

* :func:`random_entry_test` draws, in each simulation, exactly as many signal
  bars as the strategy traded, uniformly at random from the eligible bars,
  gives them the strategy's own sides (or a fair coin) and stop/target
  distances, resolves each through the engine's fill rules with the same
  costs, and records the metric. The strategy's metric is ranked against
  that null distribution.
* :func:`side_test` keeps the strategy's own decision bars and geometry and
  randomises (or flips) only the direction. It asks whether the strategy
  picks direction better than a coin given its timing.
* :func:`positive_control` runs a deliberately cheating oracle through the
  engine and through :func:`random_entry_test`; the harness must find it
  significant. A null test that cannot catch a cheat cannot clear a strategy.

P-values are one-sided (is the strategy's metric unusually *high*?) and are
computed as ``(1 + #{null >= observed}) / (1 + n_sims)``, which is never zero
and is exact for a Monte Carlo test with ``n_sims`` draws (Phipson & Smyth,
2010, "Permutation P-values Should Never Be Zero", Statistical Applications
in Genetics and Molecular Biology 9(1)). Null values within floating-point
rounding of the observed value count as ties, against the strategy.

The matching scheme (which features are copied, how unfillable setups are
handled, overlap) is a **heuristic** design with no canonical citation.

Limitations
-----------
* Only the entry timing (and, in :func:`side_test`, the direction) is under
  test. Edge that comes from the bracket geometry on a trending market, or
  from a long bias in a rising market, is shared by the null and therefore
  not credited to the signal. That is intended, but it means a low
  p-value says "the timing adds something", not "the strategy is profitable".
  Read the observed metric and costs separately.
* Random trades are resolved independently and may overlap in time, unlike
  the engine's one-position-at-a-time strategy runs. Nearby trades in the
  same direction have positively correlated outcomes, so allowing overlap
  makes the null slightly *wider* than a null of non-overlapping random
  trades would be, which errs on the conservative side. On a driftless
  random walk the resulting p-values of random strategies are uniform (the
  test suite checks this).
* ``eligible`` should describe where the strategy *could* have traded (its
  regime filter, trading hours). Drawing random entries from everywhere
  credits regime selection to the timing signal; drawing them from a mask
  fitted after looking at the results makes the null too easy or too hard.
* A p-value from one of many tried strategies must be corrected for multiple
  testing (see :mod:`backtest_skeptic.stats`).
* The strategy's trades must be reproducible with the settings passed here:
  by default each strategy trade is re-resolved through the engine and a
  mismatch (wrong costs, ``max_bars``, ``same_bar``, entry delay or bars)
  raises instead of producing a mismatched comparison.

Performance: random trades are resolved by a vectorised implementation of
the engine's fill rules (the same arithmetic as ``engine._resolve``, applied
to arrays). Every call cross-checks it against ``engine._resolve`` on the
strategy's own trades, and the test suite cross-checks it on thousands of
random setups. 1000 simulations of 300 trades take well under a second on
a laptop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import Bars
from .engine import (
    BacktestResult,
    SameBar,
    _resolve,
    _validate_options,
    resolve_many,
    run_signals,
)
from .synthetic import oracle_signal
from .trades import Costs, Setup, Trade

__all__ = [
    "Geometry",
    "Metric",
    "NullTestResult",
    "positive_control",
    "random_entry_test",
    "side_test",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Metric = Literal["expectancy", "win_rate", "total_r"] | Callable[[FloatArray], float]
Geometry = Literal["strategy"] | Callable[[Bars, int, int], tuple[float, float]]
Sides = Literal["match", "random", "long", "short"]
TestKind = Literal["random_entry", "side_random", "side_flip"]

_MIN_TRADES = 5
_TIE_RTOL = 1e-12
_REPRO_RTOL = 1e-9
_TEST_TITLES = {
    "random_entry": "Random-entry test",
    "side_random": "Random-side test",
    "side_flip": "Side-flip test",
}


# --------------------------------------------------------------------- result


@dataclass(frozen=True, eq=False)
class NullTestResult:
    """A strategy's metric ranked against a matched null distribution.

    Attributes
    ----------
    test:
        ``"random_entry"``, ``"side_random"`` or ``"side_flip"``.
    metric:
        Name of the metric (``"expectancy"`` is mean net R per trade,
        ``"win_rate"`` the share of trades with ``R > 0``, ``"total_r"`` the
        sum of net R; a callable's ``__name__`` otherwise).
    observed:
        The strategy's metric, computed from its own trades.
    null:
        One metric value per simulation (read-only). For ``"side_flip"`` a
        single value: the metric with every side reversed.
    p_value:
        One-sided Monte Carlo p-value ``(1 + #{null >= observed}) /
        (1 + n_sims)``; ``None`` for ``"side_flip"``, which is a single
        deterministic run.
    percentile:
        Mid-rank percentile of the observed value in the null, 0..100 (ties
        count half); ``None`` for ``"side_flip"``.
    null_mean, null_std:
        Mean and sample standard deviation (``ddof=1``) of the null;
        ``null_std`` is ``None`` with fewer than two simulations.
    z_score:
        ``(observed - null_mean) / null_std``, ``None`` when the null has no
        spread. A descriptive number; use ``p_value`` for inference, since
        the null need not be normal.
    n_trades:
        Strategy trades, which is also the number of trades in every random
        simulation of :func:`random_entry_test`.
    n_sims:
        Simulations run (1 for ``"side_flip"``).
    seed:
        Seed of the random generator (``None`` if unseeded or not random).
    n_eligible:
        Bars random entries were drawn from (``None`` for the side tests).
    unfilled:
        Random setups the engine could not fill because the next open was
        already beyond the stop or target. :func:`random_entry_test` redraws
        their bars (so every simulation still has ``n_trades`` trades);
        :func:`side_test` drops them, as the engine would.
    settings:
        The matching settings used (sides, geometry, costs, ``max_bars``,
        ``same_bar``, entry delay, warmup).
    notes:
        Warnings about the comparison.
    """

    test: TestKind
    metric: str
    observed: float
    null: FloatArray
    p_value: float | None
    percentile: float | None
    null_mean: float
    null_std: float | None
    z_score: float | None
    n_trades: int
    n_sims: int
    seed: int | None
    n_eligible: int | None = None
    unfilled: int = 0
    settings: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def excess(self) -> float:
        """``observed - null_mean``: how much the strategy beat the null on average.

        For ``"side_flip"`` this is observed minus flipped.
        """
        return self.observed - self.null_mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "metric": self.metric,
            "observed": self.observed,
            "null": self.null.tolist(),
            "p_value": self.p_value,
            "percentile": self.percentile,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z_score": self.z_score,
            "excess": self.excess,
            "n_trades": self.n_trades,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "n_eligible": self.n_eligible,
            "unfilled": self.unfilled,
            "settings": dict(self.settings),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        spec = ".1%" if self.metric == "win_rate" else "+.4f"

        def fmt(x: float | None, signed: bool = True) -> str:
            if x is None:
                return "n/a"
            return format(x, spec if signed else spec.lstrip("+"))

        title = _TEST_TITLES[self.test]
        if self.test == "side_flip":
            head = (
                f"{title} ({self.metric}, {self.n_trades} trades): observed "
                f"{fmt(self.observed)}, flipped {fmt(self.null_mean)}, "
                f"difference {fmt(self.excess)}"
            )
        else:
            z = "n/a" if self.z_score is None else f"{self.z_score:+.2f}"
            pct = "n/a" if self.percentile is None else f"{self.percentile:.1f}"
            p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
            head = (
                f"{title} ({self.metric}, {self.n_trades} trades): observed "
                f"{fmt(self.observed)} vs null {fmt(self.null_mean)} "
                f"(sd {fmt(self.null_std, signed=False)}) over {self.n_sims} simulations; "
                f"p = {p}, percentile {pct}, z {z}"
            )
        lines = [head]
        if self.unfilled:
            lines.append(f"  unfilled random setups: {self.unfilled}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ------------------------------------------------------------ public checks


def random_entry_test(
    bars: Bars,
    trades: Sequence[Trade] | BacktestResult,
    *,
    costs: Costs | None = None,
    max_bars: int | None = None,
    same_bar: SameBar | None = None,
    n_sims: int = 1000,
    seed: int | None = 0,
    sides: Sides = "match",
    geometry: Geometry = "strategy",
    eligible: ArrayLike | None = None,
    warmup: int = 0,
    metric: Metric = "expectancy",
    entry_delay: int | None = None,
    verify: bool = True,
) -> NullTestResult:
    """Rank a strategy against random entries matched on everything but timing.

    Catches entry signals that add nothing beyond the bracket's payoff shape,
    the costs, the direction mix and the period's drift: if random entries
    wearing the same brackets do as well, the signal is decoration.

    Parameters
    ----------
    bars:
        The bars the strategy was run on.
    trades:
        The strategy's :class:`~backtest_skeptic.Trade` objects (at least 5),
        or the :class:`~backtest_skeptic.BacktestResult` itself. A result
        supplies ``costs``, ``max_bars``, ``same_bar`` and ``entry_delay`` for
        any of those left at ``None`` here.
    costs, max_bars, same_bar, entry_delay:
        Engine settings, which must be the ones the strategy was run with.
        With a plain trade list, ``None`` means zero costs, no time limit,
        ``"stop"`` and the delay implied by the trades
        (``entry_index - signal_index - 1``, which must be the same for all).
    n_sims:
        Number of random simulations. The smallest attainable p-value is
        ``1 / (1 + n_sims)``.
    seed:
        Seed for :func:`numpy.random.default_rng`; results are reproducible
        for a given seed.
    sides:
        ``"match"`` (default): each simulation uses a random permutation of
        the strategy's own sides, so the long/short count is identical.
        ``"random"``: a fair coin per trade. ``"long"`` / ``"short"``: all one
        direction.
    geometry:
        ``"strategy"`` (default): each random trade copies the stop and
        target distances of a strategy trade picked at random, as fractions
        of the signal bar's close: ``stop_frac = side * (close[sig] - stop) /
        close[sig]``, ``target_frac = side * (target - close[sig]) /
        close[sig]``; a random entry at bar ``j`` with side ``s`` gets
        ``stop = close[j] * (1 - s * stop_frac)`` and ``target = close[j] *
        (1 + s * target_frac)``. Or a callable ``geometry(bars, j, side) ->
        (stop, target)`` for rules such as ATR multiples; it must only read
        bars ``0..j``.
    eligible:
        Optional boolean mask, one value per bar: bars a random entry may be
        decided on. Pass the strategy's regime or session filter here so the
        test measures timing within the regime, not regime selection.
        Random signal bars are always drawn from ``warmup .. len(bars) - 2 -
        entry_delay`` (bars after which a fill is possible).
    warmup:
        First bar a random entry may be decided on.
    metric:
        ``"expectancy"`` (mean net R, default), ``"win_rate"``, ``"total_r"``
        or a callable mapping a 1-D array of net R to a float. Higher must
        mean better.
    verify:
        Re-resolve every strategy trade through the engine with the given
        settings and raise ``ValueError`` if any does not reproduce. Turn off
        only for trade logs that did not come from this engine.

    Each simulation draws **exactly** ``len(trades)`` distinct signal bars
    uniformly at random. A random setup that cannot be filled (the next open
    is already beyond its stop or target) gets a fresh bar, keeping its side
    and geometry, so every simulation has exactly ``len(trades)`` trades.
    Random trades are resolved independently and may overlap; with
    correlated nearby outcomes this widens the null slightly (conservative).

    Returns
    -------
    NullTestResult
        ``p_value`` small means random entries rarely matched the strategy.
    """
    _check_bars(bars)
    tlist, cfg = _unpack(trades, costs, max_bars, same_bar, entry_delay, len(bars))
    n = len(tlist)
    n_sims = _int_option(n_sims, "n_sims", 1)
    seed = _seed_option(seed)
    warmup = _int_option(warmup, "warmup", 0)
    if sides not in ("match", "random", "long", "short"):
        raise ValueError(f"sides must be 'match', 'random', 'long' or 'short', got {sides!r}")
    metric_name, metric_rows = _metric(metric)
    if not isinstance(geometry, str) and not callable(geometry):
        raise ValueError(f"geometry must be 'strategy' or a callable, got {geometry!r}")
    if isinstance(geometry, str) and geometry != "strategy":
        raise ValueError(f"geometry must be 'strategy' or a callable, got {geometry!r}")

    observed_r = _strategy_r(bars, tlist, cfg, verify)
    observed = float(metric_rows(observed_r[None, :])[0])

    pool, notes = _eligible_pool(bars, eligible, warmup, cfg.entry_delay, tlist)
    m = len(pool)
    if m < n:
        raise ValueError(
            f"only {m} eligible signal bars for {n} trades; each simulation draws "
            f"{n} distinct bars, so widen the eligible set or shorten warmup"
        )

    rng = np.random.default_rng(seed)
    strat_sides = np.array([t.side for t in tlist], dtype=np.float64)
    side_mat = _draw_sides(rng, sides, strat_sides, n_sims)

    close = bars.close
    level_fn: Callable[[IntArray, FloatArray, IntArray], tuple[FloatArray, FloatArray]]
    if isinstance(geometry, str):
        stop_frac, target_frac = _strategy_fractions(close, tlist)
        tmpl = rng.integers(0, n, size=(n_sims, n)).ravel()
        level_fn = _fraction_levels(close, stop_frac, target_frac)
        geometry_name = "strategy"
    else:
        tmpl = np.zeros(n_sims * n, dtype=np.int64)
        level_fn = _callable_levels(bars, geometry)
        geometry_name = getattr(geometry, "__name__", "callable")

    extra = min(m - n, max(4, n // 10))
    cand = [rng.choice(m, size=n + extra, replace=False) for _ in range(n_sims)]
    sig = pool[np.stack([c[:n] for c in cand])].ravel()
    side_flat = side_mat.ravel()

    stop, target = level_fn(sig, side_flat, tmpl)
    r, _ = _resolve_batch(bars, sig, side_flat, stop, target, cfg)

    unfilled = 0
    pending = np.flatnonzero(np.isnan(r))
    ptr = np.full(n_sims, n, dtype=np.int64)
    while pending.size:
        unfilled += int(pending.size)
        new_sig = np.empty(pending.size, dtype=np.int64)
        for i, row in enumerate(pending.tolist()):
            s = row // n
            if ptr[s] >= len(cand[s]):
                cand[s] = _extend_candidates(rng, cand[s], m)
                if ptr[s] >= len(cand[s]):
                    raise ValueError(
                        f"could not fill {n} random trades in simulation {s}: every eligible "
                        "bar was tried and too few give a fillable setup with this geometry "
                        "and these costs"
                    )
            new_sig[i] = pool[cand[s][ptr[s]]]
            ptr[s] += 1
        sig[pending] = new_sig
        p_stop, p_target = level_fn(new_sig, side_flat[pending], tmpl[pending])
        p_r, _ = _resolve_batch(bars, new_sig, side_flat[pending], p_stop, p_target, cfg)
        r[pending] = p_r
        pending = pending[np.isnan(p_r)]

    r_mat = r.reshape(n_sims, n)
    null = metric_rows(r_mat)
    settings = cfg.as_dict()
    settings.update(
        sides=sides, geometry=geometry_name, warmup=warmup, eligible=eligible is not None
    )
    return _result(
        "random_entry",
        metric_name,
        observed,
        null,
        n_trades=n,
        seed=seed,
        n_eligible=m,
        unfilled=unfilled,
        settings=settings,
        notes=notes,
    )


def side_test(
    bars: Bars,
    trades: Sequence[Trade] | BacktestResult,
    *,
    mode: Literal["random", "flip"] = "random",
    n_sims: int = 1000,
    seed: int | None = 0,
    costs: Costs | None = None,
    max_bars: int | None = None,
    same_bar: SameBar | None = None,
    metric: Metric = "expectancy",
    entry_delay: int | None = None,
    verify: bool = True,
) -> NullTestResult:
    """Does the strategy choose direction better than a coin, given its timing?

    Keeps the strategy's own decision bars (``trade.signal_index``) and
    bracket distances and changes only the side. A side ``s`` at signal bar
    ``i`` gets ``stop = close[i] - s * d_stop`` and ``target = close[i] + s *
    d_target``, where ``d_stop = side * (close[i] - stop)`` and ``d_target =
    side * (target - close[i])`` are the strategy's own distances (trades
    whose side is unchanged keep their exact levels).

    Catches direction logic that adds nothing: if a coin choosing the side at
    the same bars does as well, any edge comes from *when* the strategy
    trades, not from the direction it picks. Real direction skill shows as a
    small p-value, and reversing it (``mode="flip"``) should hurt. A constant
    long bias in a rising market counts as direction skill here; use
    :func:`random_entry_test` with ``sides="match"`` to control for it.

    Parameters
    ----------
    mode:
        ``"random"``: each simulation assigns fair-coin sides; the p-value
        ranks the strategy against them. ``"flip"``: a single deterministic
        run with every side reversed; ``null`` holds the flipped metric,
        ``excess`` is observed minus flipped and ``p_value`` is ``None``.
    n_sims, seed:
        Simulations and seed for ``"random"`` (ignored for ``"flip"``).
    costs, max_bars, same_bar, entry_delay, verify, metric:
        As in :func:`random_entry_test`.

    Limitations: a reversed setup the engine cannot fill (the next open has
    gapped beyond its new stop or target) is dropped, as the engine would
    drop it, so a simulation can have slightly fewer trades; the count is in
    ``unfilled``. Trades are resolved independently, so reversing a trade
    does not change which later trades the strategy would have been free to
    take. Timing and direction skill are not separable when the strategy's
    side is a function of the same information as its timing; the test
    measures direction given the timing actually used.
    """
    _check_bars(bars)
    tlist, cfg = _unpack(trades, costs, max_bars, same_bar, entry_delay, len(bars))
    n = len(tlist)
    if mode not in ("random", "flip"):
        raise ValueError(f"mode must be 'random' or 'flip', got {mode!r}")
    metric_name, metric_rows = _metric(metric)
    n_sims = _int_option(n_sims, "n_sims", 1)
    seed = _seed_option(seed)

    observed_r = _strategy_r(bars, tlist, cfg, verify)
    observed = float(metric_rows(observed_r[None, :])[0])

    close = bars.close
    sig = np.array([t.signal_index for t in tlist], dtype=np.int64)
    orig_side = np.array([t.side for t in tlist], dtype=np.float64)
    orig_stop = np.array([t.stop for t in tlist], dtype=np.float64)
    orig_target = np.array([t.target for t in tlist], dtype=np.float64)
    c_sig = close[sig]
    d_stop = orig_side * (c_sig - orig_stop)
    d_target = orig_side * (orig_target - c_sig)

    if mode == "flip":
        side_mat = -orig_side[None, :]
        rows, used_seed = 1, None
    else:
        rng = np.random.default_rng(seed)
        side_mat = rng.choice(np.array([-1.0, 1.0]), size=(n_sims, n))
        rows, used_seed = n_sims, seed

    side_flat = side_mat.ravel()
    sig_flat = np.tile(sig, rows)
    same = side_flat == np.tile(orig_side, rows)
    stop = np.where(
        same, np.tile(orig_stop, rows), np.tile(c_sig, rows) - side_flat * np.tile(d_stop, rows)
    )
    target = np.where(
        same, np.tile(orig_target, rows), np.tile(c_sig, rows) + side_flat * np.tile(d_target, rows)
    )
    r, _ = _resolve_batch(bars, sig_flat, side_flat, stop, target, cfg)
    r_mat = r.reshape(rows, n)
    unfilled = int(np.isnan(r).sum())
    if (np.isnan(r_mat).all(axis=1)).any():
        raise ValueError(
            "a simulation has no fillable trades; the reversed setups cannot be filled"
        )
    null = metric_rows(r_mat)
    notes: list[str] = []
    if unfilled:
        notes.append(
            f"{unfilled} reversed setups could not be filled and were dropped from their simulation"
        )
    settings = cfg.as_dict()
    settings.update(mode=mode)
    return _result(
        "side_flip" if mode == "flip" else "side_random",
        metric_name,
        observed,
        null,
        n_trades=n,
        seed=used_seed,
        n_eligible=None,
        unfilled=unfilled,
        settings=settings,
        notes=tuple(notes),
    )


def positive_control(
    bars: Bars,
    *,
    horizon: int = 5,
    stop_frac: float = 0.01,
    target_frac: float = 0.01,
    n_sims: int = 500,
    seed: int | None = 0,
    costs: Costs | None = None,
) -> NullTestResult:
    """Prove the random-entry harness can catch a cheat on these bars.

    Runs :func:`backtest_skeptic.synthetic.oracle_signal` (which reads
    ``close[i + horizon]``) through :func:`~backtest_skeptic.run_signals`
    with brackets ``stop = close[i] * (1 - side * stop_frac)`` and ``target =
    close[i] * (1 + side * target_frac)``, then tests it with
    :func:`random_entry_test`. On any realistic price series the result's
    ``p_value`` should be below 0.01 (reaching that needs ``n_sims >= 99``);
    if it is not, the harness, the data or the settings cannot detect even a
    blatant edge and other results from it should not be trusted.

    Fees are zero: the oracle is judged against matched random entries, not
    against a fee-inclusive profit threshold, so the control tests the
    harness and not the cost model. ``costs`` may add spread and slippage
    (charged to the oracle and to the random entries alike) but must have
    ``fee_rate == 0``.
    """
    _check_bars(bars)
    horizon = _int_option(horizon, "horizon", 1)
    sf = _positive_float(stop_frac, "stop_frac")
    tf = _positive_float(target_frac, "target_frac")
    if sf >= 1:
        raise ValueError(f"stop_frac must be below 1, got {stop_frac!r}")
    costs = Costs() if costs is None else costs
    if not isinstance(costs, Costs):
        raise ValueError(f"costs must be a Costs instance or None, got {type(costs).__name__}")
    if costs.fee_rate != 0:
        raise ValueError(
            "positive_control runs with zero fees so it tests the harness, not the cost "
            f"model; pass costs with fee_rate=0 (got {costs.fee_rate})"
        )
    signal = oracle_signal(bars, horizon).astype(np.float64)
    close = bars.close
    stop = close * (1.0 - signal * sf)
    target = close * (1.0 + signal * tf)
    result = run_signals(bars, signal, stop, target, costs=costs)
    if len(result.trades) < _MIN_TRADES:
        raise ValueError(
            f"the oracle made only {len(result.trades)} trades on these bars; need at least "
            f"{_MIN_TRADES} (use more bars or narrower brackets)"
        )
    return random_entry_test(
        bars, result, n_sims=n_sims, seed=seed, sides="match", geometry="strategy"
    )


# ------------------------------------------------------------ settings


@dataclass(frozen=True)
class _Config:
    costs: Costs
    max_bars: int | None
    same_bar: SameBar
    entry_delay: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "costs": {
                "half_spread": self.costs.half_spread,
                "fee_rate": self.costs.fee_rate,
                "slippage": self.costs.slippage,
            },
            "max_bars": self.max_bars,
            "same_bar": self.same_bar,
            "entry_delay": self.entry_delay,
        }


def _check_bars(bars: Any) -> None:
    if not isinstance(bars, Bars):
        raise ValueError(f"bars must be a Bars instance, got {type(bars).__name__}")
    if len(bars) < 3:
        raise ValueError(f"need at least 3 bars, got {len(bars)}")


def _int_option(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _seed_option(seed: Any) -> int | None:
    if seed is None:
        return None
    return _int_option(seed, "seed", 0)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


def _unpack(
    trades: Sequence[Trade] | BacktestResult,
    costs: Costs | None,
    max_bars: int | None,
    same_bar: SameBar | None,
    entry_delay: int | None,
    n_bars: int,
) -> tuple[tuple[Trade, ...], _Config]:
    """Trade tuple plus engine settings, from a result or explicit arguments."""
    if isinstance(trades, BacktestResult):
        tlist = trades.trades
        costs = trades.costs if costs is None else costs
        max_bars = trades.max_bars if max_bars is None else max_bars
        same_bar = trades.same_bar if same_bar is None else same_bar
        entry_delay = trades.entry_delay if entry_delay is None else entry_delay
    else:
        given: Any = trades
        if isinstance(given, (str, bytes)) or not isinstance(given, Sequence):
            raise ValueError(
                "trades must be a sequence of Trade objects or a BacktestResult, "
                f"got {type(given).__name__}"
            )
        tlist = tuple(given)
    for t in tlist:
        if not isinstance(t, Trade):
            raise ValueError(f"trades must contain Trade objects, got {type(t).__name__}")
    if len(tlist) < _MIN_TRADES:
        raise ValueError(f"need at least {_MIN_TRADES} trades, got {len(tlist)}")
    costs = Costs() if costs is None else costs
    if not isinstance(costs, Costs):
        raise ValueError(f"costs must be a Costs instance or None, got {type(costs).__name__}")
    same_bar = "stop" if same_bar is None else same_bar
    delays = {t.entry_index - t.signal_index - 1 for t in tlist}
    if entry_delay is None:
        if len(delays) != 1:
            raise ValueError(
                f"trades imply different entry delays {sorted(delays)}; pass entry_delay"
            )
        entry_delay = delays.pop()
    _validate_options(max_bars, entry_delay, same_bar, None)
    entry_delay = int(entry_delay)
    max_bars = None if max_bars is None else int(max_bars)
    last_signal = n_bars - 2 - entry_delay
    for k, t in enumerate(tlist):
        if not 0 <= t.signal_index <= last_signal:
            raise ValueError(
                f"trade {k} has signal_index {t.signal_index}, outside 0..{last_signal} for "
                f"{n_bars} bars with entry_delay {entry_delay}; are these the right bars?"
            )
    return tlist, _Config(costs, max_bars, same_bar, entry_delay)


def _strategy_r(bars: Bars, tlist: tuple[Trade, ...], cfg: _Config, verify: bool) -> FloatArray:
    """The strategy's net R, after checking the settings reproduce its trades.

    Always cross-checks the vectorised resolver against ``engine._resolve`` on
    the strategy's own setups, so a future change to the engine's fill rules
    cannot silently desynchronise the null from the strategy.
    """
    observed = np.array([t.r for t in tlist], dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("trade R values must be finite")
    sig = np.array([t.signal_index for t in tlist], dtype=np.int64)
    side = np.array([t.side for t in tlist], dtype=np.float64)
    stop = np.array([t.stop for t in tlist], dtype=np.float64)
    target = np.array([t.target for t in tlist], dtype=np.float64)
    if not (np.isfinite(stop).all() and np.isfinite(target).all()):
        raise ValueError("trade stop and target levels must be finite")
    batch_r, batch_exit = _resolve_batch(bars, sig, side, stop, target, cfg)

    ref_r = np.full(len(tlist), np.nan)
    ref_exit = np.full(len(tlist), -1, dtype=np.int64)
    for k, t in enumerate(tlist):
        out = _resolve(
            bars,
            t.signal_index,
            Setup(t.side, t.stop, t.target),
            cfg.costs,
            cfg.max_bars,
            cfg.entry_delay,
            cfg.same_bar,
            None,
        )
        if isinstance(out, Trade):
            ref_r[k] = out.r
            ref_exit[k] = out.exit_index
    same_r = (batch_r == ref_r) | (np.isnan(batch_r) & np.isnan(ref_r))
    if not (same_r.all() and (batch_exit == ref_exit).all()):
        k = int(np.argmax(~same_r | (batch_exit != ref_exit)))
        raise RuntimeError(
            "internal error: the vectorised resolver disagrees with engine._resolve on trade "
            f"{k} ({batch_r[k]!r} vs {ref_r[k]!r}); please report this"
        )

    if verify:
        filled = ~np.isnan(ref_r)
        tol = _REPRO_RTOL * np.maximum(1.0, np.abs(observed))
        ok = filled & (np.abs(np.where(filled, ref_r, 0.0) - observed) <= tol)
        ok &= ref_exit == np.array([t.exit_index for t in tlist], dtype=np.int64)
        if not ok.all():
            k = int(np.argmax(~ok))
            t = tlist[k]
            got = "no fill" if not filled[k] else f"R {ref_r[k]:+.6f} exiting at bar {ref_exit[k]}"
            raise ValueError(
                f"trade {k} (signal bar {t.signal_index}) does not reproduce with these "
                f"settings: the trade has R {t.r:+.6f} exiting at bar {t.exit_index}, the engine "
                f"gives {got}. Pass the costs, max_bars, same_bar and entry_delay the strategy "
                "was run with (or the BacktestResult itself), or verify=False for an external "
                "trade log"
            )
    return observed


def _eligible_pool(
    bars: Bars,
    eligible: ArrayLike | None,
    warmup: int,
    entry_delay: int,
    tlist: tuple[Trade, ...],
) -> tuple[IntArray, tuple[str, ...]]:
    n = len(bars)
    last_signal = n - 2 - entry_delay
    in_range = np.zeros(n, dtype=bool)
    in_range[warmup : last_signal + 1] = True
    notes: list[str] = []
    if eligible is not None:
        mask = np.asarray(eligible)
        if mask.dtype != np.bool_:
            raise ValueError(f"eligible must be a boolean mask, got dtype {mask.dtype}")
        if mask.shape != (n,):
            raise ValueError(f"eligible must have one value per bar ({n}), got shape {mask.shape}")
        in_range &= mask
    sig = np.array([t.signal_index for t in tlist], dtype=np.int64)
    outside = int((~in_range[sig]).sum())
    if outside:
        notes.append(
            f"{outside} strategy trades were decided on bars outside the random-entry pool "
            "(eligible mask or warmup); the null may not describe where the strategy trades"
        )
    return np.flatnonzero(in_range).astype(np.int64), tuple(notes)


def _draw_sides(
    rng: np.random.Generator, sides: Sides, strat_sides: FloatArray, n_sims: int
) -> FloatArray:
    n = len(strat_sides)
    if sides == "match":
        out: FloatArray = rng.permuted(np.tile(strat_sides, (n_sims, 1)), axis=1)
        return out
    if sides == "random":
        coin: FloatArray = rng.choice(np.array([-1.0, 1.0]), size=(n_sims, n))
        return coin
    return np.full((n_sims, n), 1.0 if sides == "long" else -1.0)


def _strategy_fractions(
    close: FloatArray, tlist: tuple[Trade, ...]
) -> tuple[FloatArray, FloatArray]:
    sig = np.array([t.signal_index for t in tlist], dtype=np.int64)
    side = np.array([t.side for t in tlist], dtype=np.float64)
    stop = np.array([t.stop for t in tlist], dtype=np.float64)
    target = np.array([t.target for t in tlist], dtype=np.float64)
    c = close[sig]
    stop_frac = side * (c - stop) / c
    target_frac = side * (target - c) / c
    bad = (stop_frac <= 0) | (target_frac <= 0) | (stop_frac >= 1)
    if bad.any():
        k = int(np.argmax(bad))
        raise ValueError(
            f"trade {k} has its stop or target on the wrong side of its signal bar's close "
            f"(stop fraction {stop_frac[k]:.6g}, target fraction {target_frac[k]:.6g}), so its "
            "geometry cannot be copied to other bars; pass a geometry callable instead"
        )
    return stop_frac, target_frac


def _fraction_levels(
    close: FloatArray, stop_frac: FloatArray, target_frac: FloatArray
) -> Callable[[IntArray, FloatArray, IntArray], tuple[FloatArray, FloatArray]]:
    def levels(sig: IntArray, side: FloatArray, tmpl: IntArray) -> tuple[FloatArray, FloatArray]:
        c = close[sig]
        return c * (1.0 - side * stop_frac[tmpl]), c * (1.0 + side * target_frac[tmpl])

    return levels


def _callable_levels(
    bars: Bars, geometry: Callable[[Bars, int, int], tuple[float, float]]
) -> Callable[[IntArray, FloatArray, IntArray], tuple[FloatArray, FloatArray]]:
    def levels(sig: IntArray, side: FloatArray, tmpl: IntArray) -> tuple[FloatArray, FloatArray]:
        stop = np.empty(len(sig))
        target = np.empty(len(sig))
        for k, (j, s) in enumerate(zip(sig.tolist(), side.tolist(), strict=True)):
            out = geometry(bars, j, int(s))
            try:
                st, tg = out
                stop[k] = float(st)
                target[k] = float(tg)
            except (TypeError, ValueError):
                raise ValueError(
                    f"geometry(bars, {j}, {int(s)}) must return (stop, target), got {out!r}"
                ) from None
            if not (math.isfinite(stop[k]) and math.isfinite(target[k])):
                raise ValueError(
                    f"geometry(bars, {j}, {int(s)}) returned non-finite levels {out!r}"
                )
        return stop, target

    return levels


def _extend_candidates(rng: np.random.Generator, cand: IntArray, m: int) -> IntArray:
    """Append every not-yet-drawn pool index, in random order."""
    if len(cand) >= m:
        return cand
    used = np.zeros(m, dtype=bool)
    used[cand] = True
    rest = np.flatnonzero(~used)
    return np.concatenate([cand, rng.permutation(rest)])


# ------------------------------------------------------------ metrics


def _metric(metric: Metric) -> tuple[str, Callable[[FloatArray], FloatArray]]:
    """Name and a row-wise evaluator for a 2-D R matrix (NaN = no trade)."""
    if isinstance(metric, str):
        if metric == "expectancy":
            return metric, _row_expectancy
        if metric == "win_rate":
            return metric, _row_win_rate
        if metric == "total_r":
            return metric, _row_total
        raise ValueError(
            f"metric must be 'expectancy', 'win_rate', 'total_r' or a callable, got {metric!r}"
        )
    if not callable(metric):
        raise ValueError(f"metric must be a string or a callable, got {metric!r}")
    fn = metric
    name = str(getattr(metric, "__name__", "custom"))

    def rows(r: FloatArray) -> FloatArray:
        out = np.empty(r.shape[0])
        for i in range(r.shape[0]):
            row = r[i]
            value = fn(row[np.isfinite(row)])
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"metric {name} must return a number, got {value!r}") from None
            if not math.isfinite(out[i]):
                raise ValueError(f"metric {name} returned a non-finite value {out[i]}")
        return out

    return name, rows


def _row_expectancy(r: FloatArray) -> FloatArray:
    filled = np.isfinite(r)
    out: FloatArray = np.where(filled, r, 0.0).sum(axis=1) / filled.sum(axis=1)
    return out


def _row_win_rate(r: FloatArray) -> FloatArray:
    filled = np.isfinite(r)
    out: FloatArray = (filled & (np.where(filled, r, 0.0) > 0)).sum(axis=1) / filled.sum(axis=1)
    return out


def _row_total(r: FloatArray) -> FloatArray:
    out: FloatArray = np.where(np.isfinite(r), r, 0.0).sum(axis=1)
    return out


def _result(
    test: TestKind,
    metric: str,
    observed: float,
    null: FloatArray,
    *,
    n_trades: int,
    seed: int | None,
    n_eligible: int | None,
    unfilled: int,
    settings: dict[str, Any],
    notes: tuple[str, ...],
) -> NullTestResult:
    null = np.ascontiguousarray(null, dtype=np.float64)
    null.setflags(write=False)
    n_sims = len(null)
    mean = float(null.mean())
    std = float(null.std(ddof=1)) if n_sims > 1 else None
    if test == "side_flip":
        p_value = percentile = z = None
    else:
        tol = _TIE_RTOL * max(1.0, abs(observed))
        tied = np.abs(null - observed) <= tol
        above = int(((null > observed) | tied).sum())
        below = int(((null < observed) & ~tied).sum())
        p_value = (1.0 + above) / (1.0 + n_sims)
        percentile = 100.0 * (below + 0.5 * int(tied.sum())) / n_sims
        scale = max(1.0, abs(mean))
        z = None if std is None or std <= 1e-12 * scale else (observed - mean) / std
    return NullTestResult(
        test=test,
        metric=metric,
        observed=observed,
        null=null,
        p_value=p_value,
        percentile=percentile,
        null_mean=mean,
        null_std=std,
        z_score=z,
        n_trades=n_trades,
        n_sims=n_sims,
        seed=seed,
        n_eligible=n_eligible,
        unfilled=unfilled,
        settings=settings,
        notes=notes,
    )


# ------------------------------------------------------------ resolver


def _resolve_batch(
    bars: Bars,
    sig: IntArray,
    side: FloatArray,
    stop: FloatArray,
    target: FloatArray,
    cfg: _Config,
) -> tuple[FloatArray, IntArray]:
    """:func:`engine.resolve_many` with this run's settings."""
    return resolve_many(
        bars,
        sig,
        side,
        stop,
        target,
        costs=cfg.costs,
        max_bars=cfg.max_bars,
        entry_delay=cfg.entry_delay,
        same_bar=cfg.same_bar,
    )
