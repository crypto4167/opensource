"""Monte Carlo calibration of the statistical checks.

Under a true null, a test run at alpha = 0.05 should reject about 5% of the
time; under a planted edge it should reject more often (its power). This
script measures both for the checks in ``stats``, ``baseline`` and
``random_entry`` and prints one table row per scenario with the Monte Carlo
standard error ``sqrt(rate * (1 - rate) / reps)``.

Every scenario is seeded, so the table is reproducible. The default run takes
about ten minutes on a laptop; ``--scale 0.2`` gives a quick, noisier pass.
The numbers in ``docs/calibration.md`` come from the default run.

    python scripts/calibration.py [--scale S] [--only SECTION ...]
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Iterable
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from backtest_skeptic import Costs
from backtest_skeptic.baseline import no_skill_test
from backtest_skeptic.engine import BacktestResult, run_signals
from backtest_skeptic.random_entry import random_entry_test, side_test
from backtest_skeptic.stats import (
    adjust_pvalues,
    deflated_sharpe_ratio,
    newey_west,
    pbo,
    probabilistic_sharpe_ratio,
)
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars

ALPHA = 0.05
_NORMAL = NormalDist()
FloatArray = NDArray[np.float64]


def row(section: str, scenario: str, hits: Iterable[float], note: str = "") -> None:
    """Print one table row: the mean of ``hits`` (0/1 or a rate) and its MC SE."""
    x = np.asarray(list(hits), dtype=np.float64)
    rate = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    print(f"| {section} | {scenario} | {len(x)} | {rate:.4f} | {se:.4f} | {note} |", flush=True)


def reps(base: int, scale: float) -> int:
    return max(20, round(base * scale))


# ---------------------------------------------------------------- stats


def psr_section(scale: float) -> None:
    rng = np.random.default_rng(101)
    gens: dict[str, Callable[[int], FloatArray]] = {
        "normal": lambda n: rng.normal(0.0, 1.0, n),
        "Student t5": lambda n: rng.standard_t(5, n),
        "Student t3": lambda n: rng.standard_t(3, n),
        "neg. skew (-lognormal)": lambda n: np.exp(0.5) - rng.lognormal(0.0, 1.0, n),
    }
    r = reps(2000, scale)
    for name, gen in gens.items():
        for n in (60, 250, 1000):
            hits = [probabilistic_sharpe_ratio(gen(n)) > 1 - ALPHA for _ in range(r)]
            row("PSR null (SR = 0)", f"{name}, n={n}", hits)
    for sr in (0.1, 0.2):
        hits = [probabilistic_sharpe_ratio(rng.normal(sr, 1.0, 250)) > 1 - ALPHA for _ in range(r)]
        row("PSR power", f"normal, true SR {sr}/period, n=250", hits)


def dsr_section(scale: float) -> None:
    rng = np.random.default_rng(102)
    r = reps(1000, scale)
    for dist in ("normal", "Student t5"):
        for n_trials, t_obs in ((10, 250), (10, 1000), (100, 250), (100, 1000)):
            hits = []
            for _ in range(r):
                x = (
                    rng.normal(size=(t_obs, n_trials))
                    if dist == "normal"
                    else rng.standard_t(5, size=(t_obs, n_trials))
                )
                sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
                best = int(np.argmax(sr))
                dsr = deflated_sharpe_ratio(
                    x[:, best], n_trials=n_trials, var_trials_sharpe=float(sr.var(ddof=1))
                )
                hits.append(dsr > 1 - ALPHA)
            row("DSR null (best of N)", f"{dist}, N={n_trials}, T={t_obs}", hits)
    for edge in (0.1, 0.2):
        hits = []
        for _ in range(r):
            x = rng.normal(size=(1000, 10))
            x[:, 0] += edge
            sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
            best = int(np.argmax(sr))
            dsr = deflated_sharpe_ratio(
                x[:, best], n_trials=10, var_trials_sharpe=float(sr.var(ddof=1))
            )
            hits.append(dsr > 1 - ALPHA)
        row("DSR power", f"one of N=10 has SR {edge}/period, T=1000", hits)


def _ar1(rng: np.random.Generator, r: int, t_obs: int, phi: float) -> FloatArray:
    burn = 200
    e = rng.normal(size=(r, t_obs + burn))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for i in range(1, t_obs + burn):
        x[:, i] = phi * x[:, i - 1] + e[:, i]
    out: FloatArray = x[:, burn:]
    return out


def newey_west_section(scale: float) -> None:
    rng = np.random.default_rng(103)
    r = reps(2000, scale)
    for t_obs in (250, 1000):
        for phi in (0.0, 0.3, 0.6):
            x = _ar1(rng, r, t_obs, phi)
            hits = [abs(newey_west(row_).t) > 1.96 for row_ in x]
            naive = [abs(row_.mean() / row_.std(ddof=1) * math.sqrt(t_obs)) > 1.96 for row_ in x]
            row("Newey-West null", f"AR(1) phi={phi}, T={t_obs}, default lags", hits)
            row("plain t null", f"AR(1) phi={phi}, T={t_obs}", naive)
    for t_obs in (250, 1000):
        for h in (5, 20):
            e = rng.normal(size=(r, t_obs + h - 1))
            c = np.cumsum(e, axis=1)
            c = np.concatenate([np.zeros((r, 1)), c], axis=1)
            x = c[:, h:] - c[:, :-h]
            for kernel in ("bartlett", "uniform"):
                hits = [abs(newey_west(v, lags=h - 1, kernel=kernel).t) > 1.96 for v in x]
                row("Newey-West null", f"{h}-period overlapping sums, T={t_obs}, {kernel}", hits)


def adjust_section(scale: float) -> None:
    rng = np.random.default_rng(104)
    r = reps(5000, scale)
    m = 20
    cdf = np.vectorize(_NORMAL.cdf)
    for m0 in (20, 15):
        for rho in (0.0, 0.5):
            fwer: dict[str, list[bool]] = {k: [] for k in ("bonferroni", "holm", "bh", "by")}
            fdp: dict[str, list[float]] = {k: [] for k in fwer}
            for _ in range(r):
                z = math.sqrt(rho) * rng.normal() + math.sqrt(1 - rho) * rng.normal(size=m)
                z[m0:] += 3.0
                p = 1.0 - cdf(z)
                for k in fwer:
                    rej = adjust_pvalues(p, k) <= ALPHA  # type: ignore[arg-type]
                    false = int(rej[:m0].sum())
                    total = int(rej.sum())
                    fwer[k].append(false > 0)
                    fdp[k].append(false / total if total else 0.0)
            label = f"m=20, {m0} true nulls, equicorrelation {rho}"
            for k in ("bonferroni", "holm"):
                row(f"adjust_pvalues {k} FWER", label, fwer[k])
            bound = f"bound {ALPHA * m0 / m:.4f}"
            for k in ("bh", "by"):
                row(f"adjust_pvalues {k} FDR", label, fdp[k], bound)


def pbo_section(scale: float) -> None:
    rng = np.random.default_rng(105)
    r = reps(200, scale)
    for t_obs, n_cfg, s in ((1000, 10, 16), (1000, 50, 16), (500, 10, 8), (1001, 7, 10)):
        vals = [pbo(rng.normal(size=(t_obs, n_cfg)), n_splits=s).pbo for _ in range(r)]
        row("PBO mean, pure noise", f"T={t_obs}, N={n_cfg}, S={s}", vals, "should be near 0.5")
    for edge in (0.05, 0.1):
        vals = []
        for _ in range(r):
            x = rng.normal(size=(1000, 10))
            x[:, 0] += edge
            vals.append(pbo(x, n_splits=10).pbo)
        row("PBO mean, one real edge", f"SR {edge}/period, T=1000, N=10, S=10", vals)


# ---------------------------------------------------------------- trades


def _random_strategy(
    seed: int,
    *,
    n: int = 2000,
    prob: float = 0.05,
    stop_frac: float = 0.01,
    target_frac: float = 0.01,
    costs: Costs | None = None,
    max_bars: int | None = None,
    drift: float = 0.0,
    gap_vol: float = 0.0,
    side_dependent: bool = False,
) -> tuple[object, BacktestResult]:
    """Random entries and random sides on a random walk (no skill)."""
    rng = np.random.default_rng(seed + 1_000_000)
    bars = random_walk_bars(n, seed=seed, drift=drift, gap_vol=gap_vol)
    side = np.where(rng.random(n) < prob, rng.choice([-1, 1], n), 0).astype(np.float64)
    c = bars.close
    sf: FloatArray | float = stop_frac
    tf: FloatArray | float = target_frac
    if side_dependent:
        sf = np.where(side > 0, stop_frac, 2 * stop_frac)
        tf = np.where(side > 0, target_frac, target_frac / 4)
    stop = c * (1 - side * sf)
    target = c * (1 + side * tf)
    return bars, run_signals(bars, side, stop, target, costs=costs, max_bars=max_bars)


def _planted_strategy(
    seed: int, edge: float, *, n: int = 3000, frac: float = 0.01, max_bars: int | None = 1
) -> tuple[object, BacktestResult]:
    bars, signal = planted_signal_bars(n, edge=edge, signal_prob=0.1, seed=seed)
    side = signal.astype(np.float64)
    c = bars.close
    res = run_signals(bars, side, c * (1 - side * frac), c * (1 + side * frac), max_bars=max_bars)
    return bars, res


NULL_TRADE_SCENARIOS: dict[str, dict[str, object]] = {
    "1%/1% brackets": {},
    "3% stop / 0.5% target": {"stop_frac": 0.03, "target_frac": 0.005},
    "0.5% stop / 3% target": {"stop_frac": 0.005, "target_frac": 0.03},
    "1%/1%, costs": {"costs": Costs(0.02, 0.0005, 0.02)},
    "1%/1%, max_bars=5": {"max_bars": 5},
    "1%/1%, gaps (gap vol 0.3%)": {"gap_vol": 0.003},
    "1%/1%, drift +0.2%/bar": {"drift": 0.002},
    "side-dependent brackets, drift +0.2%/bar": {"drift": 0.002, "side_dependent": True},
}


def random_entry_section(scale: float) -> None:
    r = reps(400, scale)
    for name, kw in NULL_TRADE_SCENARIOS.items():
        pe, ps = [], []
        for seed in range(r):
            bars, res = _random_strategy(seed, **kw)  # type: ignore[arg-type]
            pe.append(random_entry_test(bars, res, n_sims=199, seed=seed).p_value or 1.0)  # type: ignore[arg-type]
            ps.append(side_test(bars, res, n_sims=199, seed=seed).p_value or 1.0)  # type: ignore[arg-type]
        row("random_entry_test null", name, [p <= ALPHA for p in pe], "199 sims, about 95 trades")
        if not kw.get("drift"):
            row("side_test null", name, [p <= ALPHA for p in ps], "199 sims")
    r = reps(200, scale)
    for edge in (0.05, 0.1, 0.2, 0.3):
        pe, ps = [], []
        for seed in range(r):
            bars, res = _planted_strategy(seed, edge)
            pe.append(random_entry_test(bars, res, n_sims=199, seed=seed).p_value or 1.0)  # type: ignore[arg-type]
            ps.append(side_test(bars, res, n_sims=199, seed=seed).p_value or 1.0)  # type: ignore[arg-type]
        note = "1%/1% brackets, max_bars=1, about 300 trades"
        row("random_entry_test power", f"planted edge {edge}", [p <= ALPHA for p in pe], note)
        row("side_test power", f"planted edge {edge}", [p <= ALPHA for p in ps], note)


BASELINE_SCENARIOS: dict[str, dict[str, object]] = {
    "1%/1% brackets": {"n": 4000},
    "3% stop / 0.5% target": {"n": 4000, "stop_frac": 0.03, "target_frac": 0.005},
    "0.5% stop / 3% target": {"n": 4000, "stop_frac": 0.005, "target_frac": 0.03},
    "1%/1%, costs": {"n": 4000, "costs": Costs(0.02, 0.0005, 0.02)},
    "3% stop / 0.5% target, max_bars=5": {
        "n": 4000,
        "stop_frac": 0.03,
        "target_frac": 0.005,
        "max_bars": 5,
    },
    "0.5% stop / 3% target, max_bars=5": {
        "n": 4000,
        "stop_frac": 0.005,
        "target_frac": 0.03,
        "max_bars": 5,
    },
    "0.5% stop / 3% target, max_bars=5, costs": {
        "n": 4000,
        "stop_frac": 0.005,
        "target_frac": 0.03,
        "max_bars": 5,
        "costs": Costs(0.02, 0.0005, 0.02),
    },
    "1%/1%, max_bars=1": {"n": 4000, "max_bars": 1},
    "1%/1%, gaps (gap vol 0.3%)": {"n": 4000, "gap_vol": 0.003},
    "0.5% stop / 3% target, gaps (gap vol 1%)": {
        "n": 4000,
        "stop_frac": 0.005,
        "target_frac": 0.03,
        "gap_vol": 0.01,
    },
    "0.3% stop / 5% target, about 19 trades": {
        "n": 400,
        "stop_frac": 0.003,
        "target_frac": 0.05,
    },
    "0.5% stop / 3% target, about 38 trades": {
        "n": 800,
        "stop_frac": 0.005,
        "target_frac": 0.03,
    },
}


def baseline_section(scale: float) -> None:
    r = reps(1000, scale)
    for name, kw in BASELINE_SCENARIOS.items():
        res_p, all_p, warned, n_tr = [], [], [], []
        for seed in range(r):
            _, bt = _random_strategy(seed, **kw)  # type: ignore[arg-type]
            try:
                out = no_skill_test(bt)
            except ValueError:
                continue
            res_p.append(out.p_value)
            all_p.append(out.all_exits_p_value)
            warned.append(bool(out.warnings))
            n_tr.append(out.all_exits_n)
        res_hits = [p <= ALPHA for p in res_p]
        silent = [h and not w for h, w in zip(res_hits, warned, strict=True)]
        note = f"about {np.mean(n_tr):.0f} trades; rejected without a warning {np.mean(silent):.4f}"
        row("no_skill_test resolved-only null", name, res_hits, note)
        row("no_skill_test all-exits null", name, [p <= ALPHA for p in all_p])
    r = reps(300, scale)
    for edge in (0.05, 0.1, 0.2, 0.3):
        res_p, all_p = [], []
        for seed in range(r):
            _, bt = _planted_strategy(seed, edge)
            out = no_skill_test(bt)
            res_p.append(out.p_value)
            all_p.append(out.all_exits_p_value)
        note = "1%/1% brackets, max_bars=1, about 300 trades"
        row(
            "no_skill_test resolved-only power",
            f"planted edge {edge}",
            [p <= ALPHA for p in res_p],
            note,
        )
        row(
            "no_skill_test all-exits power",
            f"planted edge {edge}",
            [p <= ALPHA for p in all_p],
            note,
        )


SECTIONS: dict[str, Callable[[float], None]] = {
    "psr": psr_section,
    "dsr": dsr_section,
    "newey_west": newey_west_section,
    "adjust": adjust_section,
    "pbo": pbo_section,
    "random_entry": random_entry_section,
    "baseline": baseline_section,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every rep count")
    parser.add_argument("--only", nargs="*", choices=sorted(SECTIONS), help="sections to run")
    args = parser.parse_args()
    print(f"alpha = {ALPHA}; rate = share of replications rejecting (or the mean, where noted)")
    print()
    print("| test | scenario | reps | rate | MC SE | note |")
    print("|---|---|---|---|---|---|")
    start = time.perf_counter()
    for name in args.only or list(SECTIONS):
        SECTIONS[name](args.scale)
    print()
    print(f"total time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
