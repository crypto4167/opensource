"""Signal functions that leak the future, their honest versions, and the tests that catch them.

Each leaky function below is the kind of code that passes review: it runs, it
returns one value per bar, and a backtest built on it looks good. Each one
reads bars it could not have seen when the decision was made.

The lookahead tests never read the code. They call each function on prefixes
of the data (truncation_test) and on copies whose future has been scrambled
(perturbation_test), and report any output that changed.

Run with:  python examples/lookahead_bugs.py
"""

from __future__ import annotations

import numpy as np

from backtest_skeptic import Bars, check_factory, perturbation_test, truncation_test
from backtest_skeptic.synthetic import random_walk_bars

WINDOW = 20


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of the last ``window`` values (NaN until enough exist)."""
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        cs = np.cumsum(np.concatenate([[0.0], x]))
        out[window - 1 :] = (cs[window:] - cs[:-window]) / window
    return out


# --- Leak 1: a centred rolling mean ------------------------------------------
# pandas' rolling(window, center=True) and scipy's smoothing filters centre the
# window by default in some setups. Bar i then averages bars i-10 .. i+9.


def mean_reversion_centred(bars: Bars) -> np.ndarray:
    close = bars.close
    half = WINDOW // 2
    centred = np.full(len(close), np.nan)
    trailing = rolling_mean(close, WINDOW)
    centred[: len(close) - half] = trailing[half:]
    return np.sign(centred - close)


def mean_reversion_honest(bars: Bars) -> np.ndarray:
    close = bars.close
    return np.sign(rolling_mean(close, WINDOW) - close)


# --- Leak 2: a z-score with the full-sample mean and standard deviation -------
# Standardising a feature with statistics of the whole series is routine in
# machine-learning pipelines. Every bar then knows where the price ends up.


def zscore_full_sample(bars: Bars) -> np.ndarray:
    close = bars.close
    sd = close.std()
    if sd == 0:
        return np.zeros(len(close))
    z = (close - close.mean()) / sd
    return np.where(z < -1.0, 1.0, np.where(z > 1.0, -1.0, 0.0))


def zscore_expanding(bars: Bars) -> np.ndarray:
    close = bars.close
    n = np.arange(1, len(close) + 1)
    mean = np.cumsum(close) / n
    var = np.cumsum(close**2) / n - mean**2
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (close - mean) / sd, 0.0)
    z[: WINDOW - 1] = 0.0
    return np.where(z < -1.0, 1.0, np.where(z > 1.0, -1.0, 0.0))


# --- Leak 3: a "next bar" shift -------------------------------------------------
# The author wanted "the return that this signal will earn" for a label, wrote
# shift(-1), and the label ended up as a feature.


def momentum_next_bar(bars: Bars) -> np.ndarray:
    close = bars.close
    nxt = np.full(len(close), np.nan)
    nxt[:-1] = close[1:]
    return np.sign(nxt - close)


def momentum_honest(bars: Bars) -> np.ndarray:
    close = bars.close
    prev = np.full(len(close), np.nan)
    prev[1:] = close[:-1]
    return np.sign(close - prev)


# --- Leak 4: a strategy factory that precomputes on the full data --------------
# run() only passes decide() the bars up to i, but a closure can still read an
# array the factory built from every bar.


def leaky_factory(bars: Bars):
    close = bars.close
    half = WINDOW // 2
    centred = np.full(len(close), np.nan)
    centred[: len(close) - half] = rolling_mean(close, WINDOW)[half:]

    def decide(i: int, history: Bars):
        c = float(history.close[-1])
        if np.isnan(centred[i]) or c >= centred[i]:
            return None
        return (1, c * 0.99, c * 1.01)

    return decide


def honest_factory(bars: Bars):
    trailing = rolling_mean(bars.close, WINDOW)

    def decide(i: int, history: Bars):
        c = float(history.close[-1])
        if np.isnan(trailing[i]) or c >= trailing[i]:
            return None
        return (1, c * 0.99, c * 1.01)

    return decide


def verdict(report) -> str:
    if report.leaks:
        return f"LEAK (earliest bar {report.first_bad_index})"
    return "pass" if report.passed else "inconclusive"


def main() -> None:
    bars = random_walk_bars(500, seed=7)
    pairs = [
        ("centred rolling mean", mean_reversion_centred, mean_reversion_honest),
        ("full-sample z-score", zscore_full_sample, zscore_expanding),
        ("next-bar shift", momentum_next_bar, momentum_honest),
    ]
    print(f"Signal functions on {len(bars)} random-walk bars\n")
    print(f"{'signal':<22} {'version':<8} {'truncation':<26} {'perturbation':<26}")
    for name, leaky, honest in pairs:
        for label, fn in (("leaky", leaky), ("honest", honest)):
            trunc = truncation_test(fn, bars, check="prefix")
            pert = perturbation_test(fn, bars)
            print(f"{name:<22} {label:<8} {verdict(trunc):<26} {verdict(pert):<26}")

    print("\nStrategy factories (check_factory)")
    for label, factory in (
        ("precomputed centred mean", leaky_factory),
        ("precomputed trailing mean", honest_factory),
    ):
        report = check_factory(factory, bars, warmup=WINDOW)
        print(f"  {label:<28} {verdict(report)}")

    print("\nWhat check_factory reports for the leaky factory:")
    lines = str(check_factory(leaky_factory, bars, warmup=WINDOW)).splitlines()
    print("\n".join(lines[:2]))
    print(
        "\nTruncation with check='prefix' compares every earlier row, so its earliest bar is"
        "\nexact. Perturbation only sees rows near each cut, so its earliest bar is"
        "\nearliest one it happened to probe."
    )


if __name__ == "__main__":
    main()
