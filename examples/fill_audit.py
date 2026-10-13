"""Audit the fills of a naive backtester, and of the engine, on the same signals.

The naive backtester below has four fill bugs that are common in hand-written
and popular backtesting code:

1. It enters at the close of the bar that produced the signal.
2. It fills a stop at the stop price even when the bar opened through it.
3. When one bar touches both the stop and the target, it books the target.
4. Its exit loop stops one bar early, so the last bar is never checked.

fills.audit_fills reads its trade log next to the bars and flags each bug.
The engine's own trades for the same signals go through the same audit.

Run with:  python examples/fill_audit.py
"""

from __future__ import annotations

import numpy as np

from backtest_skeptic import Costs, TradeLog, audit_fills, run_signals
from backtest_skeptic.fills import CODES
from backtest_skeptic.synthetic import random_walk_bars

N_BARS = 5_000
STOP_FRAC = 0.01
TARGET_FRAC = 0.01
COSTS = Costs(half_spread=0.01, slippage=0.005)
SEED = 4


def naive_backtest(bars, side, stop, target, costs):
    """A backtester with four fill bugs. Returns a list of trade records."""
    h, lo, c = bars.high, bars.low, bars.close
    n = len(bars)
    hs, slip = costs.half_spread, costs.slippage
    records = []
    i = 0
    while i < n - 1:
        s = int(side[i])
        if s == 0:
            i += 1
            continue
        entry = c[i] + s * (hs + slip)  # bug 1: fills at the signal bar's close
        exit_index, exit_price, reason = n - 1, c[n - 1] - s * (hs + slip), "end"
        for j in range(i + 1, n - 1):  # bug 4: never looks at bar n - 1
            hit_stop = lo[j] <= stop[i] if s > 0 else h[j] >= stop[i]
            hit_target = h[j] >= target[i] if s > 0 else lo[j] <= target[i]
            if hit_target:  # bug 3: target first when a bar touches both
                exit_index, exit_price, reason = j, target[i] - s * hs, "target"
                break
            if hit_stop:  # bug 2: the stop price, even after a gap through it
                exit_index, exit_price, reason = j, stop[i] - s * (hs + slip), "stop"
                break
        records.append(
            {
                "side": s,
                "signal_index": i,
                "entry_index": i,
                "exit_index": exit_index,
                "entry_price": entry,
                "exit_price": exit_price,
                "stop": stop[i],
                "target": target[i],
                "reason": reason,
            }
        )
        i = exit_index + 1
    return records


def last_bar_setup(bars):
    """Side, stop and target for a signal at the second-to-last bar.

    Bug 4 only shows when a trade is still open on the last bar, so the example
    adds one signal whose target the last bar trades through.
    """
    o, h, lo, c = bars.open[-1], bars.high[-1], bars.low[-1], bars.close[-2]
    if h - max(o, c) >= min(o, c) - lo:
        side = 1.0
        target = (max(o, c) + h) / 2
    else:
        side = -1.0
        target = (min(o, c) + lo) / 2
    stop = c * (1 - side * STOP_FRAC)
    return side, stop, target


def main() -> None:
    bars = random_walk_bars(N_BARS, vol=0.01, gap_vol=0.005, seed=SEED)
    rng = np.random.default_rng(SEED)
    fires = rng.random(N_BARS) < 0.1
    side = np.where(fires, rng.choice([-1.0, 1.0], size=N_BARS), 0.0)
    side[-50:] = 0.0
    stop = bars.close * (1 - side * STOP_FRAC)
    target = bars.close * (1 + side * TARGET_FRAC)
    side[-2], stop[-2], target[-2] = last_bar_setup(bars)

    naive_log = TradeLog.from_records(naive_backtest(bars, side, stop, target, COSTS), bars)
    naive = audit_fills(bars, naive_log, costs=COSTS)
    engine = audit_fills(bars, run_signals(bars, side, stop, target, costs=COSTS))

    print(f"{N_BARS} gappy random-walk bars, the same signals and costs for both backtesters\n")
    print(f"{'':<42} {'naive':>10} {'engine':>10}")
    print(f"{'trades':<42} {naive.n_trades:>10} {engine.n_trades:>10}")
    for code in CODES:
        if code in naive.counts or code in engine.counts:
            label = f"{code} ({CODES[code][0]})"
            print(f"{label:<42} {naive.counts.get(code, 0):>10} {engine.counts.get(code, 0):>10}")
    print(f"{'trades with a fail':<42} {naive.trades_with_fail:>10} {engine.trades_with_fail:>10}")
    for name in ("booked", "worst", "best"):
        a = getattr(naive, f"{name}_expectancy_r")
        b = getattr(engine, f"{name}_expectancy_r")
        print(f"{name + ' expectancy (R)':<42} {a:>+10.4f} {b:>+10.4f}")

    if naive.booked_expectancy_r > naive.best_expectancy_r:
        print(
            "\nThe naive log books more than the best-case replay of its own trades: its"
            "\nfills are better than the bars can support."
        )
    if engine.booked_expectancy_r == engine.worst_expectancy_r:
        print("The engine's booked expectancy equals its worst case: where the bars cannot")
        print("decide, it books the outcome worse for the trader.")

    print("\nOne example of each issue in the naive log:")
    seen = set()
    for issue in naive.issues:
        if issue.code not in seen:
            seen.add(issue.code)
            print(f"  {issue}")
    print("\nThe late trade that was still open on the last bar (bug 4):")
    for issue in naive.issues_for(naive.n_trades - 1):
        print(f"  {issue}")


if __name__ == "__main__":
    main()
