"""An 84% win rate from random entries, and how the audit tells it apart from a real edge.

The strategy enters at random bars in a random direction. Its target is five
times closer than its stop, so it wins most of the time with no skill at all.
On its face the win rate looks excellent. The no-skill baseline and the
matched random-entry test show where that win rate comes from.

The same audit is then run on bars with a planted, genuine edge, trading the
planted signal with the same brackets and costs. There the edge is detected.

Run with:  python examples/random_strategy_audit.py
"""

from __future__ import annotations

import numpy as np

from backtest_skeptic import Costs, no_skill_test, random_entry_test, run_signals, summarize
from backtest_skeptic.synthetic import planted_signal_bars, random_walk_bars

N_BARS = 20_000
VOL = 0.002  # standard deviation of each bar's log return
STOP_FRAC = 0.004  # stop 0.4% from the signal bar's close
TARGET_FRAC = 0.0008  # target 0.08% away: five times closer than the stop
COSTS = Costs(half_spread=0.005, fee_rate=0.00002, slippage=0.002)
SEED = 0


def run(bars, side):
    """Bracket every signal at fixed fractions of the signal bar's close."""
    close = bars.close
    stop = close * (1 - side * STOP_FRAC)
    target = close * (1 + side * TARGET_FRAC)
    return run_signals(bars, side, stop, target, costs=COSTS)


def audit(bars, result):
    s = summarize(result.trades)
    base = no_skill_test(result)
    null = random_entry_test(bars, result, n_sims=1000, seed=SEED)
    return {
        "trades": f"{s.trades}",
        "win rate": f"{s.win_rate:.1%}",
        "profit factor": f"{s.profit_factor:.2f}",
        "expectancy (R, net)": f"{s.expectancy_r:+.4f}",
        "no-skill win rate": f"{base.no_skill_rate:.1%}",
        "break-even win rate": f"{base.breakeven_rate:.1%}",
        "skill edge (pts)": f"{100 * base.skill_edge:+.1f}",
        "margin after costs (pts)": f"{100 * base.margin:+.1f}",
        "no-skill test p": f"{base.p_value:.3g}",
        "random-entry null (R)": f"{null.null_mean:+.4f}",
        "random-entry test p": f"{null.p_value:.4f}",
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Random entries on a driftless random walk: there is nothing to find.
    bars = random_walk_bars(N_BARS, vol=VOL, seed=SEED)
    fires = rng.random(N_BARS) < 0.05
    side = np.where(fires, rng.choice([-1.0, 1.0], size=N_BARS), 0.0)
    random_result = run(bars, side)

    # A planted edge: after each signal the next bar drifts its way by 0.5 sigma.
    planted, signal = planted_signal_bars(N_BARS, vol=VOL, edge=0.5, signal_prob=0.05, seed=SEED)
    planted_result = run(planted, signal.astype(np.float64))

    before = audit(bars, random_result)
    after = audit(planted, planted_result)

    print(
        f"Both runs: {N_BARS} bars, stop {STOP_FRAC:.2%} and target {TARGET_FRAC:.2%} from the"
        f"\nsignal close, half-spread {COSTS.half_spread}, slippage {COSTS.slippage}, fee rate"
        f" {COSTS.fee_rate} per side.\n"
    )
    print(f"{'':<26} {'random entries':>16} {'planted signal':>16}")
    for key in before:
        print(f"{key:<26} {before[key]:>16} {after[key]:>16}")

    print(
        "\nThe random strategy's win rate matches the no-skill rate a / (a + b) for its"
        "\nbrackets, sits below the break-even rate after costs, and random entries with"
        "\nthe same brackets do about as well. The planted signal beats both baselines."
    )


if __name__ == "__main__":
    main()
