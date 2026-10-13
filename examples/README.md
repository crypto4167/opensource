# Examples

Three scripts, each showing a check catching a real problem on synthetic data. They need only numpy and the package (`pip install -e .` from the repository root), take under three seconds each, and print a short result. CI runs all three.

```bash
python examples/lookahead_bugs.py
python examples/random_strategy_audit.py
python examples/fill_audit.py
```

The numbers below are from running each script as committed. Every script fixes its random seeds, so the output is reproducible.

## lookahead_bugs.py

Three signal functions with leaks that show up in real code, each next to its honest version:

- a mean-reversion signal built on a centred rolling mean (bar `i` averages bars `i - 10` to `i + 9`),
- a z-score normalised with the mean and standard deviation of the whole series,
- a momentum signal that compares the next bar's close with this one (a `shift(-1)`).

It runs `truncation_test` (with `check="prefix"`) and `perturbation_test` on each, then `check_factory` on two strategy factories whose `decide()` reads an indicator through a closure: one precomputed as a centred mean on the full data, one as a trailing mean.

```text
signal                 version  truncation                 perturbation
centred rolling mean   leaky    LEAK (earliest bar 9)      LEAK (earliest bar 240)
centred rolling mean   honest   pass                       pass
full-sample z-score    leaky    LEAK (earliest bar 0)      LEAK (earliest bar 0)
full-sample z-score    honest   pass                       pass
next-bar shift         leaky    LEAK (earliest bar 0)      LEAK (earliest bar 249)
next-bar shift         honest   pass                       pass

Strategy factories (check_factory)
  precomputed centred mean     LEAK (earliest bar 250)
  precomputed trailing mean    pass
```

Bar 9 is the first bar where the centred mean exists, so it is the first leaking bar. The perturbation test only compares rows up to each cut, so its earliest bar depends on where the cuts fell.

## random_strategy_audit.py

A strategy that enters at random bars in a random direction, with a target five times closer than the stop, on a driftless random walk with spread, slippage and fees. Its win rate looks good. The same brackets and costs are then used to trade the planted signal from `synthetic.planted_signal_bars` (the bar after each signal drifts its way by half a bar's standard deviation).

```text
                             random entries   planted signal
trades                                  968              973
win rate                              84.0%            91.0%
profit factor                          0.80             1.50
expectancy (R, net)                 -0.0336          +0.0461
no-skill win rate                     83.3%            83.3%
break-even win rate                   86.8%            87.1%
skill edge (pts)                       +0.7             +7.6
margin after costs (pts)               -2.8             +3.9
no-skill test p                        0.31         4.61e-12
random-entry null (R)               -0.0491          -0.0526
random-entry test p                  0.1339           0.0010
```

The random strategy's 84.0% win rate is the 83.3% its brackets give with no skill, and it needs 86.8% to break even. Random entries with the same brackets do about as well (p = 0.13). The planted signal clears both baselines; 0.0010 is the smallest p-value 1000 simulations can give.

## fill_audit.py

A deliberately naive backtester with four fill bugs: it enters at the signal bar's close, fills gapped stops at the stop price, books the target when a bar touches both levels, and never checks the last bar. It runs on random-walk bars with gaps between bars. `audit_fills` audits its trade log and the engine's trades for the same signals and costs. One late signal is placed so that its trade is still open on the last bar.

```text
                                                naive     engine
trades                                            461        477
filled_at_signal_close (warn)                     461          0
phantom_exit_fill (fail)                            4          0
gap_stop_fill_too_good (fail)                      24          0
missed_earlier_touch (fail)                         2          0
ambiguous_resolved_optimistically (warn)           10          0
trades with a fail                                 26          0
booked expectancy (R)                         +0.0162    -0.0475
worst expectancy (R)                          -0.0437    -0.0475
best expectancy (R)                           -0.0007    +0.0340
```

Each bug has its own code: signal-close entries (`filled_at_signal_close`), stops filled at the stop price after a gap (`gap_stop_fill_too_good`, and `phantom_exit_fill` when the stop price lies outside the bar entirely), targets booked on bars that touched both levels (`ambiguous_resolved_optimistically`, or `missed_earlier_touch` when the bar opened through the stop), and the unchecked last bar (`missed_earlier_touch` on bar 4999). The naive log books +0.0162R per trade, above the best case the bars allow for its own trades (-0.0007R). The engine's booked figure equals its worst case.
