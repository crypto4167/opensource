# backtest-skeptic

backtest-skeptic checks whether a trading backtest's edge is real. It tests the signal code for lookahead by calling it on truncated and scrambled data, compares a strategy's trades with random entries that share its brackets and costs, computes the win rate a bracket gives with no skill and the win rate needed to break even after costs, audits every fill in a trade log against the bars, and implements the standard overfitting statistics (deflated Sharpe ratio, probability of backtest overfitting, Newey-West t-statistics). It is for people who write their own strategies and want to find out, before trading them, whether the result comes from the entries or from a bug, a bracket shape, a cost assumption or luck. The only runtime dependency is numpy.

## Install

Once published to PyPI:

```bash
pip install backtest-skeptic
```

From source:

```bash
git clone https://github.com/crypto4167/opensource
cd opensource
pip install -e ".[dev]"
```

Python 3.10 or later.

## Quickstart

Build bars, run a strategy, audit it. This uses a synthetic random walk, so the strategy should show no edge.

```python
import numpy as np

from backtest_skeptic import (
    Costs,
    audit_fills,
    no_skill_test,
    random_entry_test,
    run_signals,
    summarize,
    truncation_test,
)
from backtest_skeptic.synthetic import random_walk_bars

# 1. Bars. Use Bars.from_csv("prices.csv") for real data.
bars = random_walk_bars(5_000, seed=1)


# 2. A signal: +1 after three higher closes, -1 after three lower closes.
def signal(bars):
    d = np.sign(np.diff(bars.close, prepend=bars.close[0]))
    out = np.zeros(len(d))
    same = (d[2:] == d[1:-1]) & (d[1:-1] == d[:-2])
    out[2:] = np.where(same, d[2:], 0.0)
    return out


# 3. Run it: decide at the close of bar i, fill at the open of bar i + 1.
side = signal(bars)
stop = bars.close * (1 - side * 0.01)
target = bars.close * (1 + side * 0.01)
costs = Costs(half_spread=0.01, fee_rate=0.0001)
result = run_signals(bars, side, stop, target, costs=costs)
print(summarize(result.trades))

# 4. Audit it.
print(truncation_test(signal, bars))  # does the signal read future bars?
print(no_skill_test(result))  # does it beat the bracket's free win rate, and costs?
print(random_entry_test(bars, result))  # does it beat random entries with the same brackets?
print(audit_fills(bars, result))  # are the fills possible on these bars?
```

Output (about one second):

```text
1009 trades, win rate 49.0%, expectancy -0.088R, total -89.02R, t -2.87, PF 0.83, max DD 93.68R
truncation test PASSED: 40 prefix(es), 40 values compared, nothing changed when later bars were removed.
No-skill baseline over 1008 resolved trades (excluded: 0 time, 1 end, 0 invalid geometry)
  win rate 49.0% vs no-skill 50.0%: skill edge -1.0 pts (z -0.63, one-sided p 0.7458, exact)
  break-even win rate 53.6%: cost drag +3.6 pts, margin over break-even -4.5 pts
  a no-skill trader with these brackets expects -0.0690R per trade; these trades averaged -0.0883R
  all exits (1009 trades, time/end scored by position): edge -1.0 pts (z -0.63, p 0.7355)
Random-entry test (expectancy, 1009 trades): observed -0.0882 vs null -0.0749 (sd 0.0304) over 1000 simulations; p = 0.6893, percentile 31.1, z -0.44
Fill audit (heuristic) of 1009 trades: PASSED (no fails)
  ambiguous exit bars: 6 (0 booked as target)
  expectancy over 1009 rescored trades: booked -0.0882R, worst case -0.0882R, best case -0.0767R
```

R is the result in units of the trade's initial risk, after costs. The win rate matches the 50% a symmetric bracket gives with no skill, random entries with the same brackets do as well, and the costs make the expectancy negative. The engine's fill rules (next-bar-open entries, gapped stops filled at the open, stop first on bars that touch both levels) are in [docs/fill-rules.md](docs/fill-rules.md).

The [examples](examples/) show each check catching a real problem: leaky signal functions, a random strategy with an 84% win rate, and a backtester with four fill bugs.

## What each check catches

Full inputs, outputs, thresholds and limitations are in [docs/checks.md](docs/checks.md).

| Check | Catches | Source |
| --- | --- | --- |
| [`truncation_test`, `perturbation_test`](docs/checks.md#lookahead) | Signal code that reads future bars: centred windows, `shift(-1)`, backward fills, full-sample normalisation, models fitted on all the data | heuristic |
| [`check_factory`](docs/checks.md#lookahead) | A strategy whose `decide()` reads an indicator precomputed on the full data | heuristic |
| [`random_entry_test`](docs/checks.md#random-entry-nulls) | Entry signals that do no better than random entries with the same trade count, sides, brackets, costs and exits | Phipson & Smyth (2010) for the p-value; matching is heuristic |
| [`side_test`](docs/checks.md#random-entry-nulls) | Direction logic that does no better than a coin at the same entry bars | as above |
| [`no_skill_test`](docs/checks.md#no-skill-baseline) | A win rate that comes from the bracket shape, and one that beats chance but not costs | Feller, vol. 1, ch. XIV |
| [`audit_fills`](docs/checks.md#fill-realism) | Fills the bars cannot support: signal-bar fills, gapped stops filled at the stop, missed earlier touches, optimistic same-bar outcomes, prices outside the bar | heuristic |
| [`concentration`, `drop_best`, `subperiods`](docs/checks.md#robustness) | A result carried by one trade, one day or one period | heuristic |
| [`cost_stress`, `entry_delay_test`](docs/checks.md#robustness) | An edge that disappears with slightly higher costs or a one-bar later fill | heuristic |
| [`block_bootstrap`, `clustered_t`](docs/checks.md#robustness) | Confidence intervals and t-statistics too narrow for dependent trades | Kunsch (1989), Politis & Romano (1994), Cameron & Miller (2015) |
| [`deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`](docs/checks.md#return-series-statistics) | A Sharpe ratio explained by the number of trials, a short sample or fat tails | Bailey & Lopez de Prado (2012, 2014) |
| [`pbo`](docs/checks.md#return-series-statistics) | A parameter search whose in-sample winner is no better than average out of sample | Bailey, Borwein, Lopez de Prado & Zhu (2017) |
| [`newey_west`, `lo_annualized_sharpe`](docs/checks.md#return-series-statistics) | t-statistics and annualised Sharpe ratios inflated by autocorrelation | Newey & West (1987, 1994), Hansen & Hodrick (1980), Lo (2002) |
| [`adjust_pvalues`, `haircut_sharpe`](docs/checks.md#return-series-statistics) | Significance claimed for the best of many tests | Holm (1979), Benjamini & Hochberg (1995), Benjamini & Yekutieli (2001), Harvey & Liu (2015) |
| [`HoldoutLedger`, `TrialRegistry`, `split_holdout`](docs/checks.md#holdout-bookkeeping) | A holdout read more than once, an unrecorded trial count, train and holdout sharing bars | Dwork et al. (2015), Bailey & Lopez de Prado (2014) |
| [`spanning_regression`, `proxy_replication`](docs/checks.md#factor-spanning) | An "edge" that is exposure to a factor you could have held | Huberman & Kandel (1987) |

## Command-line interface

```text
backtest-skeptic audit --bars BARS.csv --trades TRADES.csv
    [--half-spread X] [--fee-rate X] [--slippage X] [--max-bars N]
    [--same-bar stop|target] [--sims N] [--seed N] [--periods-per-year P]
    [--format text|markdown|json] [--fail-on fail|warn|never] [--time-column NAME]

backtest-skeptic lookahead MODULE:FUNCTION --bars BARS.csv
    [--mode truncation|perturbation|both] [--min-history N]

backtest-skeptic dsr --sharpe S --n-obs T --trials N --trials-variance V
    [--skew G] [--kurtosis K] [--periods-per-year P --annualized] [--format text|json]

backtest-skeptic selftest
```

- `audit` runs every check the inputs support on a bars CSV and a trade-log CSV (columns are described in [docs/faq.md](docs/faq.md#can-i-use-this-with-vectorbt-backtestingpy-or-my-own-engine)): fill realism, the no-skill baseline, the random-entry null, concentration, stability and Sharpe statistics. Pass the costs and engine settings the backtest used. Exit code 0 means no check reached the `--fail-on` level, 1 means one did, 2 means a usage or data error, so it can gate a CI job.
- `lookahead` imports `FUNCTION` from a module or a `.py` file, calls it on the bars and runs the truncation test, the perturbation test, or both. It exits 1 when it finds a leak.
- `dsr` computes the probabilistic and deflated Sharpe ratios, the expected maximum Sharpe ratio of the trials and the minimum track record length from summary numbers. Kurtosis is raw (3 for a normal distribution).
- `selftest` runs the built-in positive and negative controls and exits 1 if any is judged wrongly.

`python -m backtest_skeptic` works the same way. The Python equivalent of `audit` is `backtest_skeptic.report.audit(bars, trades, costs=...)`, which returns a report with `to_text()`, `to_markdown()` and `to_json()`.

The quickstart strategy above, written to CSV and audited:

```text
$ backtest-skeptic audit --bars bars.csv --trades trades.csv --half-spread 0.01 --fee-rate 0.0001
Backtest audit: 1009 trades on 5000 bars (trade log)
Worst status: WARN

[PASS] Fill realism       Every fill in 1009 trades is consistent with the bars. Replayed from the bars, expectancy lies between -0.088R and -0.077R per trade; the log books -0.088R.
[WARN] No-skill baseline  0.1% of trades exited on time or at the end of the data, so every exit is scored by how far the price got toward the target: edge -1.0 pts over no skill across 1009 trades (p = 0.736). On the 1008 trades that hit a level, the win rate 49.0% is -4.5 pts from the break-even rate 53.6%. Not distinguishable from random entries with the same brackets.
[WARN] Random-entry null  Expectancy -0.088R per trade against -0.075R for random entries with the same sides, brackets and costs (1000 simulations, p = 0.689). Not distinguishable from random entries.
[SKIP] Concentration      Skipped. The total is -89.025R, so there is no profit for a few days or trades to carry.
[WARN] Stability          0 of 3 quarters (0.0%) were positive; the 95% block-bootstrap interval for the mean per trade is [-0.141R, -0.028R]. Too few quarters made money.
[WARN] Sharpe statistics  Sharpe ratio -0.090 per trade over 1009 trades (skew 0.04, raw kurtosis 1.01): PSR against zero is 0.002 and the minimum track record for 95% confidence is infinite. PSR is below 0.95 and the Sharpe ratio is not above zero.
```

The full output also explains why the deflated Sharpe ratio is not computed: it needs the number of configurations tried, which a trade log does not record.

## How it differs from existing tools

Return-series statistics such as the deflated Sharpe ratio and the probability of backtest overfitting are implemented elsewhere. This package implements them too, and adds checks that work on the strategy's code, trades and fills:

- behavioural lookahead tests for any Python function that computes a signal,
- matched random-entry nulls that keep a strategy's trade count, sides, bracket geometry, costs and exit rules,
- no-skill and break-even win rates for bracket trades, with an exact test,
- a fill-realism audit for a trade log from any backtester.

pypbo and backtesting.py are licensed under the AGPL, and mlfinlab is no longer open source. backtest-skeptic is a clean-room implementation from the published formulas, under the MIT license.

## Limitations

- **Bars are mid prices, and OHLC data cannot show the order of prices inside a bar.** The engine and the fill audit resolve that doubt against the trader; they do not remove it. Tick data would settle it.
- **The engine is simple.** One position at a time, fixed stop and target levels, a fixed risk of 1R per trade, no compounding, no partial fills, no trailing stops. Volume, liquidity and market impact are not modelled.
- **Lookahead tests are one-sided.** A failure proves a leak. A pass covers only the prefixes and cuts that were probed, and a leak through volume or timestamps alone is not seen.
- **The no-skill baseline assumes a driftless price.** On a trending market, compare with random entries instead.
- **Random-entry nulls test timing only.** Edge from a long bias in a rising market, or from the bracket shape, is shared with the null. A small p-value means the timing adds something. Whether the strategy makes money after costs is a separate reading of the expectancy.
- **Several checks are heuristics** with no significance level (lookahead, fill audit, concentration, sub-periods, cost stress, entry delay). Their thresholds are stated in [docs/checks.md](docs/checks.md).
- **The overfitting statistics need honest inputs.** The deflated Sharpe ratio needs the number of trials you ran, and PBO needs the returns of every configuration you tried. `TrialRegistry` records them, but only if you use it from the start.
- A random strategy still passes a test at the 5% level about 5% of the time. Correct for the number of strategies you tested.
- **Measured error rates are in [docs/calibration.md](docs/calibration.md).** Most tests reject a true null at close to their nominal 5%. The exceptions are stated there: the probabilistic Sharpe ratio rejects 7% to 8% of the time with 60 fat-tailed or skewed returns, and Newey-West t-statistics with the default lags reject 10% to 13% of the time on AR(1) series with coefficient 0.6 and 250 to 1000 points.

## Citing

If you use a check that implements a published method, cite the paper. The references, as given in the module docstrings:

- Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS* 61(5).
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. *Journal of the Royal Statistical Society B* 57(1).
- Benjamini, Y. & Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple Testing under Dependency. *Annals of Statistics* 29(4).
- Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources* 50(2).
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. & Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. *Science* 349(6248).
- Feller, W. *An Introduction to Probability Theory and Its Applications*, vol. 1, ch. XIV (random walks and ruin problems).
- Hansen, L. P. & Hodrick, R. J. (1980). Forward Exchange Rates as Optimal Predictors of Future Spot Rates. *Journal of Political Economy* 88(5).
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1).
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2).
- Huberman, G. & Kandel, S. (1987). Mean-Variance Spanning. *Journal of Finance* 42(4).
- Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3).
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4).
- Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3).
- Newey, W. K. & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies* 61(4).
- Phipson, B. & Smyth, G. K. (2010). Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1).
- Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New checks need a test where they must report nothing and a test where they must catch the problem, and statistical methods need a citation.

## License

MIT. See [LICENSE](LICENSE).
