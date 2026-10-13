# Checks reference

Every public check in `backtest_skeptic`, grouped by module. For each one: what it catches, its inputs and their units, what the output means, its default thresholds, its limitations, and its source. A check with no published source is labelled **heuristic**. Citations are copied from the module docstrings.

Conventions that apply throughout:

- **Bars** are mid prices. Stops and targets trigger on bar prices; the spread is charged through `Costs`.
- **R** is a trade's net result in units of its initial risk, `side * (entry_price - stop)`, after every cost.
- **Costs** (`Costs(half_spread, fee_rate, slippage)`): `half_spread` and `slippage` are in price units, `fee_rate` is a fraction of traded notional per side (`0.0005` is 5 basis points). Checks that take costs must be given the costs the backtest used.
- **Sharpe ratios** in `stats` are per period (the sampling frequency of the returns) unless a parameter name says `annual`.
- **Kurtosis** is raw kurtosis (3 for a normal distribution), never excess kurtosis.

Contents:

- [Lookahead](#lookahead): `truncation_test`, `perturbation_test`, `check_factory`, `self_test`
- [Random-entry nulls](#random-entry-nulls): `random_entry_test`, `side_test`, `positive_control`
- [No-skill baseline](#no-skill-baseline): `no_skill_test`, `expectancy_decomposition`, `bracket_odds`, `breakeven_win_rate`
- [Fill realism](#fill-realism): `audit_fills`
- [Robustness](#robustness): `concentration`, `drop_best`, `subperiods`, `block_bootstrap`, `clustered_t`, `cost_stress`, `entry_delay_test`
- [Return-series statistics](#return-series-statistics): `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `min_track_record_length`, `min_backtest_length`, `pbo`, `newey_west`, `lo_annualized_sharpe`, `adjust_pvalues`, `haircut_sharpe`
- [Holdout bookkeeping](#holdout-bookkeeping): `split_holdout`, `HoldoutLedger`, `TrialRegistry`
- [Factor spanning](#factor-spanning): `spanning_regression`, `proxy_replication`

## Lookahead

Module `backtest_skeptic.lookahead`. **Heuristic** (no canonical citation).

The tests treat the code under test as a black box and only call it. A value computed for bar `i` may use bars `0..i` and nothing later. The tests are one-sided: a failure is hard evidence of a leak (an output demonstrably changed when later bars changed), while a pass only covers the probes that ran.

All three return a `LookaheadReport` with `passed`, `leaks`, `n_checked`, `n_values`, `n_mismatches`, `mismatches` (up to 50, earliest row first), `errors`, `first_bad_index`, `checkpoints` and `notes`. `passed` is true only when at least one probe ran, nothing changed and the function never raised. A report with nothing checked does not pass.

### `truncation_test(fn, bars, ...)`

**Catches:** centred windows, `shift(-k)`, backward fills, whole-series normalisation, models fitted on all the data, windows that run one bar too far.

**How:** recomputes `fn(bars.prefix(k))` for a set of prefix lengths `k` and compares with `fn(bars)`. A causal function returns the same value for bar `k - 1` whether or not later bars exist.

**Inputs:** `fn(bars)` returns an array-like with one row per bar (1-D, or 2-D for several columns; pandas objects work). Non-numeric outputs such as `"BUY"`/`"SELL"` are compared by exact equality.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `checkpoints` | `None` | Prefix lengths to test. By default `n_checkpoints` are chosen: half evenly spaced, a quarter just after rows where the output is non-zero, the rest at random. |
| `n_checkpoints` | `40` | Number of prefixes when `checkpoints` is not given. Every prefix is tested when there are no more than this many. |
| `min_history` | `1` | Shortest prefix `fn` must handle. An exception below it skips that prefix; at or above it the test fails. |
| `check` | `"last"` | `"last"` compares row `k - 1` only. `"prefix"` compares every row below `k`, which also catches repainting and pins down the earliest leaking row exactly. |
| `rtol`, `atol` | `1e-9`, `0.0` | Values match when `abs(a - b) <= atol + rtol * max(abs(a), abs(b))`. NaN matches NaN; NaN against a number is a mismatch. |
| `seed` | `0` | Seed for the random checkpoints. |

**Output:** `first_bad_index` is the earliest row seen to change among the rows compared. With `check="prefix"` and dense checkpoints it is the earliest leaking row in the data.

**Limitations:** only the probed prefixes are tested; pass `checkpoints=range(1, len(bars) + 1)` to test all of them. Numerical noise that depends on series length (FFT convolutions, pairwise summation) can exceed `rtol` and show as small mismatches. The function must be deterministic; a non-deterministic `fn` raises `ValueError`.

### `perturbation_test(fn, bars, ...)`

**Catches:** the same leaks as `truncation_test`, plus code that behaves the same on short data but reads ahead when the future exists. It never runs `fn` on a short prefix.

**How:** for each cut `c`, replaces every bar after `c` with scrambled but valid prices (each OHLC value multiplied by an independent factor in `[1.5, 3]`, or divided by one; high and low rebuilt so every bar stays valid) and requires output rows `0..c` to be unchanged.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `cuts` | `None` | Cut indices in `0..len(bars) - 2`. By default `n_cuts` are chosen: half evenly spaced, some at rows where the output is non-zero, the rest at random. |
| `n_cuts` | `5` | Number of cuts when `cuts` is not given. |
| `n_trials` | `3` | Perturbations per cut. Even trials scale the future up, odd trials scale it down. |
| `rtol`, `atol`, `seed` | `1e-9`, `0.0`, `0` | As for `truncation_test`. |

**Limitations:** volume and timestamps are not perturbed, so a leak through future volume or time alone is invisible. A leak that leaves the output unchanged on the scrambled future (a signal that clips to the same value either way) is missed; more cuts and trials reduce that chance. Because only rows `0..c` are compared at each cut, `first_bad_index` depends on where the cuts fell.

### `check_factory(factory, bars, ...)`

**Catches:** a strategy whose `decide(i, history)` only receives past bars but reads an indicator its factory precomputed from the full data through a closure.

**How:** builds the strategy from the real bars and from bars scrambled after a cut, calls `decide` on every bar from `warmup` to the cut, and requires every decision to match. Decisions are compared after parsing, so `(1, 99, 102)` and `Setup(1, 99, 102)` are the same decision.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `factory` | | `factory(bars)` returns a `decide(i, history)` callable, as accepted by `run`. |
| `cuts`, `n_cuts` | `None`, `5` | As for `perturbation_test`, in `warmup..len(bars) - 2`. |
| `n_trials` | `2` | Perturbations per cut. |
| `warmup` | `0` | First bar `decide` is called on. |
| `rtol`, `atol` | `1e-9`, `0.0` | Tolerance for comparing stop and target levels. |

**Limitations:** costs about `n_cuts * n_trials * len(bars) / 2` calls to `decide`. `decide` must be deterministic. Leaks through volume or timestamps are not seen.

### `self_test(n_bars=300, seed=0)`

Runs the harness on built-in probes and returns `{probe_name: True}` for each probe it judged correctly. On a driftless random walk both tests must pass a causal rolling mean and an EWMA, and catch a centred rolling mean, a full-sample z-score, a `shift(-1)`, a backward fill of a sparse series and a window that includes bar `i + 1`. `check_factory` must pass an honest factory and catch a leaky one. Use it to confirm the harness works in your environment.

## Random-entry nulls

Module `backtest_skeptic.random_entry`.

The question: could random entries with the same number of trades, the same direction mix, the same bracket geometry, the same costs and the same exit rules have produced this result? Only the entry timing (and in `side_test`, the direction) is under test.

P-values are one-sided and computed as `(1 + #{null >= observed}) / (1 + n_sims)`, which is never zero. Null values within floating-point rounding of the observed value count as ties, against the strategy.

**Citation for the p-value:** Phipson & Smyth, 2010, "Permutation P-values Should Never Be Zero", Statistical Applications in Genetics and Molecular Biology 9(1). The matching scheme (which features are copied, how unfillable setups are handled, overlap) is **heuristic**.

All three return a `NullTestResult`: `observed`, `null` (one metric value per simulation), `p_value`, `percentile` (mid-rank, 0 to 100), `null_mean`, `null_std`, `z_score` (descriptive only), `excess` (`observed - null_mean`), `n_trades`, `n_sims`, `unfilled`, `settings` and `notes`.

### `random_entry_test(bars, trades, ...)`

**Catches:** entry signals that add nothing beyond the bracket's payoff shape, the costs, the direction mix and the period's drift.

**How:** each simulation draws exactly `len(trades)` distinct signal bars uniformly from the eligible bars, gives them the strategy's sides and stop/target distances, resolves each through the engine's fill rules with the same costs, and records the metric. A random setup that cannot be filled gets a fresh bar.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `trades` | | The strategy's `Trade` objects (at least 5) or the `BacktestResult`. A result supplies `costs`, `max_bars`, `same_bar` and `entry_delay`. |
| `n_sims` | `1000` | Simulations. The smallest attainable p-value is `1 / (1 + n_sims)`. |
| `sides` | `"match"` | `"match"` permutes the strategy's own sides; `"random"` is a fair coin; `"long"`/`"short"` is all one direction. |
| `geometry` | `"strategy"` | Copies a random strategy trade's stop and target distances as fractions of the signal bar's close. Or a callable `geometry(bars, j, side) -> (stop, target)` that reads only bars `0..j`. |
| `eligible` | `None` | Boolean mask of bars a random entry may be decided on. Pass the strategy's regime or session filter. |
| `warmup` | `0` | First bar a random entry may be decided on. |
| `metric` | `"expectancy"` | `"expectancy"` (mean net R), `"win_rate"`, `"total_r"`, or a callable on a 1-D array of R. Higher must mean better. |
| `verify` | `True` | Re-resolve every strategy trade through the engine and raise if any does not reproduce. Turn off only for trade logs from another engine. |
| `seed` | `0` | Random seed. |

**Output:** a small `p_value` means random entries rarely matched the strategy. It says the timing adds something. Whether the strategy is profitable is a separate question: read the observed metric and the costs.

**Limitations:** edge from the bracket geometry on a trending market, or from a long bias in a rising market, is shared by the null and is not credited to the signal. Random trades are resolved independently and may overlap, which makes the null slightly wider (conservative). `eligible` should describe where the strategy could have traded; a mask fitted after seeing the results biases the null. A p-value from one of many tried strategies needs a multiple-testing correction.

### `side_test(bars, trades, mode="random", ...)`

**Catches:** direction logic that adds nothing. Keeps the strategy's own decision bars and bracket distances and changes only the side.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mode` | `"random"` | `"random"`: fair-coin sides per simulation, ranked with a p-value. `"flip"`: one deterministic run with every side reversed; `p_value` is `None` and `excess` is observed minus flipped. |
| `n_sims`, `seed`, `metric`, `costs`, `max_bars`, `same_bar`, `entry_delay`, `verify` | | As in `random_entry_test`. |

**Limitations:** a reversed setup the engine cannot fill is dropped (counted in `unfilled`). Reversing a trade does not change which later trades the strategy was free to take. A constant long bias in a rising market counts as direction skill here; use `random_entry_test` with `sides="match"` to control for it.

### `positive_control(bars, horizon=5, stop_frac=0.01, target_frac=0.01, n_sims=500, ...)`

Runs `synthetic.oracle_signal` (which reads `close[i + horizon]`) through the engine and through `random_entry_test`. The p-value should be below 0.01 on any realistic price series (which needs `n_sims >= 99`). If it is not, the harness, the data or the settings cannot detect a blatant edge. Fees must be zero; spread and slippage are allowed.

## No-skill baseline

Module `backtest_skeptic.baseline`. The derivation and the cost arithmetic are in [fill-rules.md](fill-rules.md#the-no-skill-win-rate).

**Citation:** Feller, W. *An Introduction to Probability Theory and Its Applications*, vol. 1, ch. XIV (random walks and ruin problems). The warning thresholds (10% unresolved trades, half a standard error of exclusion bias) are **heuristic**.

### `no_skill_test(trades, costs=None, tolerance=1e-9, exact_limit=20_000)`

**Catches:** a high win rate that comes from bracket geometry (a target much closer than the stop) instead of from the entries, and a win rate that beats chance but not costs.

**Inputs:** a `BacktestResult` (its costs are used) or an iterable of `Trade`. `costs` must be the costs the trades were filled with; trades whose fills contradict them raise `ValueError`. `tolerance` is in R. Above `exact_limit` resolved trades the p-value uses a normal approximation.

**Output (`BaselineResult`):** rates are fractions, edges are differences of fractions.

| Field | Meaning |
| --- | --- |
| `win_rate` | Target hits / resolved trades. |
| `no_skill_rate` | Mean of `a / (a + b)`, the win rate with no skill. |
| `breakeven_rate` | Mean break-even win rate after costs. |
| `skill_edge` | `win_rate - no_skill_rate`. Positive means better than chance. |
| `cost_drag` | `breakeven_rate - no_skill_rate`. |
| `margin` | `win_rate - breakeven_rate`. Positive means the trades make money after costs. |
| `p_value` | One-sided `P(W >= wins)` with `W` Poisson-binomial under no skill. |
| `p_value_lower` | `P(W <= wins)`. Small when the entries are worse than random, which usually means a bug or an adverse filter. |
| `no_skill_expectancy_r` | Mean R a skill-less trader expects with these brackets and costs. |
| `exclusion_bias`, `all_exits_*` | Estimated bias from leaving out time and end exits, and an exclusion-free score over every trade. |
| `gap_fills`, `ambiguous`, `ambiguous_wins` | Stops that filled beyond the stop, and exits on bars that touched both levels. |

Only trades that exited at the stop or the target are tested against `a / (a + b)`. A warning is raised when time and end-of-data exits are more than 10% of the trades, or when leaving them out is estimated to move `skill_edge` by more than half a standard error.

**Limitations:** assumes a driftless mid price; a trend gives a long-only strategy a win rate above `a / (a + b)` with no timing skill. Gap fills lose more than the closed-form loss, so `breakeven_rate` understates what such trades need. Ambiguous bars booked as stops bias the win rate down. The exact p-value treats outcomes as independent Bernoulli trials, which volatility clustering and adaptive brackets make approximate.

### `expectancy_decomposition(trades, costs=None, tolerance=1e-9)`

Splits mean R into `skill_r` (R from hitting targets more often than chance, zero on average without skill) and `cost_r` (what a skill-less trader with the same brackets expects, negative whenever there are costs). `mean_r = skill_r + cost_r + residual_r`, where `residual_r` is rounding error. Uses trades that exited at the target or at the stop without a gap. Positive `mean_r` with `skill_r` near zero means the profit does not come from the entries; positive `skill_r` with negative `mean_r` means the entries predict direction but not well enough to pay for trading.

### `bracket_odds(trade, costs=None)`

No-skill probability, payoffs and break-even rate of a single `Trade`: `stop_distance` (`a`), `target_distance` (`b`), `no_skill_prob`, `r_win`, `r_loss`, `breakeven_prob` and `no_skill_expectancy_r`.

### `breakeven_win_rate(risk, reward, cost_r=0.0)`

Planning helper: `(1 + cost_r) * risk / (risk + reward)`, with `risk` and `reward` as distances in the same price units and `cost_r` the round-trip cost in units of `risk`. With zero costs it equals the no-skill rate. A result of 1 or more means no win rate breaks even.

## Fill realism

Module `backtest_skeptic.fills`. **Heuristic** (no canonical citation).

### `audit_fills(bars, trades, costs=None, tolerance=0.0)`

**Catches:** fills the bars cannot support. Works on trade logs from any backtester.

**Inputs:** `bars` (the bars the backtest ran on); `trades` as a `TradeLog`, a `BacktestResult`, or an iterable of `Trade` objects or mappings. Required per trade: side, entry and exit price, entry and exit bar (as bar indices or timestamps). Optional: signal bar, stop, target, exit reason; each one enables more checks. `costs` are taken back out of each fill before comparing with the bars, so they must be the costs the backtest charged. `tolerance` is extra price slack on top of a relative `1e-9`; set it to about half a tick when the log rounds prices.

**Issue codes:**

| Code | Severity | Meaning |
| --- | --- | --- |
| `exit_before_entry` | fail | The exit bar comes before the entry bar. |
| `entry_before_signal` | fail | The entry filled on a bar before the signal bar. |
| `phantom_signal_bar_fill` | fail | Filled on the signal bar at a price other than its close, which traded before the signal existed. |
| `filled_at_signal_close` | warn | Filled at the close the signal was computed from. |
| `entry_outside_bar` | fail | Entry mid price outside the entry bar's low-high range. |
| `phantom_exit_fill` | fail | Exit mid price outside the exit bar's range, better for the trader than any traded price. |
| `exit_outside_bar` | warn | Exit mid price outside the exit bar's range, worse for the trader. |
| `gap_stop_fill_too_good` | fail | A stop filled better than the open of a bar that gapped through it. |
| `missed_earlier_touch` | fail | The price traded through the stop or target before the logged exit, or a target was booked on a bar that opened through the stop. |
| `possible_entry_bar_touch` | warn | After an intrabar entry, the entry bar traded through a level, possibly before the entry. |
| `ambiguous_resolved_optimistically` | warn | The exit bar touched both levels, opened between them, and the log books the target. |

**Output (`FillAudit`):** `issues`, `counts` per code, `trades_with_fail`, `fail_share`, `passed` (no fail anywhere; warnings do not stop a pass), `ambiguous_exits`, `checked` (how many trades each group of checks could examine), and `booked_expectancy_r`, `worst_expectancy_r`, `best_expectancy_r`: the mean R of trades with a stop and a target under the logged prices and two replays that differ only where OHLC data cannot decide the order of prices. A booked figure above the best case almost always means the log got fills the bars cannot support.

**Limitations:** bars cannot show the order of prices inside a bar; the audit assumes only that the open is the first price and, for intrabar entries, that the price moves continuously from the open. Stops and targets are assumed fixed for the life of the trade (for trailing stops, leave out the levels and use reason `other`). An intrabar fill is checked against the bar's range only. Volume, liquidity and market impact are not modelled. A misaligned log shows up as many fails across the checks, with no single diagnosis.

## Robustness

Module `backtest_skeptic.robustness`. Values are per-trade (or per-period) results, usually net R; a `BacktestResult` or a sequence of `Trade` is read as its R values. Empty input and non-finite values are rejected.

### `concentration(values, groups=None, top=(1, 5, 0.1))` (heuristic)

**Catches:** a total made by one outlier or one day. Reports the share of the total from the best item, the best five and the best 10% (an `int` is a count, a `float` in `(0, 1)` a fraction), optionally after summing values by a group key such as `period_keys(times, "D")`. Compare each `share` with `even_share`. A share above 100% means everything else lost money (`fragile`). When the total is zero or negative every share is `None`. **Limitations:** descriptive only; a positively skewed payoff is concentrated by design.

### `drop_best(values, k, groups=None)` (heuristic)

**Catches:** an expectancy that only exists because of the right tail. Reports mean, total, t-statistic and win share before and after removing the best `k` values or groups. **Limitations:** removing the best values after seeing them is a selection, so the "after" t-statistic is a stress figure with no significance level; values are treated as independent.

### `subperiods(values, keys)` and `period_keys(times, freq)` (heuristic)

**Catches:** an edge that lived in one regime. `period_keys` labels timestamps by day, ISO week, month, quarter or year (`"D"`, `"W"`, `"M"`, `"Q"`, `"Y"`). `subperiods` reports per-period statistics, the share of periods that made money, the worst and best period, and `concentrated` when the best period produced more than half of a positive total (a heuristic threshold). **Limitations:** periods with few values are noisy, and the share of positive periods depends on the period length.

### `block_bootstrap(values, block=None, n_boot=10_000, seed=0, statistic="mean", ci=0.95)`

**Catches:** a confidence interval that is too narrow because trades are serially dependent. Circular moving-block bootstrap: blocks of `block` consecutive values wrap around the end of the series. Default block length `ceil(n ** (1/3))`; `block=1` is the ordinary bootstrap. Returns `estimate`, `ci_low`, `ci_high`, `std_error` and `prob_positive` (the share of resamples above zero; not a p-value). **Limitations:** assumes stationarity and dependence shorter than a block; the default block length is a rule of thumb; percentile intervals undercover in small samples.

**Citations:** Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3). Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428).

### `clustered_t(values, clusters)`

**Catches:** an inflated t-statistic from trades that share a day, a symbol or a market event. Returns `(mean, se, t, n_clusters)` with the CR1 cluster-robust standard error. With one value per cluster it equals the ordinary standard error. **Limitations:** needs at least two clusters and is only reliable with many (roughly 30 or more).

**Citation:** Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources* 50(2).

### `cost_stress(bars, strategy, costs, multipliers=(0.5, 1.0, 1.5, 2.0, 3.0), **engine_kwargs)` (heuristic)

**Catches:** an edge that disappears under slightly worse execution than assumed. Re-runs the engine with every cost component scaled by each multiplier (`strategy` is a `decide` callable or a `(side, stop, target)` tuple of arrays) and reports the multiplier where expectancy crosses zero, by linear interpolation between tested multipliers. It never extrapolates: `status` is `"crossing"`, `"unprofitable_at_lowest"`, `"profitable_at_all"` or `"no_trades"`. A break-even below about 1.5x leaves little room for error (a heuristic guide). **Limitations:** scales spread, fees and slippage by the same factor.

### `entry_delay_test(bars, strategy, delays=(0, 1, 2, 3), costs=None, **engine_kwargs)` (heuristic)

**Catches:** a result that needs an immediate fill, or uses information from the signal bar. Re-runs the engine with each entry delay; stop and target levels stay where the strategy put them. `edge_at_first` is true when the smallest delay shows expectancy above zero with t-statistic of at least 2; `collapse_delay` is the first later delay whose expectancy falls below 25% of the first (both heuristic thresholds). **Limitations:** a genuine one-bar edge collapses too, and this test cannot tell it from a leak; use the lookahead tests for that. Delayed entries are skipped when the open has passed the stop or target, so the trade count can fall.

## Return-series statistics

Module `backtest_skeptic.stats`. Returns are per-period simple returns or P&L in any consistent unit; R multiples work too. Functions that take a kurtosis reject values that violate `kurtosis >= 1 + skew^2`, which catches excess kurtosis passed by mistake.

### `probabilistic_sharpe_ratio(returns=None, *, sr, n, skew, kurtosis, benchmark=0.0)`

**Catches:** a Sharpe ratio that looks high only because the track record is short or the returns are negatively skewed and fat-tailed. Returns `P(true Sharpe > benchmark)`. Give either `returns` or all four of `sr` (per period), `n`, `skew` and `kurtosis`; mixing the forms or omitting one raises. **Limitations:** asymptotic and IID; does not correct for the number of strategies tried.

**Citation:** Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).

### `deflated_sharpe_ratio(returns=None, *, sr, n, skew, kurtosis, n_trials, var_trials_sharpe)`

**Catches:** selection bias. Returns the PSR against the Sharpe ratio expected from the best of `n_trials` skill-less trials. `n_trials` (independent trials) and `var_trials_sharpe` (variance of the trials' per-period Sharpe ratios) are required; there is no default. To convert annual inputs, divide an annual Sharpe ratio by `sqrt(periods_per_year)` and an annual variance by `periods_per_year`. The test suite reproduces the paper's example (0.9004 for 100 trials). **Limitations:** needs an honest count of independent trials; correlated trials count as fewer. Inherits PSR's IID asymptotics.

**Citation:** Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management* 40(5).

### `min_track_record_length(sr, benchmark=0.0, skew=0.0, kurtosis=3.0, prob=0.95)`

Number of return observations needed for a per-period Sharpe ratio `sr` to be significantly above `benchmark` with confidence `prob`. Returns `inf` when `sr <= benchmark`. Divide by periods per year for years. **Limitations:** asymptotic IID result; the moments need a long sample to be estimated reliably. **Citation:** Bailey & Lopez de Prado (2012), as above.

### `min_backtest_length(n_trials, target_sharpe)`

Minimum backtest length in years below which the best of `n_trials` independent skill-less configurations is expected to reach an in-sample annual Sharpe of `target_sharpe` by luck. The paper's example: with 5 years of data, no more than about 45 configurations before an annual Sharpe of 1 is expected from noise. **Limitations:** a necessary condition only; assumes independent trials with normal, serially uncorrelated returns.

**Citation:** Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS* 61(5).

### `pbo(performance, n_splits=16, metric=None)`

**Catches:** a parameter search whose winner is a fluke. `performance` is a `T x N` matrix (rows are periods, columns are every configuration tried). Combinatorially symmetric cross-validation splits the rows into `n_splits` blocks and uses every half as in-sample once (12,870 splits for 16 blocks). `pbo` is the share of splits where the in-sample winner ranks below the out-of-sample median; near or above 0.5 means the search is fitting noise. `prob_oos_loss` is the share of splits where the winner lost out of sample. Default metric: per-period Sharpe ratio. **Limitations:** columns must be comparable on the same metric; blocks are treated as exchangeable, which strong regime changes break; a low PBO does not mean the winner is profitable.

**Citation:** Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).

### `newey_west(x, lags=None, kernel="bartlett")`

**Catches:** t-statistics inflated by serial correlation (overlapping trades, multi-bar holds, overlapping forward-return labels). Returns `mean, se, t`. Default lags `floor(4 * (T / 100) ** (2 / 9))`. `kernel="uniform"` gives Hansen-Hodrick weights; use `lags = h - 1` for overlapping `h`-period returns. **Limitations:** asymptotic; a lag choice made after seeing the result is another degree of freedom.

**Citations:** Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3). Newey, W. K. & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies* 61(4). Hansen, L. P. & Hodrick, R. J. (1980). Forward Exchange Rates as Optimal Predictors of Future Spot Rates. *Journal of Political Economy* 88(5).

### `lo_annualized_sharpe(returns, periods)`

**Catches:** the `sqrt(12)` or `sqrt(252)` rule overstating the annual Sharpe of autocorrelated returns. Returns the corrected and the naive annualised ratio. **Limitations:** assumes stationarity; `periods` should be small relative to the sample.

**Citation:** Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4).

### `adjust_pvalues(pvalues, method="holm")`

Multiple-testing adjusted p-values: `"bonferroni"`, `"holm"` (family-wise error rate under any dependence), `"bh"` (false discovery rate for independent or positively dependent tests), `"by"` (false discovery rate under arbitrary dependence). **Limitation:** the count must include every test run, including the ones dropped for looking bad.

**Citations:** Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2). Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. *Journal of the Royal Statistical Society B* 57(1). Benjamini, Y. & Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple Testing under Dependency. *Annals of Statistics* 29(4).

### `haircut_sharpe(sharpe_annual, n_obs, n_trials, *, periods_per_year, method="bonferroni", distribution="t")`

**Catches:** a Sharpe ratio reported as if it were the only strategy tested. Converts the Sharpe ratio to a p-value, adjusts it for `n_trials` tests (`"bonferroni"` or `"sidak"`), and converts back. Returns the haircut Sharpe ratio and the haircut fraction. **Limitations:** the single-test, independent-trials version; the Holm and BHY haircuts are not implemented.

**Citation:** Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1).

Helpers in the same module: `sharpe_ratio`, `sharpe_standard_error`, `moments`, `expected_max_sharpe`, `annualize_sharpe`, `deannualize_sharpe`, `lo_scale_factor`.

## Holdout bookkeeping

Module `backtest_skeptic.holdout`.

### `split_holdout(n, holdout_fraction=0.2, holdout_start=None, purge=0)`

Splits `n` bars into train, a purged gap and a final holdout that never share a bar (train is `[0, start - purge)`, the holdout `[start, n)`). Set `purge` to at least the number of bars a label or trade looks ahead (for engine trades, `max_bars + entry_delay`), so no training label is computed from holdout prices. **Heuristic** (bookkeeping, no statistical test).

### `HoldoutLedger(path)`

Append-only JSON-lines record that each holdout was sealed once (`seal`) and read once (`read`). A second read raises `HoldoutAlreadyRead` unless `allow_reread=True`, which records it as a re-read. Each line carries the SHA-256 of the previous line, so editing, reordering or deleting a line is detected on load. **Limitations:** an audit trail with no access control; nothing stops someone from evaluating the holdout without telling the ledger.

**Citation (motivation):** Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. & Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. *Science* 349(6248).

### `TrialRegistry(path)`

Append-only record of every configuration tried. `n_trials` and `sharpe_variance` are the `n_trials` and `var_trials_sharpe` inputs of `deflated_sharpe_ratio`. Configurations are identified by a hash of their parameters; re-running one does not add a trial. **Limitations:** distinct configurations overstate the number of independent trials when they are correlated (conservative); trials you ran but did not record make it optimistic.

**Citation:** Bailey & Lopez de Prado (2014), as above.

## Factor spanning

Module `backtest_skeptic.spanning`.

### `spanning_regression(returns, factors, names=None, hac_lags=None, periods_per_year=None)`

**Catches:** an "edge" that is exposure to something you could have bought without the strategy. OLS of per-period strategy returns on factor returns with an intercept and Newey-West standard errors. `alpha` is the mean return the factors do not explain; `alpha_share` is `alpha / mean_return`; `appraisal_ratio` is `alpha / residual_std`. A long bias in a rising market shows as a large beta, a high R-squared, alpha near zero and a small `alpha_share`. NaN rows are dropped and counted. **Limitations:** linear and static, so time-varying exposure appears as alpha; a factor you do not include explains nothing; HAC errors are noisy with few observations per lag.

**Citations:** Huberman, G. & Kandel, S. (1987). Mean-Variance Spanning. *Journal of Finance* 42(4). Newey & West (1987, 1994), as above.

### `proxy_replication(returns, proxy, hac_lags=None, periods_per_year=None)`

The one-factor case phrased as a question: does a no-information proxy (the market, a size or volatility spread, an equal-weight basket) reproduce the edge? `reproduced_share` near 1 with `alpha_t` near 0 means the proxy explains it. **Limitations:** one proxy, linear constant beta; ruling out one proxy does not rule out every factor.
