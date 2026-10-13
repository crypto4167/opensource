# Calibration of the statistical checks

This page records how often each test in `stats`, `baseline` and
`random_entry` rejects at alpha = 0.05 when there is nothing to find (its
false-positive rate, which should be close to 5%), and how often it rejects
when a known edge is planted (its power). Every number below comes from one
run of `scripts/calibration.py` at its default settings:

```
python scripts/calibration.py
```

The run is seeded and took 294 seconds with Python 3.13 and numpy 2.5 on the
machine used for this review. `--scale 0.2` gives a quicker, noisier pass and
`--only SECTION` runs one section.

Each rate is the share of independent replications that rejected. The Monte
Carlo standard error (MC SE) is `sqrt(rate * (1 - rate) / reps)`; for the
FDR and PBO rows it is the standard error of a mean. A rate within about two
standard errors of 0.05 is consistent with a correctly sized test. A rate
well below 0.05 means the test is conservative: it is safe but needs a larger
edge to see one.

## Summary

* `random_entry_test` and `side_test` held their size in every null scenario
  (2.5% to 5.0% with 400 replications, MC SE about 0.9 points), including
  costs, `max_bars`, gaps, drift and side-dependent brackets. With about 300
  trades and 1% brackets they found a planted edge of 0.2 bar volatilities in
  88% of runs and an edge of 0.3 in 99.5% to 100%.
* `no_skill_test` held its size with no time exits, with costs, gaps of 0.3%
  and small samples (2.4% to 4.1% over 1000 replications). The
  resolved-only `p_value` is biased when trades time out: with a 3% stop,
  0.5% target and `max_bars=5` it rejected 25.4% of null runs. A warning
  accompanied all but 5 of those 1000 runs, and `all_exits_p_value`, which
  scores every exit, rejected 3.9%. With far targets and many time-outs it is
  the other way round (0.3%). With gaps as large as the bar volatility and a
  far target both scores reached 5.3% to 5.4%.
* PSR is close to 5% for normal returns (4.7% to 5.4%). For fat-tailed or
  skewed returns it over-rejects at small n (Student t3: 7.0% at n = 60;
  negated lognormal: 8.05% at n = 60, 5.7% at n = 1000). At SR = 0 the
  formula reduces to `Phi(SR * sqrt(n - 1))`, so this is the finite-sample
  behaviour of the sample Sharpe ratio, not an error in the higher-moment
  terms.
* DSR applied to the best of N pure-noise trials, with the trial variance
  estimated from the same trials, rejected at most 0.3% of runs. It is
  conservative by construction: it asks whether the selected Sharpe ratio
  exceeds the expected maximum by a full single-test margin.
* Newey-West t-statistics with the default lag are close to 5% for white
  noise and weak autocorrelation at T = 1000 (4.9%, 5.4%), but reject 12.55%
  (T = 250) and 10.0% (T = 1000) on AR(1) data with coefficient 0.6. A
  plain t-statistic rejects 33% there. For h-period overlapping sums the
  uniform kernel with `lags = h - 1` is close to 5% at T = 1000 (4.55%, 5.8%)
  and the Bartlett kernel is not (11% to 16%); at T = 250 with h = 20 the
  uniform kernel still rejects 10.75%, because 250 observations hold only
  about 12 independent 20-period sums.
* `adjust_pvalues` controlled the family-wise error rate (Bonferroni, Holm:
  2.5% to 4.8%) and the false discovery rate (BH: at or below its bound
  `0.05 * m0 / m` in all four settings; BY: 0.9% to 1.5%) for 20 tests,
  independent or equicorrelated at 0.5.
* PBO on pure-noise configuration matrices averaged 0.49 to 0.52 (MC SE 0.012
  to 0.018), as it should. With one configuration having a true per-period
  Sharpe ratio of 0.1 it averaged 0.17.

## Defects found by this calibration and fixed

These were found during the review that produced this page. Each has a
regression test and its own commit; the rates in the table are after the
fixes.

1. `random_entry_test(sides="match")` drew each random trade's side and its
   bracket template independently. A strategy whose longs and shorts use
   different brackets then had its brackets re-paired across sides in the
   null. On random-walk bars with a drift of 0.2% per bar, random entries,
   longs with a 1% stop and 2% target and shorts with a 2% stop and 0.5%
   target, the test rejected 35.5% of 200 null runs (27% in the lower tail
   with the drift reversed). Each simulation now uses every strategy trade's
   side and bracket together. The same scenario then rejected 6.0% of 200
   runs, and the harness scenario "side-dependent brackets, drift
   +0.2%/bar" rejects 3.0%.
2. `no_skill_test` computed `all_exits_p_value` with a normal approximation.
   With about 19 trades, a 0.3% stop and a 5% target it rejected 7.9% of
   2000 null runs (MC SE 0.6 points); with about 38 trades, a 0.5% stop and a
   3% target, 7.3%. It now uses the Poisson-binomial tail at the score sum
   rounded up, which is exact when every trade reached a level. The table
   rows "about 19 trades" and "about 38 trades" (3.6%, 3.4%) are after the
   fix. For fractional scores from time exits this p-value is not proven
   valid in general. It stayed at or below 5% in every scenario tried: the
   time-exit rows of the table, and separate runs of 1500 replications
   with `max_bars` of 1 or 2 and far or near targets, where it rejected 0%
   to 2.5%.

Two numerical defects were found by edge-case tests rather than by
simulation: `expected_max_sharpe` (and with it `min_backtest_length` and
`deflated_sharpe_ratio`) raised for N of about 1e16 or more trials, and
`sharpe_ratio`, `moments`, `newey_west`, `lo_annualized_sharpe` and `pbo`
overflowed or underflowed on values near 1e200 or 1e-200 (for example
`sharpe_ratio([1, 2, 3] * 1e200)` returned 0.0).

## Scenarios

Trade-level scenarios use `synthetic.random_walk_bars` (a geometric random
walk with 1% bar volatility and continuous intrabar highs and lows) and
enter at random: on each bar a signal fires with probability 0.05 and its
side is a fair coin. Brackets are fractions of the signal bar's close. Costs,
where listed, are `Costs(half_spread=0.02, fee_rate=0.0005, slippage=0.02)`
on prices near 100. `random_entry_test` and `side_test` use 2000 bars (about
95 trades) and 199 simulations, so the smallest attainable p-value is 0.005.
`no_skill_test` uses 4000 bars (about 190 trades) unless the scenario says
otherwise.

Power scenarios use `synthetic.planted_signal_bars` with 3000 bars, a signal
on 10% of bars, and a drift of `edge` bar volatilities in the signal's
direction during the next bar. The strategy trades every signal with 1%
brackets and `max_bars=1` (about 300 trades). With 0.5% brackets a quarter of
the trades touched both levels in their one bar; the engine books those as
stops, and the all-exits `no_skill_test` found none of 40 planted edges of
0.3, which is why 1% brackets are used here.

Return-series scenarios: PSR uses n returns from the stated distribution with
mean zero. DSR draws T x N independent returns, selects the column with the
highest sample Sharpe ratio and passes N and the sample variance of the N
Sharpe ratios. Newey-West uses a stationary AR(1) after a 200-observation
burn-in, or sums of h consecutive standard normal draws. `adjust_pvalues`
uses one-sided normal p-values of 20 z-scores with the stated equicorrelation;
the non-null z-scores are shifted by 3. PBO uses T x N independent standard
normal returns, and for the edge rows adds the stated mean to column 0.

## Full table

| test | scenario | reps | rate | MC SE | note |
|---|---|---|---|---|---|
| PSR null (SR = 0) | normal, n=60 | 2000 | 0.0525 | 0.0050 |  |
| PSR null (SR = 0) | normal, n=250 | 2000 | 0.0470 | 0.0047 |  |
| PSR null (SR = 0) | normal, n=1000 | 2000 | 0.0540 | 0.0051 |  |
| PSR null (SR = 0) | Student t5, n=60 | 2000 | 0.0580 | 0.0052 |  |
| PSR null (SR = 0) | Student t5, n=250 | 2000 | 0.0585 | 0.0052 |  |
| PSR null (SR = 0) | Student t5, n=1000 | 2000 | 0.0580 | 0.0052 |  |
| PSR null (SR = 0) | Student t3, n=60 | 2000 | 0.0700 | 0.0057 |  |
| PSR null (SR = 0) | Student t3, n=250 | 2000 | 0.0635 | 0.0055 |  |
| PSR null (SR = 0) | Student t3, n=1000 | 2000 | 0.0515 | 0.0049 |  |
| PSR null (SR = 0) | neg. skew (-lognormal), n=60 | 2000 | 0.0805 | 0.0061 |  |
| PSR null (SR = 0) | neg. skew (-lognormal), n=250 | 2000 | 0.0640 | 0.0055 |  |
| PSR null (SR = 0) | neg. skew (-lognormal), n=1000 | 2000 | 0.0570 | 0.0052 |  |
| PSR power | normal, true SR 0.1/period, n=250 | 2000 | 0.4510 | 0.0111 |  |
| PSR power | normal, true SR 0.2/period, n=250 | 2000 | 0.9370 | 0.0054 |  |
| DSR null (best of N) | normal, N=10, T=250 | 1000 | 0.0010 | 0.0010 |  |
| DSR null (best of N) | normal, N=10, T=1000 | 1000 | 0.0010 | 0.0010 |  |
| DSR null (best of N) | normal, N=100, T=250 | 1000 | 0.0010 | 0.0010 |  |
| DSR null (best of N) | normal, N=100, T=1000 | 1000 | 0.0030 | 0.0017 |  |
| DSR null (best of N) | Student t5, N=10, T=250 | 1000 | 0.0000 | 0.0000 |  |
| DSR null (best of N) | Student t5, N=10, T=1000 | 1000 | 0.0000 | 0.0000 |  |
| DSR null (best of N) | Student t5, N=100, T=250 | 1000 | 0.0000 | 0.0000 |  |
| DSR null (best of N) | Student t5, N=100, T=1000 | 1000 | 0.0000 | 0.0000 |  |
| DSR power | one of N=10 has SR 0.1/period, T=1000 | 1000 | 0.1500 | 0.0113 |  |
| DSR power | one of N=10 has SR 0.2/period, T=1000 | 1000 | 0.9730 | 0.0051 |  |
| Newey-West null | AR(1) phi=0.0, T=250, default lags | 2000 | 0.0545 | 0.0051 |  |
| plain t null | AR(1) phi=0.0, T=250 | 2000 | 0.0450 | 0.0046 |  |
| Newey-West null | AR(1) phi=0.3, T=250, default lags | 2000 | 0.0775 | 0.0060 |  |
| plain t null | AR(1) phi=0.3, T=250 | 2000 | 0.1540 | 0.0081 |  |
| Newey-West null | AR(1) phi=0.6, T=250, default lags | 2000 | 0.1255 | 0.0074 |  |
| plain t null | AR(1) phi=0.6, T=250 | 2000 | 0.3340 | 0.0105 |  |
| Newey-West null | AR(1) phi=0.0, T=1000, default lags | 2000 | 0.0485 | 0.0048 |  |
| plain t null | AR(1) phi=0.0, T=1000 | 2000 | 0.0470 | 0.0047 |  |
| Newey-West null | AR(1) phi=0.3, T=1000, default lags | 2000 | 0.0540 | 0.0051 |  |
| plain t null | AR(1) phi=0.3, T=1000 | 2000 | 0.1470 | 0.0079 |  |
| Newey-West null | AR(1) phi=0.6, T=1000, default lags | 2000 | 0.0995 | 0.0067 |  |
| plain t null | AR(1) phi=0.6, T=1000 | 2000 | 0.3350 | 0.0106 |  |
| Newey-West null | 5-period overlapping sums, T=250, bartlett | 2000 | 0.1095 | 0.0070 |  |
| Newey-West null | 5-period overlapping sums, T=250, uniform | 2000 | 0.0620 | 0.0054 |  |
| Newey-West null | 20-period overlapping sums, T=250, bartlett | 2000 | 0.1555 | 0.0081 |  |
| Newey-West null | 20-period overlapping sums, T=250, uniform | 2000 | 0.1075 | 0.0069 |  |
| Newey-West null | 5-period overlapping sums, T=1000, bartlett | 2000 | 0.1095 | 0.0070 |  |
| Newey-West null | 5-period overlapping sums, T=1000, uniform | 2000 | 0.0455 | 0.0047 |  |
| Newey-West null | 20-period overlapping sums, T=1000, bartlett | 2000 | 0.1180 | 0.0072 |  |
| Newey-West null | 20-period overlapping sums, T=1000, uniform | 2000 | 0.0580 | 0.0052 |  |
| adjust_pvalues bonferroni FWER | m=20, 20 true nulls, equicorrelation 0.0 | 5000 | 0.0476 | 0.0030 |  |
| adjust_pvalues holm FWER | m=20, 20 true nulls, equicorrelation 0.0 | 5000 | 0.0476 | 0.0030 |  |
| adjust_pvalues bh FDR | m=20, 20 true nulls, equicorrelation 0.0 | 5000 | 0.0480 | 0.0030 | bound 0.0500 |
| adjust_pvalues by FDR | m=20, 20 true nulls, equicorrelation 0.0 | 5000 | 0.0146 | 0.0017 | bound 0.0500 |
| adjust_pvalues bonferroni FWER | m=20, 20 true nulls, equicorrelation 0.5 | 5000 | 0.0398 | 0.0028 |  |
| adjust_pvalues holm FWER | m=20, 20 true nulls, equicorrelation 0.5 | 5000 | 0.0398 | 0.0028 |  |
| adjust_pvalues bh FDR | m=20, 20 true nulls, equicorrelation 0.5 | 5000 | 0.0434 | 0.0029 | bound 0.0500 |
| adjust_pvalues by FDR | m=20, 20 true nulls, equicorrelation 0.5 | 5000 | 0.0132 | 0.0016 | bound 0.0500 |
| adjust_pvalues bonferroni FWER | m=20, 15 true nulls, equicorrelation 0.0 | 5000 | 0.0364 | 0.0026 |  |
| adjust_pvalues holm FWER | m=20, 15 true nulls, equicorrelation 0.0 | 5000 | 0.0434 | 0.0029 |  |
| adjust_pvalues bh FDR | m=20, 15 true nulls, equicorrelation 0.0 | 5000 | 0.0365 | 0.0012 | bound 0.0375 |
| adjust_pvalues by FDR | m=20, 15 true nulls, equicorrelation 0.0 | 5000 | 0.0106 | 0.0008 | bound 0.0375 |
| adjust_pvalues bonferroni FWER | m=20, 15 true nulls, equicorrelation 0.5 | 5000 | 0.0252 | 0.0022 |  |
| adjust_pvalues holm FWER | m=20, 15 true nulls, equicorrelation 0.5 | 5000 | 0.0306 | 0.0024 |  |
| adjust_pvalues bh FDR | m=20, 15 true nulls, equicorrelation 0.5 | 5000 | 0.0338 | 0.0015 | bound 0.0375 |
| adjust_pvalues by FDR | m=20, 15 true nulls, equicorrelation 0.5 | 5000 | 0.0089 | 0.0007 | bound 0.0375 |
| PBO mean, pure noise | T=1000, N=10, S=16 | 200 | 0.5212 | 0.0161 | should be near 0.5 |
| PBO mean, pure noise | T=1000, N=50, S=16 | 200 | 0.4936 | 0.0122 | should be near 0.5 |
| PBO mean, pure noise | T=500, N=10, S=8 | 200 | 0.5143 | 0.0178 | should be near 0.5 |
| PBO mean, pure noise | T=1001, N=7, S=10 | 200 | 0.5141 | 0.0157 | should be near 0.5 |
| PBO mean, one real edge | SR 0.05/period, T=1000, N=10, S=10 | 200 | 0.3864 | 0.0157 |  |
| PBO mean, one real edge | SR 0.1/period, T=1000, N=10, S=10 | 200 | 0.1707 | 0.0129 |  |
| random_entry_test null | 1%/1% brackets | 400 | 0.0325 | 0.0089 | 199 sims, about 95 trades |
| side_test null | 1%/1% brackets | 400 | 0.0300 | 0.0085 | 199 sims |
| random_entry_test null | 3% stop / 0.5% target | 400 | 0.0300 | 0.0085 | 199 sims, about 95 trades |
| side_test null | 3% stop / 0.5% target | 400 | 0.0300 | 0.0085 | 199 sims |
| random_entry_test null | 0.5% stop / 3% target | 400 | 0.0300 | 0.0085 | 199 sims, about 95 trades |
| side_test null | 0.5% stop / 3% target | 400 | 0.0300 | 0.0085 | 199 sims |
| random_entry_test null | 1%/1%, costs | 400 | 0.0500 | 0.0109 | 199 sims, about 95 trades |
| side_test null | 1%/1%, costs | 400 | 0.0375 | 0.0095 | 199 sims |
| random_entry_test null | 1%/1%, max_bars=5 | 400 | 0.0375 | 0.0095 | 199 sims, about 95 trades |
| side_test null | 1%/1%, max_bars=5 | 400 | 0.0300 | 0.0085 | 199 sims |
| random_entry_test null | 1%/1%, gaps (gap vol 0.3%) | 400 | 0.0325 | 0.0089 | 199 sims, about 95 trades |
| side_test null | 1%/1%, gaps (gap vol 0.3%) | 400 | 0.0500 | 0.0109 | 199 sims |
| random_entry_test null | 1%/1%, drift +0.2%/bar | 400 | 0.0250 | 0.0078 | 199 sims, about 95 trades |
| random_entry_test null | side-dependent brackets, drift +0.2%/bar | 400 | 0.0300 | 0.0085 | 199 sims, about 95 trades |
| random_entry_test power | planted edge 0.05 | 200 | 0.1650 | 0.0263 | 1%/1% brackets, max_bars=1, about 300 trades |
| side_test power | planted edge 0.05 | 200 | 0.1900 | 0.0278 | 1%/1% brackets, max_bars=1, about 300 trades |
| random_entry_test power | planted edge 0.1 | 200 | 0.4000 | 0.0347 | 1%/1% brackets, max_bars=1, about 300 trades |
| side_test power | planted edge 0.1 | 200 | 0.3900 | 0.0346 | 1%/1% brackets, max_bars=1, about 300 trades |
| random_entry_test power | planted edge 0.2 | 200 | 0.8800 | 0.0230 | 1%/1% brackets, max_bars=1, about 300 trades |
| side_test power | planted edge 0.2 | 200 | 0.8850 | 0.0226 | 1%/1% brackets, max_bars=1, about 300 trades |
| random_entry_test power | planted edge 0.3 | 200 | 1.0000 | 0.0000 | 1%/1% brackets, max_bars=1, about 300 trades |
| side_test power | planted edge 0.3 | 200 | 0.9950 | 0.0050 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test resolved-only null | 1%/1% brackets | 1000 | 0.0330 | 0.0057 | about 195 trades; rejected without a warning 0.0330 |
| no_skill_test all-exits null | 1%/1% brackets | 1000 | 0.0330 | 0.0057 |  |
| no_skill_test resolved-only null | 3% stop / 0.5% target | 1000 | 0.0370 | 0.0060 | about 190 trades; rejected without a warning 0.0370 |
| no_skill_test all-exits null | 3% stop / 0.5% target | 1000 | 0.0370 | 0.0060 |  |
| no_skill_test resolved-only null | 0.5% stop / 3% target | 1000 | 0.0410 | 0.0063 | about 189 trades; rejected without a warning 0.0410 |
| no_skill_test all-exits null | 0.5% stop / 3% target | 1000 | 0.0410 | 0.0063 |  |
| no_skill_test resolved-only null | 1%/1%, costs | 1000 | 0.0330 | 0.0057 | about 195 trades; rejected without a warning 0.0330 |
| no_skill_test all-exits null | 1%/1%, costs | 1000 | 0.0330 | 0.0057 |  |
| no_skill_test resolved-only null | 3% stop / 0.5% target, max_bars=5 | 1000 | 0.2540 | 0.0138 | about 192 trades; rejected without a warning 0.0050 |
| no_skill_test all-exits null | 3% stop / 0.5% target, max_bars=5 | 1000 | 0.0390 | 0.0061 |  |
| no_skill_test resolved-only null | 0.5% stop / 3% target, max_bars=5 | 1000 | 0.0030 | 0.0017 | about 191 trades; rejected without a warning 0.0000 |
| no_skill_test all-exits null | 0.5% stop / 3% target, max_bars=5 | 1000 | 0.0420 | 0.0063 |  |
| no_skill_test resolved-only null | 0.5% stop / 3% target, max_bars=5, costs | 1000 | 0.0030 | 0.0017 | about 191 trades; rejected without a warning 0.0000 |
| no_skill_test all-exits null | 0.5% stop / 3% target, max_bars=5, costs | 1000 | 0.0420 | 0.0063 |  |
| no_skill_test resolved-only null | 1%/1%, max_bars=1 | 1000 | 0.0330 | 0.0057 | about 200 trades; rejected without a warning 0.0000 |
| no_skill_test all-exits null | 1%/1%, max_bars=1 | 1000 | 0.0200 | 0.0044 |  |
| no_skill_test resolved-only null | 1%/1%, gaps (gap vol 0.3%) | 1000 | 0.0240 | 0.0048 | about 196 trades; rejected without a warning 0.0020 |
| no_skill_test all-exits null | 1%/1%, gaps (gap vol 0.3%) | 1000 | 0.0240 | 0.0048 |  |
| no_skill_test resolved-only null | 0.5% stop / 3% target, gaps (gap vol 1%) | 1000 | 0.0530 | 0.0071 | about 133 trades; rejected without a warning 0.0000 |
| no_skill_test all-exits null | 0.5% stop / 3% target, gaps (gap vol 1%) | 1000 | 0.0540 | 0.0072 |  |
| no_skill_test resolved-only null | 0.3% stop / 5% target, about 19 trades | 1000 | 0.0330 | 0.0057 | about 19 trades; rejected without a warning 0.0320 |
| no_skill_test all-exits null | 0.3% stop / 5% target, about 19 trades | 1000 | 0.0360 | 0.0059 |  |
| no_skill_test resolved-only null | 0.5% stop / 3% target, about 38 trades | 1000 | 0.0300 | 0.0054 | about 38 trades; rejected without a warning 0.0300 |
| no_skill_test all-exits null | 0.5% stop / 3% target, about 38 trades | 1000 | 0.0340 | 0.0057 |  |
| no_skill_test resolved-only power | planted edge 0.05 | 300 | 0.1167 | 0.0186 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test all-exits power | planted edge 0.05 | 300 | 0.0900 | 0.0166 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test resolved-only power | planted edge 0.1 | 300 | 0.3100 | 0.0267 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test all-exits power | planted edge 0.1 | 300 | 0.2567 | 0.0253 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test resolved-only power | planted edge 0.2 | 300 | 0.8167 | 0.0224 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test all-exits power | planted edge 0.2 | 300 | 0.8067 | 0.0228 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test resolved-only power | planted edge 0.3 | 300 | 0.9867 | 0.0066 | 1%/1% brackets, max_bars=1, about 300 trades |
| no_skill_test all-exits power | planted edge 0.3 | 300 | 0.9867 | 0.0066 | 1%/1% brackets, max_bars=1, about 300 trades |

## Formulas checked against their sources

* Probabilistic Sharpe ratio: Bailey & Lopez de Prado (2012), "The Sharpe
  Ratio Efficient Frontier", eq. (11), p. 9 of the davidhbailey.com preprint.
  The variance factor `1 - skew*SR + (kurtosis - 1)/4 * SR^2` uses raw
  kurtosis and the per-period Sharpe ratio (footnote 5, p. 9; eq. (8) on
  p. 8 states it with `(kurtosis - 3)/4` plus `SR^2 / 2`, which is the same
  thing). `probabilistic_sharpe_ratio` reproduces the paper's p. 10 example:
  0.982 under normality and 0.913 with skew -2.448 and kurtosis 10.164 for
  SR 0.458 and n = 24, and 0.953 with n = 36.
* Minimum track record length: same paper, eq. (13), p. 11.
  `min_track_record_length` gives 59.895 months for the appendix example
  on p. 20 (monthly SR 2/sqrt(12) against 1/sqrt(12), skew -0.72, kurtosis
  5.78, 95%), as printed there.
* Expected maximum Sharpe ratio and the deflated Sharpe ratio: Bailey &
  Lopez de Prado (2014), "The Deflated Sharpe Ratio", eq. (1), p. 7, and
  eq. (2), p. 8. The library assumes the trials' mean Sharpe ratio is zero,
  as in eq. (2). The p. 10 example reproduces: SR0 = 0.1132, DSR 0.9004 for
  N = 100, 0.9505 for N = 46, and 0.9505 for N = 88 with normal returns.
* Minimum backtest length: Bailey, Borwein, Lopez de Prado & Zhu (2014),
  "Pseudo-Mathematics and Financial Charlatanism", Theorem 3.1 and
  eq. (3.2), p. 11 of the April 2014 preprint; the expected maximum is
  Proposition 2.1, eq. (2.4), p. 9. `min_backtest_length(45, 1.0)` is 4.998
  years, matching the "5 years, no more than 45 configurations" example, and
  `expected_max_sharpe(10, 1.0)` is 1.575 against the paper's 1.57.
* PBO by CSCV: Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of
  Backtest Overfitting", Algorithm 2.3, pp. 11-12 (November 2014 and February
  2015 preprints): relative rank `rank / (N + 1)`, logit, and PBO as the
  share of logits below zero (section 3.1, p. 13); the degradation regression
  and probability of loss are section 3.2, p. 14. The paper prints 12,780
  combinations for S = 16; C(16, 8) is 12,870, which the code uses.
* Harvey & Liu (2015), "Backtesting", Journal of Portfolio Management:
  eq. (3)-(5), p. 14. With the normal distribution, T = 240 months, annual
  SR 0.75 and N = 200, `haircut_sharpe` gives p = 0.0008, p^M = 0.159 and a
  haircut Sharpe ratio of 0.315 (the paper rounds to 0.15 and 0.32). The
  Bonferroni and Holm examples on p. 15 reproduce exactly with
  `adjust_pvalues`. The BHY example on p. 16 does not for the two largest
  p-values (0.06 in the paper, 0.132 and 0.147 here): the paper's recursion
  starts from the unscaled largest p-value, while Benjamini & Yekutieli
  (2001) apply the BH step-up at level `q / c(m)`, which multiplies every
  adjusted p-value, including the largest, by `c(m)`. The code follows
  Benjamini & Yekutieli; for the first four p-values both give 0.0496.
* Lo (2002), "The Statistics of Sharpe Ratios": `eta(q) = q / sqrt(q + 2
  sum_{k=1}^{q-1} (q - k) rho_k)` is eq. (20), on p. 5 of the copy hosted at
  traders.studentorg.berkeley.edu (FAJ 58(4)).
* Newey & West (1994), NBER technical working paper 144: Bartlett weights
  `1 - j/(m + 1)` (p. 17), and the lag-selection parameter
  `n = 4 (T/100)^(2/9)` for the Bartlett kernel (Table II, panel C,
  discussed on p. 18). In that paper n is a pilot value from which the
  bandwidth `gamma * T^(1/3)` is estimated (eqs. (3-10) to (3-13), p. 12);
  `newey_west` uses n itself as the lag, a common rule of thumb, and its
  docstring now says so. In a separate run, using the full automatic
  bandwidth instead lowered the AR(1), coefficient 0.6, T = 1000 rejection
  rate from 10.0% to 7.3% but left T = 250 at 11.0%.
* Phipson & Smyth (2010): the Monte Carlo p-value `(b + 1)/(m + 1)` is the
  exact p-value when the m null draws are independent (arXiv preprint
  1603.05766, section 4, p. 6). The random-entry and side tests count null
  values within floating-point rounding of the observed value as ties
  against the strategy, which keeps them at or above that value.
* Holm (1979), Benjamini & Hochberg (1995) and Benjamini & Yekutieli (2001)
  were checked against the procedures as restated by Harvey & Liu (2015,
  pp. 15-16) and by simulation above, not against the original papers,
  which were not retrieved. The Newey & West (1987) and Hansen & Hodrick
  (1980) papers were also not retrieved; the Bartlett and uniform weights
  are checked against Newey & West (1994) and by the overlapping-returns
  simulations above.

## Known limitations shown by the table

* The resolved-only `no_skill_test` p-value is not a valid no-skill test
  when trades time out; read `all_exits_p_value` whenever `warnings` mention
  time exits.
* `no_skill_test` assumes a driftless continuous mid price. Gaps through the
  target are filled at the target, so a gap raises the chance that a far
  target is reached before a near stop. With gap volatility equal to the bar
  volatility, a 0.5% stop and a 3% target, the rates were 5.3% and 5.4%; with
  gap volatility of 0.3% and 1% brackets they were 2.4%.
* `side_test` becomes very conservative when large gaps make reversed setups
  unfillable: in a separate run of 600 replications with gap volatility
  equal to the bar volatility and a 3% stop with a 0.5% target, it rejected
  0.2% of null runs.
* PSR and DSR are asymptotic. With fewer than a few hundred fat-tailed or
  skewed returns PSR over-rejects; DSR is far below its nominal size and
  needs a large edge.
