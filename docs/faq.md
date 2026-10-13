# FAQ

## Why is my 85% win rate not an edge?

Because the bracket gives most of it away for free. If the price has no drift, a trade reaches its target before its stop with probability `a / (a + b)`, where `a` is the distance from the fill to the stop and `b` the distance to the target. A stop five times farther away than the target wins 83.3% of the time with random entries. Costs then push the win rate needed to break even above that.

`examples/random_strategy_audit.py` runs exactly this case: random entries, stop 0.40% and target 0.08% from the signal close, with spread, slippage and fees. On 968 trades it wins 84.0% of the time, against a no-skill rate of 83.3% and a break-even rate of 86.8%. `no_skill_test` gives a one-sided p-value of 0.31 for the win rate, the margin over break-even is -2.8 points, and the expectancy is -0.0336R per trade.

Run `no_skill_test(result)` and read three numbers: `no_skill_rate` (what the brackets give without skill), `breakeven_rate` (what you need after costs) and your `win_rate`. The derivation is in [fill-rules.md](fill-rules.md#the-no-skill-win-rate).

## Why do my results drop with `entry_delay=1`?

`entry_delay=1` fills one bar later: at the open of bar `i + 2` for a decision at the close of bar `i`. Stop and target levels stay where the strategy put them. A large drop has two possible causes:

1. The edge really lives in the first bar after the signal. Then the strategy needs to fill at the next open, and you should check that you can.
2. The signal uses information from the signal bar that was not available at its close, or from later bars. Run the lookahead tests (`truncation_test`, `perturbation_test`, `check_factory`) on the code that produces the signal.

`entry_delay_test` cannot tell these apart. On `synthetic.planted_signal_bars` with `edge=0.5`, where the planted drift lasts exactly one bar, the brackets from the random-strategy example give +0.0461R per trade over 973 trades at delay 0 and -0.0568R over 421 trades at delay 1, and the test reports a collapse at delay 1. There is no leak in that data. The trade count also falls: the target is 0.08% away and each bar moves 0.2%, so after an extra bar the open is often already past the target and the engine skips the setup.

## What do I pass as `n_trials` for the deflated Sharpe ratio?

The number of independent strategy configurations you tried before picking this one, including the ones you threw away. Every parameter set, feature set, filter or variant you backtested on the same data counts. `var_trials_sharpe` is the variance of those trials' Sharpe ratios, per period (the same units as the Sharpe ratio you test).

Nobody remembers these numbers afterwards, so record them as you go:

```python
from backtest_skeptic import TrialRegistry, deflated_sharpe_ratio

reg = TrialRegistry("trials.jsonl")
for params in grid:
    r = backtest(train, **params).r
    reg.record("breakout", r, params=params)

deflated_sharpe_ratio(best_returns, n_trials=reg.n_trials,
                      var_trials_sharpe=reg.sharpe_variance)
```

Both arguments are required; there is no default, because a deflated Sharpe ratio computed without them would be a plain probabilistic Sharpe ratio under a different name. If your trials are strongly correlated (a lookback of 20 and of 21), the effective number of independent trials is smaller than the count, and using the full count makes the result conservative. Trials you ran but did not record make it optimistic.

To convert annual figures: divide an annual Sharpe ratio by `sqrt(periods_per_year)` and an annual variance of Sharpe ratios by `periods_per_year`.

## Why random entries instead of shuffling returns?

Each null changes one thing and keeps the rest. Random entries keep the real price path, the number of trades, the long/short mix, the bracket distances, the costs and the exit rules, and change only when the trades are entered. That isolates the one thing an entry signal is supposed to contribute.

Shuffling the order of the strategy's trade results leaves their mean unchanged, so it cannot test whether the expectancy is real. Shuffling the market's bar returns builds a different market: it removes trends and volatility clustering, so the null differs from the strategy's market in more than timing, and the comparison credits or blames the signal for those differences.

`random_entry_test` copies each strategy trade's stop and target as fractions of the signal bar's close, resolves the random trades with the engine's fill rules, and ranks the strategy's metric against the random ones. Its limitations are in [checks.md](checks.md#random-entry-nulls).

## Can I use this with vectorbt, backtesting.py or my own engine?

Yes. The checks split into three groups by what they need.

**Any callable:** the lookahead tests only call your function. `truncation_test` and `perturbation_test` take any `fn(bars)` that returns one row per bar (numpy, pandas, or a wrapper around another library). Load your prices with `Bars.from_csv`, `Bars.from_records` or `Bars.from_dataframe`.

**Any trade log:** `audit_fills` reads a log from any backtester. Export the log to CSV and load it with `TradeLog.from_csv`, or pass a list of dicts:

```python
from backtest_skeptic import Bars, Costs, TradeLog, audit_fills

bars = Bars.from_csv("prices.csv")
log = TradeLog.from_csv(
    "trades.csv",
    bars,
    columns={"entry_time": "EntryTime", "exit_time": "ExitTime",
             "entry_price": "AvgEntry", "exit_price": "AvgExit"},
)
print(audit_fills(bars, log, costs=Costs(half_spread=0.01)))
```

Required per trade: side (`+1`/`-1`, `BUY`/`SELL` or `long`/`short`), entry and exit price, and the entry and exit bar as indices or as timestamps that match `bars.time`. Stop, target, signal bar and exit reason are optional, and each one enables more checks. Common column names (`stop_loss`, `sl`, `take_profit`, `tp`, `direction`, `exit_reason`) are recognised; `columns` maps the rest. Pass the costs your backtester charged.

**`Trade` objects:** `no_skill_test`, `random_entry_test`, `side_test` and the robustness checks take `Trade` objects or a `BacktestResult`. The simplest route is to run your signals through `run_signals` with the same costs. `random_entry_test` re-resolves each trade through the engine and raises if one does not reproduce; pass `verify=False` for `Trade` objects built from another engine's log. `no_skill_test` checks each trade's prices, fees and R against the costs and raises on a mismatch. If your engine's results differ from `run_signals`, `audit_fills` on your engine's log usually shows why.

**Return series:** the functions in `stats` and `spanning` take plain arrays of per-period returns from anywhere.
