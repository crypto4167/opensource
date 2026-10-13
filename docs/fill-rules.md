# Fill rules

`backtest_skeptic.engine` simulates bracket orders (an entry with a fixed stop and a fixed target) bar by bar. The same code scores a strategy (`run`, `run_signals`) and its random-entry controls (`resolve_many`), so the strategy and the null are filled identically. The rules are chosen so that, when OHLC data cannot say what happened, the engine assumes the outcome worse for the trader.

## The rules

### 1. Decide after the close, using only the past

`decide(i, history)` runs after bar `i` closes, and `history` is `bars.prefix(i + 1)`: bars `0..i` only. `run_signals` takes precomputed arrays, where `side[i]`, `stop[i]` and `target[i]` describe a decision made at the close of bar `i`.

**Why:** a decision cannot use a bar that has not finished. Passing only the prefix makes it impossible for `decide` to index into the future by accident. It does not stop a closure from reading an array computed on the full data; `lookahead.check_factory` tests for that, and `truncation_test` and `perturbation_test` test the functions that build precomputed arrays.

### 2. Fill at the next bar's open

An order decided at the close of bar `i` fills at the open of bar `i + 1` (bar `i + 1 + entry_delay` with a delay). The fill price is the open plus half the spread plus slippage, against the trader:

```
entry_price = open[i + 1] + side * (half_spread + slippage)
```

**Why:** the close of bar `i` is the price the signal was computed from. Filling at that same print assumes the order was computed and executed in zero time. Filling at a price inside bar `i` is worse: that price traded before the signal existed. `fills.audit_fills` flags both (`filled_at_signal_close`, `phantom_signal_bar_fill`).

### 3. Skip setups the open has already invalidated

If the fill bar's open (the mid price, before costs) is at or beyond the stop, the setup is skipped as `stop_beyond_entry`. If the fill leaves no room before the target, it is skipped as `target_beyond_entry`. Skips are counted in `BacktestResult.skipped`, together with `no_bar_to_fill` and `invalid_levels` (non-finite levels or an unknown side).

**Why:** an order whose stop is already hit at entry is not a trade, and filling it would book an instant loss or a negative risk. The stop test uses the mid open so that the spread cannot lift a fill back above a stop the market has opened through and fake a tiny risk.

### 4. Gapped stops fill at the open

From the entry bar on, each bar is checked for the stop and the target. A stop the market gapped through fills at the bar's open, not at the stop price. A target fills at the target price, even after a favourable gap.

**Why:** a stop order becomes a market order at the first price through it, which after a gap is the open. Filling it at the stop price hides the gap loss; `fills.audit_fills` reports this as `gap_stop_fill_too_good`. A target is a resting limit order, which fills at its price; crediting the better open after a favourable gap would be optimistic, so the engine does not.

### 5. A bar that touches both levels books the stop

When one bar touches both the stop and the target, its open decides if it can: a bar that opened beyond the stop hit the stop first, and one that opened beyond the target hit the target first. A bar that opened between the levels and touched both is ambiguous: OHLC data cannot say which came first. The engine books the stop (`same_bar="stop"`, the default) and sets `Trade.ambiguous`.

**Why:** booking the target on such bars inflates the win rate of tight brackets, where these bars are common. `same_bar="target"` exists for measuring how much the result depends on the choice; `summarize()` counts ambiguous trades and `no_skill_test` warns when wins come from them.

### 6. Time exits and the last bar

`max_bars` closes the position at the close of its `max_bars`-th bar, counting the entry bar (exit reason `time`). A position still open at the end of the data closes at the last close (reason `end`), after the last bar has been checked for the stop and the target.

**Why:** a loop that stops one bar early never checks the last bar and books an end-of-data exit that the bracket would have closed. `fills.audit_fills` reports it as `missed_earlier_touch`.

### 7. Exit costs

Every exit pays half the spread. Stop, time and end-of-data exits also pay slippage; target exits do not, since they are limit orders. Fees are `fee_rate` times the notional of each fill, on entry and on exit:

```
fees = fee_rate * (abs(entry_price) + abs(exit_price))
r    = (side * (exit_price - entry_price) - fees) / risk
risk = side * (entry_price - stop)
```

### 8. One position, fixed risk

One position at a time, a fixed risk of 1R per trade, no compounding. After a trade exits at bar `j`, the strategy may decide again at the close of bar `j`.

### Optional filter: `risk_band`

`risk_band=(low, high)` skips a setup when the risk measured from the actual fill falls outside `low..high` times the risk measured from the signal bar's close. It is off by default because it selects trades on the fill bar's open, which biases the no-skill baseline when that baseline is anchored anywhere other than the fill.

## The no-skill win rate

For a trade on side `s` (`+1` long, `-1` short) that filled at mid price `m` (the fill bar's open before costs, `Trade.mid_entry`), let

- `a = s * (m - stop)`, the distance from the fill to the stop, and
- `b = s * (target - m)`, the distance from the fill to the target.

If the mid price is a driftless continuous martingale from the fill on, it reaches the target before the stop with probability

```
p0 = a / (a + b)
```

This is the fair-game ruin probability (Feller, vol. 1, ch. XIV). For a continuous martingale it follows from optional stopping: the stopped price has mean `m` and can only end at the stop or at the target, so `p0 * (m + s*b) + (1 - p0) * (m - s*a) = m`.

A stop four times farther than the target therefore wins 80% of the time with no skill. `synthetic.random_walk_bars` generates prices with this property (highs and lows come from a continuous path inside each bar), and `examples/random_strategy_audit.py` shows random entries reaching the rate their brackets predict.

The baseline is anchored at the fill bar's open, not at the signal bar's close. The engine brackets from the fill, and when trades are selected on the fill-bar open (the skip rule above, or `risk_band`), a signal-close baseline turns that selection into apparent skill.

## Costs and the break-even rate

Costs move the break-even point, not the no-skill rate. With half-spread `h`, slippage `sl`, fee rate `f`, entry fill `e` (`Trade.entry_price`) and risk `R` (`Trade.risk`), the engine books a target hit and a stop hit without a gap as:

```
r_win  = (s * (target - s*h - e)    - f * (|e| + |target - s*h|))    / R
r_loss = (s * (stop - s*(h+sl) - e) - f * (|e| + |stop - s*(h+sl)|)) / R
```

A trader who wins with probability `p` breaks even at

```
p* = -r_loss / (r_win - r_loss)
```

Without costs `p* = a / (a + b)`. With costs `p*` is higher, so there are two thresholds:

- `win_rate > no_skill_rate` means the entries predict direction better than chance, and
- `win_rate > breakeven_rate` means the trades make money after costs.

At the no-skill rate a trader loses about one round-trip cost per trade, exactly `-(2h + sl*(a + 2b)/(a + b)) / R` without fees.

### Worked example

A long bracket filled at a mid price of 100, stop at 99 (`a = 1`), target at 100.25 (`b = 0.25`). Computed with `baseline.bracket_odds` on a trade from `resolve_bracket`:

| Costs | No-skill rate | Break-even rate | Win | Loss | No-skill expectancy |
| --- | --- | --- | --- | --- | --- |
| none | 80.0% | 80.0% | +0.250R | -1.000R | +0.0000R |
| `half_spread=0.01`, `slippage=0.005` | 80.0% | 82.1% | +0.222R | -1.015R | -0.0256R |
| the same plus `fee_rate=0.0001` | 80.0% | 83.7% | +0.202R | -1.034R | -0.0453R |

A strategy with these brackets and the second row's costs needs a win rate above 82.1% to make money. An 81% win rate would be skill (above 80%) that does not pay for trading.

`baseline.breakeven_win_rate(risk, reward, cost_r=...)` gives the planning version, `(1 + cost_r) * risk / (risk + reward)`, where `cost_r` is the round-trip cost in units of risk charged equally on wins and losses. For trades from a backtest, `no_skill_test` computes the exact per-trade figure from the fills.

### What the closed form leaves out

- **Drift.** A trend gives a long-only strategy a win rate above `a / (a + b)` with no timing skill. Use `random_entry_test` on the same data, which shares the drift with the null.
- **Gaps.** A stop that gaps fills at the open and loses more than `r_loss`, so the true break-even rate is higher than reported. `no_skill_test` counts these as `gap_fills`.
- **Ambiguous bars.** Booking the stop on bars that touch both levels biases the observed win rate down.
- **Time and end exits.** These reach neither level and are excluded from the win/loss test; `no_skill_test` reports how many and estimates the bias (`exclusion_bias`, `all_exits_edge`).
