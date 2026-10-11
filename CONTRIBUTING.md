# Contributing

Thanks for helping. This project exists to catch backtests that look better than they are, so the bar for correctness is high and the bar for new features is evidence.

## Setup

```bash
git clone https://github.com/crypto4167/opensource
cd opensource
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check src tests examples
ruff format src tests examples
mypy
pytest -q
```

All four must pass. CI runs them on Python 3.10 to 3.13 and on the oldest supported numpy.

## Rules for new checks

1. **Negative and positive controls.** Every check needs a test where it must report nothing (a driftless random walk from `backtest_skeptic.synthetic.random_walk_bars`, or other null data) and a test where it must catch the problem (`planted_signal_bars`, `oracle_signal`, or a deliberately broken input). A check that never fails proves nothing.
2. **Cite real sources.** Statistical methods need a citation to the paper the formula comes from, and the test suite should reproduce a published number where the paper gives one. Methods with no canonical source are labelled as heuristics in their docstring.
3. **No silent fallbacks.** Bad input raises `ValueError` with a clear message. A function never quietly computes something other than what its name says (for example, a deflated Sharpe ratio with no trial count must raise, not return the probabilistic Sharpe ratio).
4. **Clean-room implementations.** Do not copy code from copyleft or proprietary projects (pypbo and backtesting.py are AGPL, Freqtrade is GPL, mlfinlab is proprietary). Implement from the published formulas.
5. **numpy is the only runtime dependency.** Optional integrations can go behind extras, but the core must install with numpy alone.

## Reporting a wrong result

If a check gives an answer you believe is wrong, open an issue with the smallest input that reproduces it and the answer you expected. Correctness bugs take priority over everything else.
