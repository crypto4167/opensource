"""backtest-skeptic: find out whether a backtest's edge is real.

The package bundles the checks that catch the usual ways a backtest lies:

* :mod:`~backtest_skeptic.engine` - a pessimistic bracket-order engine with
  explicit fill rules, shared by strategies and their controls.
* :mod:`~backtest_skeptic.lookahead` - behavioural tests that catch signals
  reading future bars.
* :mod:`~backtest_skeptic.random_entry` - matched random-entry nulls.
* :mod:`~backtest_skeptic.baseline` - the no-skill win rate of a bracket and
  a test against it.
* :mod:`~backtest_skeptic.fills` - fill-realism audit of any trade log.
* :mod:`~backtest_skeptic.robustness` - concentration, fragility and
  stability heuristics.
* :mod:`~backtest_skeptic.stats` - probabilistic and deflated Sharpe ratio,
  PBO, Newey-West and multiple-testing haircuts.
* :mod:`~backtest_skeptic.holdout` - a read-once holdout ledger and a trial
  registry.
* :mod:`~backtest_skeptic.spanning` - does a known factor explain the returns?
"""

from .data import Bars, DataError
from .engine import BacktestResult, resolve_bracket, run, run_signals
from .metrics import Summary, summarize
from .trades import LONG, SHORT, Costs, Setup, Trade

__version__ = "0.1.0"

__all__ = [
    "LONG",
    "SHORT",
    "BacktestResult",
    "Bars",
    "Costs",
    "DataError",
    "Setup",
    "Summary",
    "Trade",
    "__version__",
    "resolve_bracket",
    "run",
    "run_signals",
    "summarize",
]
