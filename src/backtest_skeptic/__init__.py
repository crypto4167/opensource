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
* :mod:`~backtest_skeptic.report` - every check the inputs support, in one
  call (:func:`audit`), also available as the ``backtest-skeptic`` command.
"""

from . import (
    baseline,
    fills,
    holdout,
    lookahead,
    random_entry,
    report,
    robustness,
    spanning,
    stats,
    synthetic,
)
from .baseline import no_skill_test
from .data import Bars, DataError
from .engine import BacktestResult, resolve_bracket, resolve_many, run, run_signals
from .fills import TradeLog, audit_fills
from .holdout import HoldoutLedger, TrialRegistry, split_holdout
from .lookahead import check_factory, perturbation_test, truncation_test
from .metrics import Summary, summarize
from .random_entry import random_entry_test, side_test
from .report import AuditReport, audit
from .robustness import block_bootstrap, concentration, cost_stress, entry_delay_test
from .spanning import spanning_regression
from .stats import deflated_sharpe_ratio, pbo, probabilistic_sharpe_ratio
from .trades import LONG, SHORT, Costs, Setup, Trade

__version__ = "0.1.0"

__all__ = [
    "LONG",
    "SHORT",
    "AuditReport",
    "BacktestResult",
    "Bars",
    "Costs",
    "DataError",
    "HoldoutLedger",
    "Setup",
    "Summary",
    "Trade",
    "TradeLog",
    "TrialRegistry",
    "__version__",
    "audit",
    "audit_fills",
    "baseline",
    "block_bootstrap",
    "check_factory",
    "concentration",
    "cost_stress",
    "deflated_sharpe_ratio",
    "entry_delay_test",
    "fills",
    "holdout",
    "lookahead",
    "no_skill_test",
    "pbo",
    "perturbation_test",
    "probabilistic_sharpe_ratio",
    "random_entry",
    "random_entry_test",
    "report",
    "resolve_bracket",
    "resolve_many",
    "robustness",
    "run",
    "run_signals",
    "side_test",
    "spanning",
    "spanning_regression",
    "split_holdout",
    "stats",
    "summarize",
    "synthetic",
    "truncation_test",
]
