"""Command-line interface: ``backtest-skeptic <command> ...``.

Commands
--------
``audit``
    Load bars and a trade log from CSV files and print
    :func:`backtest_skeptic.report.audit`'s report.
``lookahead``
    Import a signal function and run the truncation and perturbation tests
    from :mod:`backtest_skeptic.lookahead` on it.
``dsr``
    Probabilistic and deflated Sharpe ratio, expected maximum Sharpe ratio
    and minimum track record length from summary numbers.
``selftest``
    Check that the lookahead and random-entry harnesses catch planted
    problems on synthetic data.

Exit codes: 0 when the command found nothing at or above its failure level,
1 when it did, 2 for usage errors and unreadable or inconsistent data (the
error message goes to stderr, without a traceback).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from . import __version__, lookahead, random_entry, stats
from .data import Bars, DataError
from .fills import TradeLog
from .report import STATUSES, audit
from .synthetic import random_walk_bars
from .trades import Costs

__all__ = ["main"]

EXIT_OK = 0
EXIT_FOUND = 1
EXIT_ERROR = 2


class _UsageError(Exception):
    """Bad arguments or unreadable inputs: exit code 2."""


def _non_negative(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be a finite number >= 0, got {text!r}")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text!r}")
    return value


def _finite(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite, got {text!r}")
    return value


# -------------------------------------------------------------------- audit


def _load_bars(path: str, time_column: str | None) -> Bars:
    if not Path(path).is_file():
        raise _UsageError(f"bars file not found: {path}")
    return Bars.from_csv(path, time_column=time_column)


def _cmd_audit(args: argparse.Namespace) -> int:
    bars = _load_bars(args.bars, args.time_column)
    if not Path(args.trades).is_file():
        raise _UsageError(f"trades file not found: {args.trades}")
    log = TradeLog.from_csv(args.trades, bars)
    costs = Costs(half_spread=args.half_spread, fee_rate=args.fee_rate, slippage=args.slippage)
    report = audit(
        bars,
        log,
        costs=costs,
        max_bars=args.max_bars,
        same_bar=args.same_bar,
        n_sims=args.sims,
        seed=args.seed,
        periods_per_year=args.periods_per_year,
    )
    if args.format == "json":
        print(report.to_json())
    elif args.format == "markdown":
        print(report.to_markdown(), end="")
    else:
        print(report.to_text())
    if args.fail_on == "never":
        return EXIT_OK
    level = STATUSES.index("FAIL" if args.fail_on == "fail" else "WARN")
    found = any(STATUSES.index(c.status) >= level for c in report.checks)
    return EXIT_FOUND if found else EXIT_OK


# ---------------------------------------------------------------- lookahead


def _import_target(spec: str) -> Callable[[Bars], Any]:
    """``module:function`` or ``path/to/file.py:function``."""
    module_name, sep, func_name = spec.rpartition(":")
    if not sep or not module_name or not func_name:
        raise _UsageError(f"expected MODULE:FUNCTION or FILE.py:FUNCTION, got {spec!r}")
    module: ModuleType
    if module_name.endswith(".py") or os.sep in module_name or "/" in module_name:
        path = Path(module_name)
        if not path.is_file():
            raise _UsageError(f"module file not found: {module_name}")
        name = f"_backtest_skeptic_target_{path.stem}"
        loader_spec = importlib.util.spec_from_file_location(name, path)
        if loader_spec is None or loader_spec.loader is None:
            raise _UsageError(f"cannot import {module_name}")
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[name] = module
        try:
            loader_spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise _UsageError(f"importing {module_name} failed: {exc!r}") from None
    else:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise _UsageError(f"importing {module_name} failed: {exc!r}") from None
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise _UsageError(f"{module_name} has no callable named {func_name!r}")
    out: Callable[[Bars], Any] = fn
    return out


def _cmd_lookahead(args: argparse.Namespace) -> int:
    fn = _import_target(args.target)
    bars = _load_bars(args.bars, args.time_column)
    reports: list[lookahead.LookaheadReport] = []
    try:
        if args.mode in ("truncation", "both"):
            reports.append(lookahead.truncation_test(fn, bars, min_history=args.min_history))
        if args.mode in ("perturbation", "both"):
            reports.append(lookahead.perturbation_test(fn, bars))
    except Exception as exc:
        # The tests catch errors on probes; an error on the full data reaches here.
        raise _UsageError(f"{args.target} failed on the full data: {exc!r}") from None
    print("\n".join(str(r) for r in reports))
    return EXIT_OK if all(r.passed for r in reports) else EXIT_FOUND


# ---------------------------------------------------------------------- dsr


def _cmd_dsr(args: argparse.Namespace) -> int:
    if args.annualized and args.periods_per_year is None:
        raise _UsageError("--annualized needs --periods-per-year")
    ppy = args.periods_per_year
    if ppy is not None and ppy <= 0:
        raise _UsageError(f"--periods-per-year must be positive, got {ppy:g}")
    sr = args.sharpe
    var = args.trials_variance
    if var < 0:
        raise _UsageError(f"--trials-variance must be >= 0, got {var:g}")
    if args.annualized:
        sr = stats.deannualize_sharpe(sr, ppy)
        var = var / ppy
    common = {"sr": sr, "n": args.n_obs, "skew": args.skew, "kurtosis": args.kurtosis}
    psr = stats.probabilistic_sharpe_ratio(**common)
    dsr = stats.deflated_sharpe_ratio(**common, n_trials=args.trials, var_trials_sharpe=var)
    emax = stats.expected_max_sharpe(args.trials, var)
    mintrl = stats.min_track_record_length(sr, 0.0, args.skew, args.kurtosis)
    mintrl_emax = stats.min_track_record_length(sr, emax, args.skew, args.kurtosis)
    out: dict[str, Any] = {
        "sharpe_per_period": sr,
        "n_obs": args.n_obs,
        "skew": args.skew,
        "kurtosis": args.kurtosis,
        "trials": args.trials,
        "trials_variance_per_period": var,
        "psr": psr,
        "dsr": dsr,
        "expected_max_sharpe_per_period": emax,
        "min_track_record_length": None if math.isinf(mintrl) else mintrl,
        "min_track_record_length_vs_expected_max": (
            None if math.isinf(mintrl_emax) else mintrl_emax
        ),
        "periods_per_year": ppy,
    }
    if ppy is not None:
        out["expected_max_sharpe_annual"] = stats.annualize_sharpe(emax, ppy)
        out["min_track_record_years"] = None if math.isinf(mintrl) else mintrl / ppy
    if args.format == "json":
        print(json.dumps(out, indent=2))
        return EXIT_OK

    def trl(value: float) -> str:
        if math.isinf(value):
            return "infinite (the Sharpe ratio is not above the benchmark)"
        text = f"{value:.1f} observations"
        if ppy is not None:
            text += f" ({value / ppy:.2f} years)"
        return text

    lines = [
        f"Sharpe ratio per period:           {sr:.6f}",
        f"Observations:                      {args.n_obs:g}",
        f"Skew / raw kurtosis:               {args.skew:g} / {args.kurtosis:g}",
        f"Trials / variance per period:      {args.trials:g} / {var:.6g}",
        f"PSR (against 0):                   {psr:.6f}",
        f"Expected max Sharpe of the trials: {emax:.6f} per period"
        + ("" if ppy is None else f" ({out['expected_max_sharpe_annual']:.4f} annual)"),
        f"DSR (against the expected max):    {dsr:.6f}",
        f"MinTRL at 95% (against 0):         {trl(mintrl)}",
        f"MinTRL at 95% (against the max):   {trl(mintrl_emax)}",
    ]
    print("\n".join(lines))
    return EXIT_OK


# ----------------------------------------------------------------- selftest


def _cmd_selftest(args: argparse.Namespace) -> int:
    results = dict(lookahead.self_test(seed=args.seed))
    bars = random_walk_bars(2000, seed=args.seed)
    control = random_entry.positive_control(bars, n_sims=199, seed=args.seed)
    assert control.p_value is not None
    results["random_entry_positive_control"] = control.p_value < 0.01
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"{len(failed)} of {len(results)} probes failed: {', '.join(failed)}")
        return EXIT_FOUND
    print(f"All {len(results)} probes passed.")
    return EXIT_OK


# ------------------------------------------------------------------- parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest-skeptic",
        description="Checks that tell you whether a trading backtest's edge is real.",
        epilog="Exit codes: 0 nothing found, 1 a problem found, 2 usage or data error.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p = sub.add_parser(
        "audit",
        help="audit a trade log against its bars",
        description=(
            "Run every check the inputs support: fill realism, no-skill baseline, "
            "random-entry null, concentration, stability and Sharpe statistics. Pass the "
            "costs and engine settings the backtest used."
        ),
    )
    p.add_argument("--bars", required=True, help="CSV with open, high, low, close columns")
    p.add_argument(
        "--trades",
        required=True,
        help="CSV trade log: side, entry_price, exit_price, entry/exit bar index or time; "
        "optional signal_index, stop, target, reason",
    )
    p.add_argument(
        "--half-spread",
        type=_non_negative,
        default=0.0,
        metavar="X",
        help="half the bid/ask spread, in price units, paid on entry and exit (default 0)",
    )
    p.add_argument(
        "--fee-rate",
        type=_non_negative,
        default=0.0,
        metavar="X",
        help="commission as a fraction of notional per side, e.g. 0.0005 (default 0)",
    )
    p.add_argument(
        "--slippage",
        type=_non_negative,
        default=0.0,
        metavar="X",
        help="extra adverse price units on entries and stop, time and end exits (default 0)",
    )
    p.add_argument(
        "--max-bars",
        type=_positive_int,
        default=None,
        metavar="N",
        help="time exit after N bars, counting the entry bar, if the backtest used one",
    )
    p.add_argument(
        "--same-bar",
        choices=("stop", "target"),
        default="stop",
        help="how the backtest booked a bar that touched both levels (default stop)",
    )
    p.add_argument(
        "--sims",
        type=_positive_int,
        default=1000,
        metavar="N",
        help="random-entry simulations (default 1000)",
    )
    p.add_argument(
        "--seed", type=_non_negative_int, default=0, metavar="N", help="random seed (default 0)"
    )
    p.add_argument(
        "--periods-per-year",
        type=_finite,
        default=None,
        metavar="P",
        help="trades per year, to print an annualised Sharpe ratio",
    )
    p.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    p.add_argument(
        "--fail-on",
        choices=("fail", "warn", "never"),
        default="fail",
        help="exit 1 when a check reaches this status (default: fail)",
    )
    p.add_argument("--time-column", default=None, metavar="NAME", help="timestamp column in BARS")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser(
        "lookahead",
        help="test a signal function for lookahead",
        description=(
            "Import fn(bars) -> array and check that row i never depends on bars after i. "
            "Exit 1 when a leak is found or the function raises on a probe."
        ),
    )
    p.add_argument("target", metavar="MODULE:FUNCTION", help="importable module or FILE.py path")
    p.add_argument("--bars", required=True, help="CSV with open, high, low, close columns")
    p.add_argument("--mode", choices=("truncation", "perturbation", "both"), default="both")
    p.add_argument(
        "--min-history",
        type=_positive_int,
        default=1,
        metavar="N",
        help="shortest prefix the function must handle (truncation test)",
    )
    p.add_argument("--time-column", default=None, metavar="NAME", help="timestamp column in BARS")
    p.set_defaults(func=_cmd_lookahead)

    p = sub.add_parser(
        "dsr",
        help="probabilistic and deflated Sharpe ratio",
        description=(
            "PSR, DSR, the expected maximum Sharpe ratio of N skill-less trials, and the "
            "minimum track record length (Bailey & Lopez de Prado 2012, 2014). Sharpe ratios "
            "and the trials' variance are per period unless --annualized is given."
        ),
    )
    p.add_argument("--sharpe", type=_finite, required=True, metavar="S", help="observed Sharpe")
    p.add_argument(
        "--n-obs", type=_finite, required=True, metavar="T", help="number of return observations"
    )
    p.add_argument(
        "--trials",
        type=_finite,
        required=True,
        metavar="N",
        help="number of independent configurations tried",
    )
    p.add_argument(
        "--trials-variance",
        type=_finite,
        required=True,
        metavar="V",
        help="variance of the trials' Sharpe ratios",
    )
    p.add_argument("--skew", type=_finite, default=0.0, metavar="G", help="skewness (default 0)")
    p.add_argument(
        "--kurtosis",
        type=_finite,
        default=3.0,
        metavar="K",
        help="RAW kurtosis, 3 for a normal distribution (not excess kurtosis); default 3",
    )
    p.add_argument("--periods-per-year", type=_finite, default=None, metavar="P")
    p.add_argument(
        "--annualized",
        action="store_true",
        help="--sharpe is annual and --trials-variance is of annual Sharpe ratios; "
        "needs --periods-per-year",
    )
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(func=_cmd_dsr)

    p = sub.add_parser(
        "selftest",
        help="check the harness catches planted problems",
        description=(
            "Run lookahead.self_test() and random_entry.positive_control() on synthetic data. "
            "Exit 1 when any probe is judged wrongly."
        ),
    )
    p.add_argument("--seed", type=_non_negative_int, default=0, metavar="N")
    p.set_defaults(func=_cmd_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    func: Callable[[argparse.Namespace], int] = args.func
    try:
        return func(args)
    except (_UsageError, DataError, ValueError, TypeError, OSError, KeyError) as exc:
        message = str(exc) if not isinstance(exc, KeyError) else f"missing key {exc}"
        print(f"backtest-skeptic {args.command}: error: {message}", file=sys.stderr)
        return EXIT_ERROR
