# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

First public release.

### Added

- `Bars`: validated, read-only OHLC container with CSV, record and DataFrame loaders.
- `engine`: pessimistic bracket-order engine with explicit fill rules (next-bar-open entries, gapped stops filled at the open, stop-first on ambiguous bars, last bar checked before the end-of-data exit) shared by strategies and their controls.
- `lookahead`: truncation, perturbation and strategy-factory tests that catch signals reading future bars, plus a self-test of the harness.
- `random_entry`: matched random-entry and side-randomisation nulls with Phipson-Smyth p-values.
- `baseline`: no-skill and break-even win rates for bracket trades with an exact Poisson-binomial test.
- `fills`: fill-realism audit for trade logs from any backtester.
- `robustness`: concentration, drop-best, sub-period, circular block bootstrap, clustered t, cost stress and entry-delay checks.
- `stats`: probabilistic and deflated Sharpe ratio, minimum track record length, PBO via CSCV, Newey-West and Hansen-Hodrick t-statistics, Lo's Sharpe annualisation, p-value adjustments and Harvey-Liu haircuts.
- `holdout`: purged train/holdout split, read-once holdout ledger and trial registry.
- `spanning`: factor spanning regression with HAC standard errors and proxy replication.
- `synthetic`: random walks with continuous-path highs and lows, planted-signal bars and a cheating oracle for positive controls.
- `report.audit`: every check the inputs support in one call, for engine results or external trade logs, as text, Markdown or JSON.
- `backtest-skeptic` command-line interface with `audit`, `lookahead`, `dsr` and `selftest` subcommands and CI-friendly exit codes.
- Documentation: check reference, fill rules, FAQ, and measured false-positive rates and power for every statistical test (`docs/calibration.md`, reproducible with `scripts/calibration.py`).
- Examples: leaky signal functions, a random strategy with an 84% win rate, and a backtester with fill bugs.
