"""Tests for the command-line interface, run through main(argv)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backtest_skeptic import Bars, TradeLog, __version__, run_signals, stats, synthetic
from backtest_skeptic.cli import main

MAX_BARS = 5


def write_bars(path: Path, bars: Bars) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        has_time = bars.time is not None
        writer.writerow((["time"] if has_time else []) + ["open", "high", "low", "close"])
        for i in range(len(bars)):
            row = [repr(float(a[i])) for a in (bars.open, bars.high, bars.low, bars.close)]
            writer.writerow(([str(bars.time[i])] if has_time else []) + row)  # type: ignore[index]
    return path


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Bars and the trade log of a random-entry strategy on a random walk."""
    tmp = tmp_path_factory.mktemp("clean")
    bars = synthetic.random_walk_bars(3000, seed=4)
    rng = np.random.default_rng(4)
    side = np.where(rng.random(len(bars)) < 0.1, rng.choice([-1.0, 1.0], len(bars)), 0.0)
    c = bars.close
    result = run_signals(bars, side, c * (1 - side * 0.006), c * (1 + side * 0.006), max_bars=5)
    bars_csv = write_bars(tmp / "bars.csv", bars)
    trades_csv = tmp / "trades.csv"
    TradeLog.from_trades(result.trades, bars).to_csv(trades_csv, bars)
    return bars_csv, trades_csv


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


# -------------------------------------------------------------------- audit


def test_audit_clean_run_passes_fills_and_shows_no_edge(
    clean_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    bars_csv, trades_csv = clean_run
    argv = ["audit", "--bars", str(bars_csv), "--trades", str(trades_csv)]
    code, out, err = run([*argv, "--max-bars", str(MAX_BARS), "--sims", "200"], capsys)
    assert code == 0, err
    assert "[PASS] Fill realism" in out
    assert "[WARN] Random-entry null" in out
    assert "[WARN] No-skill baseline" in out
    assert err == ""
    # The same report fails under --fail-on warn.
    code, _, _ = run([*argv, "--max-bars", "5", "--sims", "50", "--fail-on", "warn"], capsys)
    assert code == 1


def test_audit_json_parses(
    clean_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    bars_csv, trades_csv = clean_run
    code, out, _ = run(
        [
            "audit",
            "--bars",
            str(bars_csv),
            "--trades",
            str(trades_csv),
            "--max-bars",
            "5",
            "--sims",
            "50",
            "--format",
            "json",
        ],
        capsys,
    )
    assert code == 0
    data = json.loads(out)
    statuses = {c["name"]: c["status"] for c in data["checks"]}
    assert statuses["fill_realism"] == "PASS"
    assert statuses["random_entry"] == "WARN"
    assert data["settings"]["max_bars"] == 5
    assert data["settings"]["n_sims"] == 50


def test_audit_markdown(clean_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    bars_csv, trades_csv = clean_run
    argv = ["audit", "--bars", str(bars_csv), "--trades", str(trades_csv)]
    code, out, _ = run([*argv, "--sims", "20", "--format", "markdown"], capsys)
    assert code == 0
    assert out.startswith("# Backtest audit")


def test_audit_phantom_fills_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bars = synthetic.random_walk_bars(60, seed=2)
    bars_csv = write_bars(tmp_path / "bars.csv", bars)
    trades_csv = tmp_path / "trades.csv"
    rows = [
        # Exit above the exit bar's high: a price that never traded.
        ("BUY", 3, 8, bars.open[3], bars.high[8] * 1.01),
        ("SELL", 12, 15, bars.open[12], bars.low[15] * 0.99),
        ("BUY", 20, 24, bars.open[20], bars.close[24]),
    ]
    with open(trades_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["side", "entry_bar", "exit_bar", "entry_price", "exit_price"])
        for side, a, b, e, x in rows:
            writer.writerow([side, a, b, repr(float(e)), repr(float(x))])
    code, out, _ = run(["audit", "--bars", str(bars_csv), "--trades", str(trades_csv)], capsys)
    assert code == 1
    assert "[FAIL] Fill realism" in out
    assert "phantom_exit_fill x2" in out
    assert "Skipped." in out
    code, _, _ = run(
        ["audit", "--bars", str(bars_csv), "--trades", str(trades_csv), "--fail-on", "never"],
        capsys,
    )
    assert code == 0


def test_audit_bad_inputs_exit_2(
    tmp_path: Path, clean_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    bars_csv, trades_csv = clean_run
    code, out, err = run(
        ["audit", "--bars", str(tmp_path / "nope.csv"), "--trades", str(trades_csv)], capsys
    )
    assert code == 2
    assert out == ""
    assert "bars file not found" in err
    assert "Traceback" not in err

    code, _, err = run(
        ["audit", "--bars", str(bars_csv), "--trades", str(tmp_path / "nope.csv")], capsys
    )
    assert code == 2 and "trades file not found" in err

    # A log that does not belong to the bars.
    short = write_bars(tmp_path / "short.csv", synthetic.random_walk_bars(20, seed=0))
    code, _, err = run(["audit", "--bars", str(short), "--trades", str(trades_csv)], capsys)
    assert code == 2 and "error:" in err

    # Malformed bars.
    broken = tmp_path / "broken.csv"
    broken.write_text("open,high,low,close\n1,2,0.5,1\n1,x,0.5,1\n", encoding="utf-8")
    code, _, err = run(["audit", "--bars", str(broken), "--trades", str(trades_csv)], capsys)
    assert code == 2 and "line 3" in err

    # Usage errors.
    code, _, err = run(["audit", "--bars", str(bars_csv)], capsys)
    assert code == 2 and "--trades" in err
    code, _, _ = run(["audit", "--bars", str(bars_csv), "--trades", "t", "--sims", "0"], capsys)
    assert code == 2
    code, _, _ = run(["audit", "--bars", "b", "--trades", "t", "--half-spread", "-1"], capsys)
    assert code == 2
    code, _, _ = run([], capsys)
    assert code == 2


def test_audit_time_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bars = synthetic.random_walk_bars(40, seed=1)
    path = tmp_path / "bars.csv"
    write_bars(path, bars)
    text = path.read_text(encoding="utf-8").replace("time,", "Stamp,", 1)
    path.write_text(text, encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text(
        "side,entry_time,exit_time,entry_price,exit_price\n"
        f"1,{bars.time[2]},{bars.time[5]},{float(bars.open[2])!r},{float(bars.close[5])!r}\n",  # type: ignore[index]
        encoding="utf-8",
    )
    argv = ["audit", "--bars", str(path), "--trades", str(trades), "--sims", "10"]
    code, _, err = run(argv, capsys)
    assert code == 2 and "timestamps" in err
    code, out, err = run([*argv, "--time-column", "Stamp"], capsys)
    assert code == 0, err
    assert "[PASS] Fill realism" in out


# ---------------------------------------------------------------- lookahead

SIGNALS = """
import numpy as np


def centred(bars):
    x = bars.close
    out = np.full(len(x), np.nan)
    for i in range(2, len(x) - 2):
        out[i] = x[i - 2 : i + 3].mean()
    return out


def causal(bars):
    x = bars.close
    out = np.full(len(x), np.nan)
    for i in range(4, len(x)):
        out[i] = x[i - 4 : i + 1].mean()
    return out


def broken(bars):
    raise RuntimeError("boom")
"""


@pytest.fixture()
def signal_module(tmp_path: Path) -> Path:
    path = tmp_path / "my_signals.py"
    path.write_text(SIGNALS, encoding="utf-8")
    return path


@pytest.fixture()
def lookahead_bars(tmp_path: Path) -> Path:
    return write_bars(tmp_path / "bars.csv", synthetic.random_walk_bars(200, seed=0))


def test_lookahead_catches_centred_mean(
    signal_module: Path, lookahead_bars: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        ["lookahead", f"{signal_module}:centred", "--bars", str(lookahead_bars)], capsys
    )
    assert code == 1
    assert "LOOKAHEAD" in out
    assert "truncation test FAILED" in out and "perturbation test FAILED" in out


def test_lookahead_passes_causal_mean(
    signal_module: Path, lookahead_bars: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for mode in ("truncation", "perturbation", "both"):
        argv = ["lookahead", f"{signal_module}:causal", "--bars", str(lookahead_bars)]
        code, out, err = run([*argv, "--mode", mode, "--min-history", "5"], capsys)
        assert code == 0, err
        assert "PASSED" in out
        assert ("truncation" in out) == (mode != "perturbation")


def test_lookahead_importable_module(
    signal_module: Path,
    lookahead_bars: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(signal_module.parent))
    code, _, _ = run(["lookahead", "my_signals:centred", "--bars", str(lookahead_bars)], capsys)
    assert code == 1
    code, _, _ = run(["lookahead", "my_signals:causal", "--bars", str(lookahead_bars)], capsys)
    assert code == 0
    sys.modules.pop("my_signals", None)


def test_lookahead_errors(
    signal_module: Path, lookahead_bars: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bars = str(lookahead_bars)
    code, _, err = run(["lookahead", f"{signal_module}:missing", "--bars", bars], capsys)
    assert code == 2 and "no callable" in err
    code, _, err = run(["lookahead", "no_such_module_xyz:fn", "--bars", bars], capsys)
    assert code == 2 and "importing" in err
    code, _, err = run(["lookahead", "nocolon", "--bars", bars], capsys)
    assert code == 2 and "MODULE:FUNCTION" in err
    code, _, err = run(["lookahead", "missing_file.py:fn", "--bars", bars], capsys)
    assert code == 2 and "not found" in err
    # A function that raises on the full data is an error, reported without a traceback.
    code, _, err = run(["lookahead", f"{signal_module}:broken", "--bars", bars], capsys)
    assert code == 2 and "RuntimeError" in err and "boom" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------- dsr


def test_dsr_matches_stats(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["dsr", "--sharpe", "0.1", "--n-obs", "500", "--trials", "20"]
    argv += ["--trials-variance", "0.002", "--skew", "-0.5", "--kurtosis", "5", "--format", "json"]
    code, out, _ = run(argv, capsys)
    assert code == 0
    data = json.loads(out)
    kw = {"sr": 0.1, "n": 500.0, "skew": -0.5, "kurtosis": 5.0}
    assert data["psr"] == pytest.approx(stats.probabilistic_sharpe_ratio(**kw), rel=1e-12)
    assert data["dsr"] == pytest.approx(
        stats.deflated_sharpe_ratio(**kw, n_trials=20, var_trials_sharpe=0.002), rel=1e-12
    )
    emax = stats.expected_max_sharpe(20, 0.002)
    assert data["expected_max_sharpe_per_period"] == pytest.approx(emax, rel=1e-12)
    assert data["min_track_record_length"] == pytest.approx(
        stats.min_track_record_length(0.1, 0.0, -0.5, 5.0), rel=1e-12
    )


def test_dsr_annualized_paper_example(capsys: pytest.CaptureFixture[str]) -> None:
    # Bailey & Lopez de Prado (2014): annual SR 2.5, 1250 daily observations,
    # 100 trials with annual variance 0.5, skew -3, kurtosis 10 -> DSR about 0.90.
    argv = ["dsr", "--sharpe", "2.5", "--n-obs", "1250", "--trials", "100"]
    argv += ["--trials-variance", "0.5", "--skew", "-3", "--kurtosis", "10"]
    argv += ["--periods-per-year", "250", "--annualized"]
    code, out, _ = run([*argv, "--format", "json"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["dsr"] == pytest.approx(0.9004, abs=1e-3)
    assert data["sharpe_per_period"] == pytest.approx(2.5 / np.sqrt(250))
    code, out, _ = run(argv, capsys)
    assert code == 0
    assert "DSR (against the expected max):    0.90" in out
    assert "years" in out


def test_dsr_errors(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["dsr", "--sharpe", "1", "--n-obs", "100", "--trials", "5", "--trials-variance", "0.1"]
    code, _, err = run([*base, "--annualized"], capsys)
    assert code == 2 and "--periods-per-year" in err
    code, _, err = run([*base, "--kurtosis", "0"], capsys)
    assert code == 2 and "RAW" in err
    code, _, err = run([*base[:-2]], capsys)
    assert code == 2 and "--trials-variance" in err


def test_dsr_help_says_raw_kurtosis(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["dsr", "--help"], capsys)
    assert code == 0
    assert "RAW kurtosis" in out and "normal" in out


# --------------------------------------------------------- selftest, version


def test_selftest_passes(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["selftest"], capsys)
    assert code == 0
    assert "FAIL" not in out
    assert "PASS  centred_rolling_mean" in out
    assert "PASS  random_entry_positive_control" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["--version"], capsys)
    assert code == 0
    assert out.strip() == f"backtest-skeptic {__version__}"


def test_python_dash_m() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "backtest_skeptic", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout
