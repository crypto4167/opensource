import datetime as dt
import hashlib
import json
import math
import sys
import threading

import numpy as np
import pytest

from backtest_skeptic import Bars, summarize
from backtest_skeptic.holdout import (
    HoldoutAlreadyRead,
    HoldoutError,
    HoldoutLedger,
    HoldoutMismatch,
    HoldoutSplit,
    LedgerCorrupted,
    TrialRegistry,
    fingerprint,
    split_holdout,
)
from backtest_skeptic.synthetic import random_walk_bars

# ---------------------------------------------------------------------------
# split_holdout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 10, 101, 1000])
@pytest.mark.parametrize("fraction", [0.01, 0.2, 0.5, 0.99])
@pytest.mark.parametrize("purge", [0, 1, 5])
def test_split_never_shares_a_bar(n, fraction, purge):
    try:
        split = split_holdout(n, holdout_fraction=fraction, purge=purge)
    except ValueError:
        return  # this combination leaves no training bar, which must raise
    idx = np.arange(n)
    train = set(idx[split.train].tolist())
    holdout = set(idx[split.holdout].tolist())
    purged = set(idx[split.purged].tolist())
    assert not train & holdout
    assert not train & purged
    assert not purged & holdout
    assert train | purged | holdout == set(range(n))
    assert max(train) < min(holdout)
    assert split.train.stop + purge == split.holdout.start
    assert split.n_train + split.n_purged + split.n_holdout == n
    assert split.n_purged == purge
    assert split.n_holdout >= 1 and split.n_train >= 1


def test_regression_boundary_bar_belongs_to_holdout_only():
    # An earlier script selected train with ``i <= boundary`` and holdout with
    # ``i >= boundary``, so the boundary bar was in both sets.
    split = split_holdout(10, holdout_start=8)
    assert split.train == slice(0, 8)
    assert split.holdout == slice(8, 10)
    assert 8 not in range(10)[split.train]
    assert 8 in range(10)[split.holdout]


def test_split_fraction_sizes():
    split = split_holdout(1000)
    assert (split.n_train, split.n_holdout) == (800, 200)
    split = split_holdout(10, holdout_fraction=0.25)  # ceil(2.5) = 3
    assert (split.n_train, split.n_holdout) == (7, 3)
    split = split_holdout(100, holdout_fraction=0.07)
    assert split.n_holdout == 7  # 0.07 * 100 is 7.000000000000001 in floating point


def test_purge_drops_the_last_training_bars():
    split = split_holdout(100, holdout_start=80, purge=5)
    assert split.train == slice(0, 75)
    assert split.purged == slice(75, 80)
    assert split.holdout == slice(80, 100)
    assert split.n_holdout == 20  # the holdout itself is untouched


def test_purge_stops_forward_looking_labels_from_reading_the_holdout():
    # Labels computed from close[i + h] cross into the holdout unless purge >= h.
    h = 4
    n = 200
    unpurged = split_holdout(n, holdout_fraction=0.25)
    reach = np.arange(n)[unpurged.train] + h
    assert (reach >= unpurged.holdout.start).any()  # positive control: the leak exists
    purged = split_holdout(n, holdout_fraction=0.25, purge=h)
    reach = np.arange(n)[purged.train] + h
    assert (reach < purged.holdout.start).all()
    assert purged.holdout == unpurged.holdout


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((1,), {}, "at least 2"),
        ((True,), {}, "integer"),
        ((10.0,), {}, "integer"),
        ((10,), {"holdout_fraction": 0.0}, "between 0 and 1"),
        ((10,), {"holdout_fraction": 1.0}, "between 0 and 1"),
        ((10,), {"holdout_fraction": math.nan}, "between 0 and 1"),
        ((10,), {"holdout_fraction": "a"}, "number"),
        ((10,), {"holdout_start": 0}, "1..9"),
        ((10,), {"holdout_start": 10}, "1..9"),
        ((10,), {"holdout_start": 5, "purge": 5}, "removes every training bar"),
        ((10,), {"purge": -1}, "purge must be >= 0"),
        ((3,), {"holdout_fraction": 0.99}, "1..2"),
    ],
)
def test_split_bad_input_raises(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        split_holdout(*args, **kwargs)


def test_split_apply_and_formatting():
    bars = random_walk_bars(50, seed=1)
    split = split_holdout(50, holdout_fraction=0.2, purge=2)
    train, holdout = split.apply(bars)
    assert len(train) == 38 and len(holdout) == 10
    assert holdout.close[0] == bars.close[40]
    with pytest.raises(ValueError, match="for 50 bars"):
        split.apply(bars[:49])
    with pytest.raises(TypeError):
        split.apply(np.zeros(50))
    assert isinstance(split, HoldoutSplit)
    assert split.as_dict() == {
        "n": 50,
        "train": [0, 38],
        "purged": [38, 40],
        "holdout": [40, 50],
        "n_train": 38,
        "n_purged": 2,
        "n_holdout": 10,
    }
    assert str(split) == "train bars 0..37 (38), purged 38..39 (2), holdout 40..49 (10) of 50"


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def tiny_bars(volume=True, time=True):
    return Bars(
        [1.0, 2.0],
        [2.0, 2.5],
        [0.5, 1.5],
        [2.0, 2.25],
        volume=[10.0, 20.0] if volume else None,
        time=np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[s]")
        if time
        else None,
    )


def reference_fingerprint(bars):
    """The documented format, re-implemented independently."""
    h = hashlib.sha256(b"backtest-skeptic/bars/v1")

    def section(label, payload):
        h.update(label + b"\x00" + len(payload).to_bytes(8, "little") + payload)

    section(b"n", len(bars).to_bytes(8, "little"))
    for label in ("open", "high", "low", "close", "volume"):
        arr = getattr(bars, label)
        if arr is not None:
            section(label.encode(), np.asarray(arr, dtype="<f8").tobytes())
    if bars.time is not None:
        parts = [str(t).encode() for t in bars.time]
        lengths = b"".join(len(p).to_bytes(8, "little") for p in parts)
        section(b"time", lengths + b"".join(parts))
    return h.hexdigest()


def test_fingerprint_matches_the_documented_format():
    for volume in (True, False):
        for time in (True, False):
            bars = tiny_bars(volume, time)
            assert fingerprint(bars) == reference_fingerprint(bars)
    # Pinned so that a format change, which would orphan every existing
    # ledger, cannot happen by accident.
    assert fingerprint(tiny_bars()) == (
        "0faece5eb442b8a955a1988c661fe2972ccfbe068ef621ece13f38c7dd8a44b5"
    )


def test_fingerprint_is_stable_for_equal_data():
    bars = random_walk_bars(300, seed=2)
    copy = Bars(
        bars.open.copy(),
        bars.high.copy(),
        bars.low.copy(),
        bars.close.copy(),
        volume=bars.volume.copy(),
        time=bars.time.copy(),
    )
    assert fingerprint(bars) == fingerprint(copy)
    assert fingerprint(bars[100:200]) == fingerprint(copy[100:200])
    assert len(fingerprint(bars)) == 64


def test_fingerprint_detects_any_change():
    bars = random_walk_bars(300, seed=3)
    base = fingerprint(bars)
    close = bars.close.copy()
    close[150] = np.nextafter(close[150], np.inf)  # one ulp
    high = np.maximum(bars.high, close)
    assert fingerprint(bars.with_prices(bars.open, high, bars.low, close)) != base
    assert fingerprint(bars[:-1]) != base
    assert fingerprint(bars[1:]) != base
    no_volume = Bars(bars.open, bars.high, bars.low, bars.close, time=bars.time)
    assert fingerprint(no_volume) != base
    shifted = Bars(
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        volume=bars.volume,
        time=bars.time + np.timedelta64(1, "s"),
    )
    assert fingerprint(shifted) != base
    vol = bars.volume.copy()
    vol[0] += 1
    assert (
        fingerprint(Bars(bars.open, bars.high, bars.low, bars.close, volume=vol, time=bars.time))
        != base
    )


def test_fingerprint_rejects_non_bars():
    with pytest.raises(TypeError):
        fingerprint(np.zeros(10))


# ---------------------------------------------------------------------------
# HoldoutLedger
# ---------------------------------------------------------------------------


@pytest.fixture
def holdout():
    bars = random_walk_bars(500, seed=4)
    return split_holdout(len(bars), holdout_fraction=0.2, purge=3).apply(bars)[1]


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_seal_records_fingerprint_and_metadata(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    seal = HoldoutLedger(path).seal("h", holdout, note="last 20%")
    assert seal.fingerprint == fingerprint(holdout)
    assert seal.n_bars == 100
    assert seal.first_time == str(holdout.time[0])
    assert seal.last_time == str(holdout.time[-1])
    assert seal.note == "last 20%"
    created = dt.datetime.fromisoformat(seal.created_at)
    assert created.utcoffset() == dt.timedelta(0)
    assert abs((dt.datetime.now(dt.timezone.utc) - created).total_seconds()) < 60
    (line,) = lines(path)
    event = json.loads(line)
    assert event["event"] == "seal" and event["name"] == "h"
    assert event["prev"] == "0" * 64
    assert "holdout 'h': 100 bars" in str(seal)


def test_negative_control_single_read_is_accepted(tmp_path, holdout):
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    ledger.seal("h", holdout)
    rec = ledger.read("h", holdout, strategy_id="breakout-v1", result={"sharpe": 1.2})
    assert rec.read_number == 1
    assert not rec.reread
    assert rec.result == {"sharpe": 1.2}
    (status,) = ledger.status().holdouts
    assert status.clean and status.n_reads == 1
    assert status.strategies == ("breakout-v1",)


def test_positive_control_second_read_is_refused(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    ledger.read("h", holdout, strategy_id="v1")
    before = lines(path)
    with pytest.raises(HoldoutAlreadyRead, match="already read 1 time"):
        ledger.read("h", holdout, strategy_id="v2")
    with pytest.raises(HoldoutAlreadyRead):
        ledger.read("h", holdout, strategy_id="v1")  # the same strategy too
    assert lines(path) == before  # a refused read writes nothing


def test_allowed_reread_is_numbered_and_flagged(tmp_path, holdout):
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    ledger.seal("h", holdout)
    ledger.read("h", holdout, strategy_id="v1")
    rec = ledger.read("h", holdout, strategy_id="v2", allow_reread=True)
    assert rec.read_number == 2 and rec.reread
    assert "RE-READ" in str(rec)
    (status,) = ledger.status().holdouts
    assert not status.clean
    assert status.n_reads == 2
    assert status.strategies == ("v1", "v2")
    assert "no longer a clean holdout" in str(ledger.status())
    assert [r.read_number for r in ledger.reads("h")] == [1, 2]


def test_tampered_data_is_detected(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    close = holdout.close.copy()
    close[-1] *= 1.001
    edited = holdout.with_prices(
        holdout.open, np.maximum(holdout.high, close), np.minimum(holdout.low, close), close
    )
    before = lines(path)
    with pytest.raises(HoldoutMismatch, match="differ from the sealed data"):
        ledger.read("h", edited, strategy_id="v1")
    with pytest.raises(HoldoutMismatch, match="got 99 bars"):
        ledger.read("h", holdout[:-1], strategy_id="v1")
    assert lines(path) == before
    # The genuine data can still be read: the failed attempts did not count.
    assert ledger.read("h", holdout, strategy_id="v1").read_number == 1


def test_ledger_survives_reopening(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    first = HoldoutLedger(path)
    first.seal("h", holdout)
    result = summarize(np.array([1.0, -1.0, 2.0, 3.0]))  # profit factor is finite
    first.read("h", holdout, strategy_id="v1", result=result)
    del first

    reopened = HoldoutLedger(str(path))
    with pytest.raises(HoldoutAlreadyRead):
        reopened.read("h", holdout, strategy_id="v2")
    (rec,) = reopened.reads("h")
    assert rec.strategy_id == "v1"
    assert rec.result["trades"] == 4
    assert rec.result["expectancy_r"] == pytest.approx(1.25)
    status = reopened.status()
    assert status.n_events == 2
    assert status.holdouts[0].n_reads == 1


def test_result_values_are_converted_to_strict_json(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    result = {
        "pf": math.inf,
        "neg": -math.inf,
        "nan": math.nan,
        "n": np.int64(3),
        "x": np.float32(0.5),
        "flag": np.bool_(True),
        "arr": np.array([1.0, 2.0]),
        "pair": (1, 2),
        "when": np.datetime64("2024-01-01T00:00"),
    }
    rec = ledger.read("h", holdout, strategy_id="v1", result=result)
    assert rec.result == {
        "pf": "Infinity",
        "neg": "-Infinity",
        "nan": "NaN",
        "n": 3,
        "x": 0.5,
        "flag": True,
        "arr": [1.0, 2.0],
        "pair": [1, 2],
        "when": "2024-01-01T00:00",
    }

    def reject(token):
        raise AssertionError(f"non-standard JSON token {token}")

    for line in lines(path):
        json.loads(line, parse_constant=reject)


def test_bad_result_writes_nothing(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    before = lines(path)
    with pytest.raises(ValueError, match="result must be a dict"):
        ledger.read("h", holdout, strategy_id="v1", result=[1, 2])
    with pytest.raises(ValueError, match="cannot be stored as JSON"):
        ledger.read("h", holdout, strategy_id="v1", result={"f": object()})
    with pytest.raises(ValueError, match="keys must be strings"):
        ledger.read("h", holdout, strategy_id="v1", result={1: 2})
    assert lines(path) == before


def test_sealing_a_name_twice(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    first = ledger.seal("h", holdout)
    again = ledger.seal("h", holdout, note="ignored")
    assert again == first  # idempotent: same data returns the original record
    assert len(lines(path)) == 1
    with pytest.raises(HoldoutMismatch, match="already sealed with different data"):
        ledger.seal("h", holdout[:-1])
    assert len(lines(path)) == 1


def test_resealing_the_same_bars_under_a_new_name_is_refused(tmp_path, holdout):
    # Positive control for the obvious workaround: rename the holdout to get
    # a fresh read count.
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    ledger.read("h", holdout, strategy_id="v1")
    with pytest.raises(HoldoutError, match="already sealed as holdout 'h'"):
        ledger.seal("h-fresh", holdout)
    assert len(lines(path)) == 2


def test_reading_an_unsealed_holdout_raises(tmp_path, holdout):
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(HoldoutError, match="no holdout named 'h'") as info:
        ledger.read("h", holdout, strategy_id="v1")
    assert not isinstance(info.value, HoldoutAlreadyRead)


def test_holdouts_are_independent(tmp_path, holdout):
    other = random_walk_bars(80, seed=99)
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    ledger.seal("a", holdout)
    ledger.seal("b", other)
    ledger.read("a", holdout, strategy_id="v1")
    assert ledger.read("b", other, strategy_id="v1").read_number == 1
    with pytest.raises(HoldoutMismatch):
        ledger.read("a", other, strategy_id="v1", allow_reread=True)
    status = ledger.status()
    assert [h.name for h in status.holdouts] == ["a", "b"]
    assert status.as_dict()["holdouts"][0]["clean"] is True


def test_status_of_empty_and_fresh_ledgers(tmp_path, holdout):
    ledger = HoldoutLedger(tmp_path / "missing.jsonl")
    status = ledger.status()
    assert status.n_events == 0 and status.holdouts == ()
    assert "no holdouts sealed" in str(status)
    assert ledger.reads("h") == ()
    ledger.seal("h", holdout)
    assert "never read" in str(ledger.status())


def test_edited_ledger_line_is_detected(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    ledger.read("h", holdout, strategy_id="v1")
    text = path.read_text(encoding="utf-8")
    head = ledger.status().head
    # Editing the last line cannot break the chain (nothing follows it), but
    # it changes the head hash, which is why LedgerStatus.head is worth
    # publishing somewhere else.
    path.write_text(text.replace('"read_number": 1', '"read_number": 0'), encoding="utf-8")
    assert ledger.status().head != head
    path.write_text(text.replace('"n_bars": 100', '"n_bars": 101'), encoding="utf-8")
    with pytest.raises(LedgerCorrupted, match="line 2: hash chain broken"):
        ledger.status()


def test_deleted_read_event_is_detected(tmp_path, holdout):
    # Deleting the read to get a "fresh" holdout breaks the chain as soon as
    # anything follows it.
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    ledger.read("h", holdout, strategy_id="v1")
    ledger.seal("other", random_walk_bars(30, seed=5))
    seal, _read, other = lines(path)
    path.write_text(seal + "\n" + other + "\n", encoding="utf-8")
    with pytest.raises(LedgerCorrupted, match="hash chain broken"):
        ledger.read("h", holdout, strategy_id="v2")


def test_partial_and_foreign_files_are_rejected(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    ledger = HoldoutLedger(path)
    ledger.seal("h", holdout)
    path.write_text(path.read_text(encoding="utf-8") + '{"event": "read"', encoding="utf-8")
    with pytest.raises(LedgerCorrupted, match="incomplete"):
        ledger.status()
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(LedgerCorrupted, match="not valid JSON"):
        ledger.status()
    trials = tmp_path / "trials.jsonl"
    TrialRegistry(trials).record("t", sharpe=0.1)
    with pytest.raises(LedgerCorrupted, match="not a holdout ledger event"):
        HoldoutLedger(trials).status()


@pytest.mark.skipif(sys.platform == "win32", reason="advisory locking is POSIX-only")
def test_concurrent_reads_record_exactly_one(tmp_path, holdout):
    path = tmp_path / "ledger.jsonl"
    HoldoutLedger(path).seal("h", holdout)
    outcomes = []
    barrier = threading.Barrier(8)

    def attempt(k):
        barrier.wait()
        try:
            HoldoutLedger(path).read("h", holdout, strategy_id=f"s{k}")
            outcomes.append("ok")
        except HoldoutAlreadyRead:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok"] + ["refused"] * 7
    assert len(HoldoutLedger(path).reads("h")) == 1


def test_ledger_input_validation(tmp_path, holdout):
    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="name"):
        ledger.seal("", holdout)
    with pytest.raises(TypeError):
        ledger.seal("h", holdout.close)
    with pytest.raises(ValueError, match="empty"):
        ledger.seal("h", holdout[:0])
    ledger.seal("h", holdout)
    with pytest.raises(ValueError, match="strategy_id"):
        ledger.read("h", holdout, strategy_id="  ")
    assert repr(ledger).startswith("HoldoutLedger(")


# ---------------------------------------------------------------------------
# TrialRegistry
# ---------------------------------------------------------------------------


def test_registry_counts_unique_configurations(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("rsi", sharpe=0.10, n_obs=500, params={"n": 14, "level": 30})
    reg.record("rsi", sharpe=0.20, n_obs=500, params={"n": 21, "level": 30})
    reg.record("rsi", sharpe=0.40, n_obs=500, params={"n": 28, "level": 30})
    assert reg.n_trials == 3
    # hand computed: mean 0.7/3, squared deviations sum to 0.046667, / 2
    assert reg.sharpe_variance == pytest.approx(0.0233333333, rel=1e-8)
    # Negative control: re-running a configuration (keys in another order)
    # does not inflate N.
    reg.record("rsi rerun", sharpe=0.40, n_obs=500, params={"level": 30, "n": 28})
    assert reg.n_trials == 3
    assert reg.sharpe_variance == pytest.approx(0.0233333333, rel=1e-8)
    assert len(reg.trials()) == 4


def test_registry_uses_latest_record_of_a_configuration(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("a", sharpe=0.1, params={"k": 1})
    reg.record("b", sharpe=0.3, params={"k": 2})
    reg.record("a fixed", sharpe=0.2, params={"k": 1})
    assert reg.n_trials == 2
    assert reg.sharpe_variance == pytest.approx(np.var([0.2, 0.3], ddof=1))
    assert reg.summary().best.name == "b"


def test_registry_computes_sharpe_from_returns(tmp_path):
    rng = np.random.default_rng(6)
    r = rng.normal(0.05, 1.0, 250)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    rec = reg.record("ma-cross", r, params={"fast": 10, "slow": 50})
    assert rec.sharpe == pytest.approx(r.mean() / r.std(ddof=1))
    assert rec.n_obs == 250
    assert rec.source == "returns"
    assert rec.params == {"fast": 10, "slow": 50}
    assert rec.record_number == 1
    expected = hashlib.sha256(b'{"fast":10,"slow":50}').hexdigest()
    assert rec.params_hash == expected


def test_registry_without_params_identifies_trials_by_name(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("a", sharpe=0.1)
    reg.record("b", sharpe=0.2)
    reg.record("a", sharpe=0.1)
    assert reg.n_trials == 2


def test_registry_survives_reopening(tmp_path):
    path = tmp_path / "trials.jsonl"
    reg = TrialRegistry(path)
    for k, sr in enumerate([0.05, -0.02, 0.11, 0.07]):
        reg.record("grid", sharpe=sr, n_obs=1000, params={"k": k})
    again = TrialRegistry(str(path))
    assert again.n_trials == 4
    assert again.sharpe_variance == pytest.approx(np.var([0.05, -0.02, 0.11, 0.07], ddof=1))
    assert [t.record_number for t in again.trials()] == [1, 2, 3, 4]
    again.record("grid", sharpe=0.0, n_obs=1000, params={"k": 4})
    assert TrialRegistry(path).n_trials == 5


def test_registry_summary_names_the_deflated_sharpe_inputs(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    empty = reg.summary()
    assert empty.n_trials == 0 and empty.best is None and empty.sharpe_variance is None
    assert "no trials" in str(empty)
    reg.record("x", sharpe=0.1, n_obs=300, params={"a": 1})
    one = reg.summary()
    assert one.sharpe_variance is None and "fewer than 2" in str(one)
    reg.record("y", sharpe=0.3, n_obs=300, params={"a": 2})
    summary = reg.summary()
    assert summary.n_trials == 2 and summary.n_records == 2
    assert summary.best.sharpe == 0.3
    assert "n_trials=2" in str(summary) and "var_trials_sharpe=0.02" in str(summary)
    assert summary.as_dict()["best"]["name"] == "y"


def test_registry_variance_needs_two_trials(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="at least 2 distinct trials"):
        _ = reg.sharpe_variance
    reg.record("x", sharpe=0.1)
    reg.record("x", sharpe=0.1)
    with pytest.raises(ValueError, match="have 1"):
        _ = reg.sharpe_variance


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (("t",), {}, "pass returns or a per-period sharpe"),
        (("t", [0.1, 0.2, 0.3]), {"sharpe": 0.5}, "not both"),
        (("t", [0.1, 0.2, 0.3]), {"n_obs": 3}, "not both"),
        (("t", [0.1, np.nan, 0.3]), {}, "finite"),
        (("t", [0.1]), {}, "at least 2"),
        (("t", [0.1, 0.1, 0.1]), {}, "constant"),
        (("t", [[0.1, 0.2]]), {}, "one-dimensional"),
        (("t",), {"sharpe": math.inf}, "finite"),
        (("t",), {"sharpe": True}, "finite number"),
        (("t",), {"sharpe": 0.1, "n_obs": 1}, "n_obs must be >= 2"),
        (("t",), {"sharpe": 0.1, "n_obs": 2.5}, "integer"),
        (("",), {"sharpe": 0.1}, "name"),
        (("t",), {"sharpe": 0.1, "params": [1, 2]}, "mapping"),
        (("t",), {"sharpe": 0.1, "params": {"x": object()}}, "cannot be stored"),
    ],
)
def test_registry_bad_input_raises(tmp_path, args, kwargs, message):
    path = tmp_path / "trials.jsonl"
    with pytest.raises(ValueError, match=message):
        TrialRegistry(path).record(*args, **kwargs)
    assert not path.exists() or path.read_text() == ""


def test_registry_feeds_the_expected_maximum_sharpe(tmp_path):
    # Purpose check with the expected-maximum formula of Bailey & Lopez de
    # Prado (2014). Forty skill-less trials: the best one looks significant
    # on its own, but not against what N and V[SR] predict for the best of
    # forty. A trial with a real edge clears that bar.
    rng = np.random.default_rng(7)
    n_obs = 250
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    for k in range(40):
        reg.record("noise", rng.normal(0.0, 1.0, n_obs), params={"k": k})
    summary = reg.summary()
    n, v = summary.n_trials, summary.sharpe_variance
    gamma = 0.5772156649015329
    from statistics import NormalDist

    inv = NormalDist().inv_cdf
    e_max = math.sqrt(v) * ((1 - gamma) * inv(1 - 1 / n) + gamma * inv(1 - 1 / (n * math.e)))
    best = summary.best.sharpe
    assert best * math.sqrt(n_obs) > 1.96  # naive t-test says "significant"
    assert (best - e_max) * math.sqrt(n_obs - 1) < 1.645  # deflated: it is not
    real = rng.normal(0.35, 1.0, n_obs)
    sr_real = real.mean() / real.std(ddof=1)
    assert (sr_real - e_max) * math.sqrt(n_obs - 1) > 1.645


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX only")
def test_reading_waits_for_an_append_in_progress(tmp_path):
    import fcntl

    path = tmp_path / "trials.jsonl"
    reg = TrialRegistry(path)
    reg.record("seed", sharpe=0.1)
    result = {}

    def read():
        try:
            result["n"] = len(reg.trials())
        except Exception as exc:  # pragma: no cover - the defect this test guards
            result["error"] = exc

    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write('{"event": "half a li')  # a writer caught mid-append
        fh.flush()
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(0.5)
        assert reader.is_alive(), result  # waits instead of reading half a line
        fh.truncate(path.stat().st_size - len('{"event": "half a li'))
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    reader.join(5)
    assert result == {"n": 1}
