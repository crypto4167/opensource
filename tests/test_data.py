import numpy as np
import pytest

from backtest_skeptic import Bars, DataError


def test_rejects_nan_and_reports_the_bar():
    with pytest.raises(DataError, match="bar\\(s\\) 1"):
        Bars([1, 1, 1], [1, np.nan, 1], [1, 1, 1], [1, 1, 1])


def test_rejects_high_below_close():
    with pytest.raises(DataError, match="inconsistent"):
        Bars([1.0], [1.0], [0.9], [1.1])


def test_rejects_unsorted_or_duplicate_time():
    with pytest.raises(DataError, match="strictly increasing"):
        Bars([1, 1], [1, 1], [1, 1], [1, 1], time=["2024-01-02", "2024-01-01"])
    with pytest.raises(DataError, match="strictly increasing"):
        Bars([1, 1], [1, 1], [1, 1], [1, 1], time=[5, 5])


def test_rejects_length_mismatch():
    with pytest.raises(DataError):
        Bars([1, 1], [1], [1, 1], [1, 1])


def test_arrays_are_read_only_but_callers_data_is_not_frozen():
    close = np.array([1.0, 2.0])
    bars = Bars(close, close, close, close)
    with pytest.raises(ValueError):
        bars.close[0] = 5.0
    close[0] = 3.0  # caller's array is still writable
    assert bars.close[0] == 1.0


def test_prefix_and_slice_are_views():
    bars = Bars.from_close(np.arange(1.0, 11.0))
    p = bars.prefix(3)
    assert len(p) == 3 and p.close[-1] == 3.0
    assert len(bars[2:5]) == 3
    with pytest.raises(TypeError):
        bars[3]  # type: ignore[index]


def test_from_csv_maps_columns_and_reports_bad_lines(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text(
        "Date,Open,High,Low,Adj Close,Volume\n2024-01-01,1,2,0.5,1.5,10\n2024-01-02,1.5,2,1,1.8,\n"
    )
    bars = Bars.from_csv(good, columns={"close": "Adj Close"})
    assert list(bars.close) == [1.5, 1.8]
    assert list(bars.volume) == [10.0, 0.0]
    assert list(bars.time) == ["2024-01-01", "2024-01-02"]
    assert bars.index_of("2024-01-02") == 1

    bad = tmp_path / "bad.csv"
    bad.write_text("time,open,high,low,close\n1,1,1,1,1\n2,1,x,1,1\n")
    with pytest.raises(DataError, match="line 3"):
        Bars.from_csv(bad)


def test_from_records_accepts_common_keys():
    bars = Bars.from_records(
        [
            {"datetime": "a", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
            {"datetime": "b", "open": 1.5, "high": 2, "low": 1, "close": 1.8},
        ]
    )
    assert list(bars.time) == ["a", "b"]


def test_from_dataframe_duck_typed():
    class Frame:
        columns = ("Open", "High", "Low", "Close")
        index = np.array([10, 20])

        def __getitem__(self, key):
            return {"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1.5], "Close": [1.5, 2.5]}[key]

    bars = Bars.from_dataframe(Frame())
    assert list(bars.time) == [10, 20]
    assert list(bars.close) == [1.5, 2.5]
