import math

import pytest

from backtest_skeptic.metrics import max_drawdown, profit_factor, summarize, t_statistic


def test_profit_factor_edge_cases():
    assert profit_factor([1.0, -0.5]) == pytest.approx(2.0)
    assert profit_factor([1.0, 2.0]) == math.inf
    assert profit_factor([0.0, 0.0]) is None
    assert profit_factor([]) is None


def test_t_statistic_degenerate_inputs():
    assert t_statistic([1.0]) is None
    assert t_statistic([0.1] * 10) == math.inf
    assert t_statistic([-0.1] * 10) == -math.inf
    assert t_statistic([0.0] * 10) == 0.0
    assert t_statistic([1.0, -1.0, 1.0, -1.0]) == pytest.approx(0.0)


def test_drawdown_counts_a_first_trade_loss():
    assert max_drawdown([-1.0, 2.0]) == pytest.approx(1.0)
    assert max_drawdown([1.0, -0.5, -1.0, 3.0]) == pytest.approx(1.5)
    assert max_drawdown([]) == 0.0


def test_summary_counts_breakeven_separately():
    s = summarize([1.0, -1.0, 0.0, 2.0])
    assert (s.wins, s.losses, s.breakeven) == (2, 1, 1)
    assert s.win_rate == pytest.approx(0.5)
    assert s.expectancy_r == pytest.approx(0.5)
    assert "4 trades" in str(s)


def test_summary_rejects_non_finite():
    with pytest.raises(ValueError):
        summarize([1.0, math.nan])


def test_empty_summary():
    s = summarize([])
    assert s.trades == 0 and s.win_rate is None and str(s) == "0 trades"
