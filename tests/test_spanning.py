import math

import numpy as np
import pytest

from backtest_skeptic.metrics import t_statistic
from backtest_skeptic.spanning import (
    ProxyResult,
    SpanningResult,
    proxy_replication,
    spanning_regression,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ar1(rng, n, phi, sd):
    """AR(1) noise with innovation standard deviation ``sd``."""
    z = rng.normal(0.0, sd, n)
    e = np.empty(n)
    e[0] = z[0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + z[t]
    return e


def orthogonalise(e, x):
    """Remove from ``e`` its projection on the columns of ``x`` (exact OLS residuals)."""
    coef, *_ = np.linalg.lstsq(x, e, rcond=None)
    return e - x @ coef


def hac_by_loops(y, f, lags):
    """Newey-West sandwich written with explicit loops, as an independent check."""
    t_obs = len(y)
    x = np.column_stack([np.ones(t_obs), f])
    p = x.shape[1]
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    e = y - x @ b
    s = np.zeros((p, p))
    for t in range(t_obs):
        s += e[t] ** 2 * np.outer(x[t], x[t])
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        for t in range(j, t_obs):
            g = e[t] * e[t - j] * np.outer(x[t], x[t - j])
            s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return b, np.sqrt(np.diag(cov))


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


def test_hand_computed_one_factor_regression():
    # mean f = 1.5, mean y = 4, Sxx = 5, Sxy = 11 -> beta 2.2, alpha 0.7.
    # Residuals 0.3, 0.1, -1.1, 0.7: SSR 1.8, s^2 = 0.9, SST = 26.
    res = spanning_regression([1.0, 3.0, 4.0, 8.0], [0.0, 1.0, 2.0, 3.0], hac_lags=0)
    assert isinstance(res, SpanningResult)
    assert res.alpha == pytest.approx(0.7)
    assert res.betas == {"f1": pytest.approx(2.2)}
    assert res.alpha_se_ols == pytest.approx(math.sqrt(0.9 * (0.25 + 2.25 / 5.0)))
    assert res.beta_se_ols["f1"] == pytest.approx(math.sqrt(0.9 / 5.0))
    assert res.alpha_t_ols == pytest.approx(0.7 / math.sqrt(0.63))
    assert res.r_squared == pytest.approx(1.0 - 1.8 / 26.0)
    assert res.adj_r_squared == pytest.approx(1.0 - (1.8 / 26.0) * 3.0 / 2.0)
    assert res.mean_return == pytest.approx(4.0)
    assert res.alpha_share == pytest.approx(0.7 / 4.0)
    assert res.residual_std == pytest.approx(math.sqrt(0.9))
    assert res.appraisal_ratio == pytest.approx(0.7 / math.sqrt(0.9))
    assert np.allclose(res.residuals, [0.3, 0.1, -1.1, 0.7])
    assert (res.n_obs, res.n_dropped, res.hac_lags) == (4, 0, 0)
    assert res.alpha_annual is None and res.appraisal_ratio_annual is None


def test_hac_matches_an_explicit_loop_implementation():
    rng = np.random.default_rng(11)
    f = rng.normal(0.0, 1.0, (60, 2))
    y = 0.2 + f @ [0.5, -1.0] + ar1(rng, 60, 0.5, 1.0)
    for lags in (0, 1, 3, 7):
        res = spanning_regression(y, f, hac_lags=lags)
        b, se = hac_by_loops(y, f, lags)
        assert res.alpha == pytest.approx(b[0])
        assert [res.betas["f1"], res.betas["f2"]] == pytest.approx(b[1:].tolist())
        assert res.alpha_se == pytest.approx(se[0], rel=1e-10)
        assert [res.beta_se["f1"], res.beta_se["f2"]] == pytest.approx(se[1:].tolist(), rel=1e-10)
        assert res.alpha_t == pytest.approx(b[0] / se[0], rel=1e-10)


def test_default_lags_follow_newey_west_1994_rule():
    rng = np.random.default_rng(0)
    for t_obs, expected in ((100, 4), (1000, 6), (10, 2)):
        f = rng.normal(size=t_obs)
        res = spanning_regression(f + rng.normal(size=t_obs), f)
        assert res.hac_lags == expected == math.floor(4 * (t_obs / 100) ** (2 / 9))


def test_annualisation():
    rng = np.random.default_rng(1)
    f = rng.normal(0.0, 0.01, 500)
    y = 0.001 + 0.5 * f + rng.normal(0.0, 0.01, 500)
    res = spanning_regression(y, f, periods_per_year=252)
    assert res.alpha_annual == pytest.approx(res.alpha * 252)
    assert res.appraisal_ratio_annual == pytest.approx(res.appraisal_ratio * math.sqrt(252))
    assert res.periods_per_year == 252.0


def test_factor_input_forms_are_equivalent():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(300, 2))
    y = 0.1 + f @ [1.0, 2.0] + rng.normal(size=300)
    a = spanning_regression(y, f, names=["mkt", "smb"])
    b = spanning_regression(y, {"mkt": f[:, 0], "smb": f[:, 1]})
    assert a.as_dict() == b.as_dict()
    one = spanning_regression(y, f[:, 0])
    two = spanning_regression(y, f[:, :1], names=["f1"])
    assert one.as_dict() == two.as_dict()
    assert a.names == ("mkt", "smb")


def test_rank_check_does_not_depend_on_factor_units():
    rng = np.random.default_rng(3)
    f = rng.normal(size=400)
    y = 0.05 + 2.0 * f + rng.normal(size=400)
    base = spanning_regression(y, f)
    tiny = spanning_regression(y, f * 1e-9)
    assert tiny.betas["f1"] == pytest.approx(base.betas["f1"] * 1e9)
    assert tiny.alpha == pytest.approx(base.alpha)
    assert tiny.alpha_t == pytest.approx(base.alpha_t)


def test_exact_fit_reports_infinite_or_zero_t_instead_of_rounding_noise():
    x = np.random.default_rng(4).normal(size=50)
    res = spanning_regression(0.01 + 2.0 * x, x)
    assert res.alpha == pytest.approx(0.01)
    assert res.alpha_t == math.inf and res.alpha_t_ols == math.inf
    assert res.beta_t["f1"] == math.inf
    assert res.appraisal_ratio == math.inf
    assert res.alpha_p == 0.0
    pure = spanning_regression(2.0 * x, x)
    assert pure.alpha_t == 0.0 and pure.appraisal_ratio == 0.0
    assert pure.alpha_share == 0.0 or abs(pure.alpha_share) < 1e-12


# ---------------------------------------------------------------------------
# positive and negative controls
# ---------------------------------------------------------------------------


def test_positive_control_recovers_planted_alpha_and_betas():
    rng = np.random.default_rng(5)
    t_obs = 5000
    f = rng.normal(0.0003, 0.01, (t_obs, 3))
    true_b = np.array([0.8, -0.3, 0.5])
    y = 0.0004 + f @ true_b + rng.normal(0.0, 0.004, t_obs)
    res = spanning_regression(y, f, names=["mkt", "smb", "vol"])
    assert abs(res.alpha - 0.0004) < 4 * res.alpha_se
    for name, b in zip(res.names, true_b, strict=True):
        assert abs(res.betas[name] - b) < 4 * res.beta_se[name]
        assert abs(res.betas[name] - b) < 0.02
    assert res.alpha_t > 4
    assert res.alpha_p < 1e-4
    assert res.alpha_share == pytest.approx(res.alpha / res.mean_return)


def test_negative_control_edge_that_is_only_factor_exposure():
    # A strategy that is 1.2x long a rising market: a "significant" mean
    # return with no alpha. The spanning regression must not credit it.
    rng = np.random.default_rng(6)
    t_obs = 5000
    mkt = rng.normal(0.0005, 0.01, t_obs)
    y = 1.2 * mkt + rng.normal(0.0, 0.003, t_obs)
    assert t_statistic(y) > 2.5  # looks like an edge on its own
    res = spanning_regression(y, {"mkt": mkt})
    assert abs(res.alpha_t) < 2.0
    assert abs(res.alpha_share) < 0.3
    assert res.betas["mkt"] == pytest.approx(1.2, abs=0.02)
    assert res.r_squared > 0.9


def test_null_rejection_rate_is_close_to_nominal():
    rng = np.random.default_rng(7)
    rejections = 0
    sims = 300
    for _ in range(sims):
        f = rng.normal(0.0, 1.0, 400)
        y = 0.7 * f + rng.normal(0.0, 1.0, 400)
        rejections += spanning_regression(y, f).alpha_p < 0.05
    assert 0.01 <= rejections / sims <= 0.10


def test_hac_corrects_overrejection_from_autocorrelated_residuals():
    rng = np.random.default_rng(8)
    sims = 300
    hac = ols = 0
    for _ in range(sims):
        f = rng.normal(0.0, 1.0, 500)
        y = 0.5 * f + ar1(rng, 500, 0.6, 1.0)
        res = spanning_regression(y, f)
        hac += abs(res.alpha_t) > 1.96
        ols += abs(res.alpha_t_ols) > 1.96
    assert ols / sims > 0.2  # plain OLS t-stats find "alpha" far too often
    assert hac / sims < ols / sims - 0.1


def test_hac_se_larger_than_ols_se_on_autocorrelated_residuals():
    rng = np.random.default_rng(9)
    f = rng.normal(0.0, 0.01, 3000)
    y = 0.0002 + 0.9 * f + ar1(rng, 3000, 0.7, 0.004)
    res = spanning_regression(y, f)
    assert res.alpha_se > 1.5 * res.alpha_se_ols
    assert abs(res.alpha_t) < abs(res.alpha_t_ols)
    # Negative control: with independent residuals the two agree.
    iid = spanning_regression(0.0002 + 0.9 * f + rng.normal(0.0, 0.004, 3000), f)
    assert 0.85 < iid.alpha_se / iid.alpha_se_ols < 1.15


def test_alpha_share_with_exact_construction():
    rng = np.random.default_rng(10)
    f = rng.normal(0.0, 0.01, 1000)
    f = f - f.mean() + 0.0006  # sample mean exactly 0.0006
    x = np.column_stack([np.ones(1000), f])
    e = orthogonalise(rng.normal(0.0, 0.005, 1000), x)
    y = 0.0003 + 1.0 * f + e  # alpha 0.0003 out of a mean of 0.0009
    res = spanning_regression(y, f)
    assert res.alpha == pytest.approx(0.0003, rel=1e-9)
    assert res.betas["f1"] == pytest.approx(1.0, rel=1e-9)
    assert res.mean_return == pytest.approx(0.0009, rel=1e-9)
    assert res.alpha_share == pytest.approx(1.0 / 3.0, rel=1e-8)


def test_alpha_share_is_none_for_zero_mean_return():
    f = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    y = np.array([0.3, -0.1, 0.2, -0.4, 0.1, -0.1])
    res = spanning_regression(y, f)
    assert res.mean_return == 0.0
    assert res.alpha_share is None


def test_regression_residual_sharpe_is_not_reported_as_alpha():
    # An earlier script printed mean(resid)/sd(resid) as the alpha measure.
    # OLS residuals have mean zero by construction, so that number is ~0 even
    # for a strategy with a large, obvious alpha.
    rng = np.random.default_rng(12)
    f = rng.normal(0.0, 0.01, 2000)
    y = 0.002 + 0.5 * f + rng.normal(0.0, 0.005, 2000)
    res = spanning_regression(y, f, periods_per_year=252)
    resid = res.residuals
    residual_sharpe = resid.mean() / resid.std(ddof=1)
    assert abs(residual_sharpe) < 1e-10
    assert res.alpha_t > 10
    assert res.appraisal_ratio == pytest.approx(res.alpha / res.residual_std)
    assert res.appraisal_ratio > 0.3
    assert not any("sharpe" in key.lower() for key in res.as_dict())
    assert "sharpe" not in str(res).lower()
    assert "appraisal ratio" in str(res)


def test_residuals_are_read_only_and_excluded_from_as_dict():
    rng = np.random.default_rng(13)
    f = rng.normal(size=100)
    res = spanning_regression(f + rng.normal(size=100), f)
    with pytest.raises(ValueError):
        res.residuals[0] = 1.0
    assert "residuals" not in res.as_dict()


# ---------------------------------------------------------------------------
# NaN handling and validation
# ---------------------------------------------------------------------------


def test_nan_rows_are_dropped_and_counted():
    rng = np.random.default_rng(14)
    f = rng.normal(size=(200, 2))
    y = 0.1 + f @ [1.0, -1.0] + rng.normal(size=200)
    y_nan, f_nan = y.copy(), f.copy()
    y_nan[[3, 50, 120]] = np.nan
    f_nan[50, 0] = np.nan  # same row as a NaN return: counted once
    f_nan[[7, 199], 1] = np.nan
    res = spanning_regression(y_nan, f_nan, hac_lags=3)
    assert res.n_dropped == 5
    assert res.n_obs == 195
    keep = np.ones(200, dtype=bool)
    keep[[3, 50, 120, 7, 199]] = False
    clean = spanning_regression(y[keep], f[keep], hac_lags=3)
    assert res.alpha == clean.alpha
    assert res.alpha_se == clean.alpha_se
    assert res.betas == clean.betas
    assert "5 NaN rows dropped" in str(res)


def test_default_lags_use_rows_after_dropping_nan():
    rng = np.random.default_rng(15)
    f = rng.normal(size=1000)
    y = f + rng.normal(size=1000)
    y[:900] = np.nan
    res = spanning_regression(y, f)
    assert res.n_obs == 100
    assert res.hac_lags == 4


def test_inf_is_an_error_not_a_dropped_row():
    y = np.array([0.1, 0.2, np.inf, 0.0, 0.3])
    with pytest.raises(ValueError, match="inf"):
        spanning_regression(y, [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="inf"):
        spanning_regression([0.1, 0.2, 0.3, 0.0, 0.3], [1.0, np.inf, 3.0, 4.0, 5.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"returns": [1.0, 2.0, 3.0], "factors": [[1.0, 2.0, 3.0]]}, "transposed"),
        ({"returns": [1.0, 2.0, 3.0, 4.0], "factors": [1.0, 2.0, 3.0]}, "rows"),
        ({"returns": [1.0, 2.0], "factors": [1.0, 3.0]}, "at least 3"),
        ({"returns": [np.nan] * 5, "factors": [1.0, 2.0, 3.0, 4.0, 5.0]}, "usable rows"),
        ({"returns": [1.0] * 6, "factors": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]}, "constant"),
        (
            {"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": [[1.0, 2.0]] * 5},
            "collinear",
        ),
        ({"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": [3.0] * 5}, "collinear"),
        ({"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": [0.0] * 5}, "zero"),
        (
            {
                "returns": [1.0, 2.0, 0.0, 4.0, 5.0],
                "factors": {"a": [1.0, 2.0, 3.0, 4.0, 6.0]},
                "names": ["a"],
            },
            "names",
        ),
        (
            {
                "returns": [1.0, 2.0, 0.0, 4.0, 5.0],
                "factors": [[1.0, 2.0]] * 5,
                "names": ["a"],
            },
            "names",
        ),
        (
            {
                "returns": [1.0, 2.0, 0.0, 4.0, 5.0],
                "factors": [[1.0, 2.0]] * 5,
                "names": ["a", "a"],
            },
            "unique",
        ),
        ({"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": {}}, "empty"),
        (
            {"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": {"a": [1.0, 2.0]}},
            "has 2 values",
        ),
        (
            {"returns": [1.0, 2.0, 0.0, 4.0, 5.0], "factors": [1.0, 2.0, 3.0, 4.0, 6.0]},
            None,
        ),
    ],
)
def test_bad_input_raises(kwargs, message):
    if message is None:
        spanning_regression(**kwargs)  # the valid baseline of the table
        return
    with pytest.raises(ValueError, match=message):
        spanning_regression(**kwargs)


@pytest.mark.parametrize("lags", [-1, 5, 1.5, True])
def test_bad_lags_raise(lags):
    with pytest.raises(ValueError, match="hac_lags"):
        spanning_regression([1.0, 2.0, 0.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 6.0], hac_lags=lags)


@pytest.mark.parametrize("ppy", [0, -252, math.nan, math.inf, "daily", True])
def test_bad_periods_per_year_raise(ppy):
    with pytest.raises(ValueError, match="periods_per_year"):
        spanning_regression(
            [1.0, 2.0, 0.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 6.0], periods_per_year=ppy
        )


def test_str_is_readable():
    rng = np.random.default_rng(16)
    f = rng.normal(size=(300, 2))
    y = 0.1 + f @ [1.0, -0.5] + rng.normal(size=300)
    text = str(spanning_regression(y, f, names=["market", "size"], periods_per_year=12))
    assert text.startswith("Spanning regression on 2 factors (n=300, HAC lags")
    assert "beta market" in text and "beta size" in text
    assert "annualised" in text
    assert "alpha share of mean" in text


# ---------------------------------------------------------------------------
# proxy replication
# ---------------------------------------------------------------------------


def test_proxy_hand_computed_against_numpy():
    rng = np.random.default_rng(17)
    p = rng.normal(0.001, 0.01, 250)
    r = 0.0005 + 0.7 * p + rng.normal(0.0, 0.006, 250)
    res = proxy_replication(r, p, periods_per_year=252)
    assert isinstance(res, ProxyResult)
    beta = np.cov(r, p, ddof=1)[0, 1] / np.var(p, ddof=1)
    assert res.beta == pytest.approx(beta)
    assert res.correlation == pytest.approx(np.corrcoef(r, p)[0, 1])
    assert res.alpha == pytest.approx(r.mean() - beta * p.mean())
    assert res.reproduced_share == pytest.approx(beta * p.mean() / r.mean())
    assert res.sharpe_strategy == pytest.approx(r.mean() / r.std(ddof=1))
    assert res.sharpe_proxy == pytest.approx(p.mean() / p.std(ddof=1))
    assert res.sharpe_strategy_annual == pytest.approx(res.sharpe_strategy * math.sqrt(252))
    assert res.sharpe_proxy_annual == pytest.approx(res.sharpe_proxy * math.sqrt(252))
    reg = spanning_regression(r, p)
    assert res.alpha_t == pytest.approx(reg.alpha_t)
    assert res.reproduced_share == pytest.approx(1.0 - reg.alpha_share)


def test_proxy_positive_control_reproduces_a_fake_edge():
    # The "strategy" is a levered proxy plus noise: the proxy explains it.
    rng = np.random.default_rng(18)
    proxy = rng.normal(0.0004, 0.01, 4000)
    strat = 1.5 * proxy + rng.normal(0.0, 0.004, 4000)
    res = proxy_replication(strat, proxy, periods_per_year=252)
    assert 0.8 < res.reproduced_share < 1.2
    assert res.correlation > 0.9
    assert abs(res.alpha_t) < 2.0
    assert "reproduces" in str(res)


def test_proxy_negative_control_does_not_explain_a_real_edge():
    # A strategy with its own alpha and a driftless, unrelated proxy.
    rng = np.random.default_rng(19)
    proxy = rng.normal(0.0, 0.01, 4000)
    strat = 0.0008 + rng.normal(0.0, 0.006, 4000)
    res = proxy_replication(strat, proxy)
    assert abs(res.reproduced_share) < 0.2
    assert abs(res.correlation) < 0.1
    assert res.alpha_t > 5
    assert res.sharpe_strategy_annual is None


def test_proxy_nan_and_validation():
    rng = np.random.default_rng(20)
    p = rng.normal(0.001, 0.01, 100)
    r = 0.5 * p + rng.normal(0.0, 0.01, 100)
    r[5] = np.nan
    p[[5, 9]] = np.nan
    res = proxy_replication(r, p)
    assert res.n_dropped == 2 and res.n_obs == 98
    with pytest.raises(ValueError, match="proxy is constant"):
        proxy_replication(r, np.full(100, 0.01))
    with pytest.raises(ValueError, match="returns is constant"):
        proxy_replication(np.full(100, 0.01), np.nan_to_num(p))
    with pytest.raises(ValueError, match="has 99 values"):
        proxy_replication(r, p[:99])
    with pytest.raises(ValueError, match="at least 3"):
        proxy_replication([0.1, np.nan, 0.2], [0.1, 0.2, np.nan])
