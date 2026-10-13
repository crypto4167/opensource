"""Does a known factor explain the returns?

A backtest can show a real, statistically significant mean return and still
have no edge worth paying for: if the strategy is mostly long a market that
went up, or tilted towards small or volatile names in a period when they did
well, the same return was available by holding the factor. This module asks
how much of a strategy's return a set of factor (or proxy) returns already
explains.

* :func:`spanning_regression` - OLS of the strategy's returns on factor
  returns with an intercept, with Newey-West (HAC) standard errors. The
  intercept (alpha) is the part of the mean return the factors do not
  explain; the appraisal ratio ``alpha / residual volatility`` is what the
  strategy adds to the factors in Sharpe-ratio terms.
* :func:`proxy_replication` - the one-factor special case phrased as a
  question: does a no-information proxy (the market, size, volatility)
  reproduce the "edge"?

Why the intercept is the test
-----------------------------
For excess returns (zero-cost strategies, or returns net of the risk-free
rate) write ``r_t = alpha + beta' f_t + e_t``. The best Sharpe ratio
attainable from the factors plus the strategy satisfies
``SR(f, r)^2 = SR(f)^2 + alpha^2 / var(e)``, so the strategy improves on the
factors exactly when ``alpha != 0``, and by how much is governed by the
appraisal ratio ``alpha / sd(e)``. This is the intercept form of the
mean-variance spanning question of Huberman & Kandel (1987). Their full
spanning hypothesis for *total* (not excess) returns also restricts the
betas to sum to one; that restriction is not tested here because backtest
returns are normally excess or zero-cost returns.

Do not use the Sharpe ratio of the residuals as an alpha measure. OLS
residuals have mean zero by construction (the intercept absorbs their mean),
so ``mean(e) / sd(e)`` is zero up to rounding for every strategy, good or
bad. The alpha lives in the intercept, which is why the appraisal ratio
divides the *intercept* by the residual volatility.

Conventions
-----------
* Returns and factors are per-period numbers in any consistent unit (simple
  returns, log returns, R multiples per period). Use excess returns (minus
  the risk-free rate) for both when the risk-free rate is not negligible.
* Alpha and the appraisal ratio are per period. With ``periods_per_year``
  they are also annualised: alpha times ``periods_per_year`` and the
  appraisal ratio times ``sqrt(periods_per_year)`` (the latter assumes
  serially uncorrelated residuals).

References
----------
* Huberman, G. & Kandel, S. (1987). Mean-Variance Spanning. *Journal of
  Finance* 42(4).
* Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica* 55(3).
* Newey, W. K. & West, K. D. (1994). Automatic Lag Selection in Covariance
  Matrix Estimation. *Review of Economic Studies* 61(4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ProxyResult", "SpanningResult", "proxy_replication", "spanning_regression"]

FloatArray = NDArray[np.float64]

_NORMAL = NormalDist()
_REL_TOL = 1e-12
_RANK_TOL = 1e-10


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


def _vector(values: ArrayLike, name: str) -> FloatArray:
    """One-dimensional float64 copy; NaN is allowed (rows are dropped later), inf is not."""
    try:
        x = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: values are not numeric ({exc})") from exc
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    if np.isinf(x).any():
        raise ValueError(f"{name} contains inf; only NaN rows are dropped, inf is an error")
    return x


def _factor_matrix(
    factors: ArrayLike | Mapping[str, ArrayLike],
    names: Sequence[str] | None,
    t_obs: int,
) -> tuple[FloatArray, tuple[str, ...]]:
    """``T x K`` factor matrix and the factor names."""
    if isinstance(factors, Mapping):
        if names is not None:
            raise ValueError("names must not be given with a dict of factors; the keys are names")
        if not factors:
            raise ValueError("factors: the dict is empty")
        keys = list(factors.keys())
        bad = [k for k in keys if not isinstance(k, str)]
        if bad:
            raise ValueError(f"factor names (dict keys) must be strings, got {bad!r}")
        cols = [_vector(factors[k], f"factor {k!r}") for k in keys]
        for k, col in zip(keys, cols, strict=True):
            if len(col) != t_obs:
                raise ValueError(f"factor {k!r} has {len(col)} values but returns has {t_obs}")
        return np.column_stack(cols), tuple(keys)

    try:
        f = np.array(factors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"factors: values are not numeric ({exc})") from exc
    if f.ndim == 1:
        f = f[:, None]
    if f.ndim != 2:
        raise ValueError(f"factors must be a T x K array, a 1-D array or a dict, got {f.shape}")
    if f.shape[0] != t_obs:
        hint = " (transposed? factors must be T x K, one row per period)"
        raise ValueError(
            f"factors has {f.shape[0]} rows but returns has {t_obs} values"
            + (hint if f.shape[1] == t_obs else "")
        )
    if f.shape[1] == 0:
        raise ValueError("factors has no columns")
    if np.isinf(f).any():
        raise ValueError("factors contains inf; only NaN rows are dropped, inf is an error")
    k = f.shape[1]
    if names is None:
        labels = tuple(f"f{j + 1}" for j in range(k))
    else:
        labels = tuple(names)
        if len(labels) != k:
            raise ValueError(f"names has {len(labels)} entries but factors has {k} columns")
        if not all(isinstance(s, str) for s in labels):
            raise ValueError("names must be strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"factor names must be unique, got {labels!r}")
    return f, labels


def _drop_nan(y: FloatArray, f: FloatArray) -> tuple[FloatArray, FloatArray, int]:
    keep = ~(np.isnan(y) | np.isnan(f).any(axis=1))
    dropped = int(len(y) - keep.sum())
    return y[keep], f[keep], dropped


def _default_lags(t_obs: int) -> int:
    """``floor(4 * (T / 100) ** (2 / 9))`` (Newey & West, 1994), at most ``T - 1``."""
    lag = math.floor(4.0 * math.pow(t_obs / 100.0, 2.0 / 9.0))
    return min(lag, t_obs - 1)


def _check_lags(lags: int | None, t_obs: int) -> int:
    if lags is None:
        return _default_lags(t_obs)
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)):
        raise ValueError(f"hac_lags must be a non-negative integer or None, got {lags!r}")
    lag = int(lags)
    if not 0 <= lag <= t_obs - 1:
        raise ValueError(f"hac_lags must be in 0..{t_obs - 1} for {t_obs} observations, got {lag}")
    return lag


def _check_ppy(periods_per_year: float | None) -> float | None:
    if periods_per_year is None:
        return None
    if isinstance(periods_per_year, bool):
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")
    try:
        ppy = float(periods_per_year)
    except (TypeError, ValueError):
        raise ValueError(
            f"periods_per_year must be a positive number, got {periods_per_year!r}"
        ) from None
    if not (math.isfinite(ppy) and ppy > 0):
        raise ValueError(f"periods_per_year must be a positive finite number, got {ppy!r}")
    return ppy


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------


def _ols(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """OLS coefficients, ``(X'X)^-1`` and residuals.

    Columns are scaled to unit length before the QR decomposition, so the
    rank check does not depend on the units of the factors.
    """
    norms = np.linalg.norm(x, axis=0)
    if (norms == 0).any():
        raise ValueError("a factor is identically zero after dropping NaN rows")
    xs = x / norms
    q, r = np.linalg.qr(xs)
    diag = np.abs(np.diag(r))
    if diag.min() <= _RANK_TOL:
        raise ValueError(
            "the factors are collinear (or one is constant, which is collinear with the "
            "intercept); drop or combine the redundant factors"
        )
    r_inv = np.linalg.inv(r)
    coef_s = r_inv @ (q.T @ y)
    coef = coef_s / norms
    xtx_inv = (r_inv @ r_inv.T) / np.outer(norms, norms)
    resid = y - x @ coef
    return coef, xtx_inv, resid


def _hac_covariance(x: FloatArray, resid: FloatArray, xtx_inv: FloatArray, lags: int) -> FloatArray:
    """Newey-West sandwich ``(X'X)^-1 S (X'X)^-1`` with Bartlett weights.

    ``S = sum_t u_t u_t' + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j')`` with
    ``u_t = x_t e_t`` and ``G_j = sum_t u_t u_{t-j}'``. No small-sample
    degrees-of-freedom correction is applied (the divisor is ``T``), matching
    :func:`backtest_skeptic.stats.newey_west`. With ``L = 0`` this is White's
    heteroskedasticity-consistent covariance.
    """
    u = x * resid[:, None]
    s = u.T @ u
    for j in range(1, lags + 1):
        g = u[j:].T @ u[:-j]
        s += (1.0 - j / (lags + 1.0)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    out: FloatArray = (cov + cov.T) / 2.0
    return out


def _ratio(num: float, den: float, negligible: bool, degenerate: bool) -> float:
    """``num / den`` with the conventions of :func:`backtest_skeptic.metrics.t_statistic`.

    When the fit is exact (``degenerate``) the denominator is zero up to
    rounding: the ratio is ``0.0`` for a negligible numerator and ``+-inf``
    otherwise, instead of a huge number that depends on rounding noise.
    """
    if degenerate or den <= 0.0:
        return 0.0 if negligible else math.copysign(math.inf, num)
    return num / den


def _two_sided_p(t: float) -> float:
    if math.isinf(t):
        return 0.0
    return 2.0 * (1.0 - _NORMAL.cdf(abs(t)))


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# spanning regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanningResult:
    """OLS of strategy returns on factor returns, with HAC standard errors.

    Attributes
    ----------
    alpha:
        Intercept, per period: the mean return the factors do not explain.
    alpha_se, alpha_t, alpha_p:
        Newey-West (HAC) standard error, t-statistic and two-sided
        normal-approximation p-value of the intercept.
    alpha_se_ols, alpha_t_ols:
        Classical OLS standard error and t-statistic of the intercept, for
        comparison. When ``alpha_se`` is much larger, the residuals are
        serially correlated (or heteroskedastic) and the plain t-statistic
        overstates the evidence.
    alpha_annual:
        ``alpha * periods_per_year``, or ``None`` without ``periods_per_year``.
    betas, beta_se, beta_t, beta_se_ols:
        Factor loadings with their HAC standard errors and t-statistics, and
        OLS standard errors, keyed by factor name.
    r_squared, adj_r_squared:
        Share of the return variance explained by the factors.
    mean_return:
        Mean strategy return per period over the rows used.
    alpha_share:
        ``alpha / mean_return``: the share of the mean return that is not
        factor exposure. Near 0 means the "edge" is the factors; it can be
        negative or above 1 when factor exposure cost money. ``None`` when
        the mean return is exactly zero. Unstable when the mean return is
        close to zero.
    residual_std:
        Residual standard deviation with ``T - K - 1`` degrees of freedom.
    appraisal_ratio, appraisal_ratio_annual:
        ``alpha / residual_std`` per period, and times
        ``sqrt(periods_per_year)``. This, not the Sharpe ratio of the
        residuals (which is always about 0), is the alpha measure.
    n_obs, n_dropped:
        Rows used, and rows dropped because a return or factor was NaN.
    hac_lags:
        Bartlett lags used for the HAC covariance.
    residuals:
        The OLS residuals (read-only), for diagnostics. Their mean is zero by
        construction. Not included in :meth:`as_dict`.
    """

    alpha: float
    alpha_se: float
    alpha_t: float
    alpha_p: float
    alpha_se_ols: float
    alpha_t_ols: float
    alpha_annual: float | None
    betas: dict[str, float]
    beta_se: dict[str, float]
    beta_t: dict[str, float]
    beta_se_ols: dict[str, float]
    r_squared: float
    adj_r_squared: float
    mean_return: float
    alpha_share: float | None
    residual_std: float
    appraisal_ratio: float
    appraisal_ratio_annual: float | None
    n_obs: int
    n_dropped: int
    hac_lags: int
    periods_per_year: float | None
    names: tuple[str, ...]
    residuals: FloatArray = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Every field except ``residuals``, as plain Python values."""
        return {
            "alpha": self.alpha,
            "alpha_se": self.alpha_se,
            "alpha_t": self.alpha_t,
            "alpha_p": self.alpha_p,
            "alpha_se_ols": self.alpha_se_ols,
            "alpha_t_ols": self.alpha_t_ols,
            "alpha_annual": self.alpha_annual,
            "betas": dict(self.betas),
            "beta_se": dict(self.beta_se),
            "beta_t": dict(self.beta_t),
            "beta_se_ols": dict(self.beta_se_ols),
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "mean_return": self.mean_return,
            "alpha_share": self.alpha_share,
            "residual_std": self.residual_std,
            "appraisal_ratio": self.appraisal_ratio,
            "appraisal_ratio_annual": self.appraisal_ratio_annual,
            "n_obs": self.n_obs,
            "n_dropped": self.n_dropped,
            "hac_lags": self.hac_lags,
            "periods_per_year": self.periods_per_year,
            "names": list(self.names),
        }

    def __str__(self) -> str:
        k = len(self.names)
        dropped = f", {self.n_dropped} NaN rows dropped" if self.n_dropped else ""
        lines = [
            f"Spanning regression on {k} factor{'s' if k != 1 else ''} "
            f"(n={self.n_obs}{dropped}, HAC lags {self.hac_lags})",
        ]
        annual = ""
        if self.alpha_annual is not None:
            annual = f" ({self.alpha_annual:.6g} annualised)"
        lines.append(
            f"  alpha {self.alpha:.6g} per period{annual}, t {self.alpha_t:.2f} "
            f"(HAC; OLS t {self.alpha_t_ols:.2f}), p {self.alpha_p:.3g}"
        )
        width = max(len(n) for n in self.names)
        for name in self.names:
            lines.append(
                f"  beta {name:<{width}} {self.betas[name]:+.4f}  t {self.beta_t[name]:.2f}"
            )
        lines.append(
            f"  R^2 {self.r_squared:.3f}, mean return {self.mean_return:.6g}, "
            f"alpha share of mean {_fmt(self.alpha_share, '.1%')}"
        )
        ar_annual = ""
        if self.appraisal_ratio_annual is not None:
            ar_annual = f" ({self.appraisal_ratio_annual:.3f} annualised)"
        lines.append(f"  appraisal ratio {self.appraisal_ratio:.4f} per period{ar_annual}")
        return "\n".join(lines)


def spanning_regression(
    returns: ArrayLike,
    factors: ArrayLike | Mapping[str, ArrayLike],
    *,
    names: Sequence[str] | None = None,
    hac_lags: int | None = None,
    periods_per_year: float | None = None,
) -> SpanningResult:
    """Regress strategy returns on factor returns and test the intercept.

    What it catches: an "edge" that is really exposure to something you
    could have bought without the strategy. A high Sharpe ratio driven by a
    long bias in a rising market shows up here as a large beta, a high
    R^2, an alpha near zero and a small ``alpha_share``.

    Model: ``returns_t = alpha + sum_k beta_k * factor_kt + e_t``, fitted by
    OLS. Standard errors come from the Newey-West (1987) HAC covariance
    ``(X'X)^-1 S (X'X)^-1`` with Bartlett weights, so serially correlated or
    heteroskedastic residuals (overlapping trades, multi-bar holds, volatility
    clustering) do not inflate the t-statistics. Classical OLS standard
    errors are reported alongside.

    Parameters
    ----------
    returns:
        Strategy returns, one per period (length ``T``).
    factors:
        Factor returns aligned with ``returns``: a ``T x K`` array, a 1-D
        array (one factor) or a dict mapping names to 1-D arrays.
    names:
        Factor names for an array input (default ``f1 .. fK``). Not allowed
        with a dict, whose keys are the names.
    hac_lags:
        Bartlett lags; default ``floor(4 * (T / 100) ** (2 / 9))`` (the rule of
        thumb of Newey & West, 1994) computed on the rows actually used. Use
        at least the holding period minus one for overlapping returns.
    periods_per_year:
        If given, alpha and the appraisal ratio are also annualised.

    Rows where the return or any factor is NaN are dropped and counted in
    ``n_dropped``; inf raises. Raises ``ValueError`` when fewer than
    ``K + 2`` rows remain, when the returns are constant, or when the
    factors are collinear (a constant factor is collinear with the
    intercept).

    Limitations: linear and static, so time-varying exposure (market timing)
    appears as alpha, which may or may not be what you want; a factor you do
    not include cannot explain anything; the HAC errors are asymptotic and
    noisy with few observations per lag. Choosing factors or lags after
    seeing the result is another degree of freedom.
    """
    y_all = _vector(returns, "returns")
    f_all, labels = _factor_matrix(factors, names, len(y_all))
    y, f, dropped = _drop_nan(y_all, f_all)
    t_obs, k = f.shape
    p = k + 1
    if t_obs < p + 1:
        raise ValueError(
            f"{t_obs} usable rows (after dropping {dropped} with NaN) for {k} factor(s) plus an "
            f"intercept; at least {p + 1} are required"
        )
    lags = _check_lags(hac_lags, t_obs)
    ppy = _check_ppy(periods_per_year)

    mean_y = float(y.mean())
    y_scale = float(np.abs(y).max())
    if float(y.std()) <= _REL_TOL * y_scale:
        raise ValueError("returns are constant; a regression on factors is undefined")
    sst = float(np.sum((y - mean_y) ** 2))

    x = np.column_stack([np.ones(t_obs), f])
    coef, xtx_inv, resid = _ols(y, x)
    ssr = float(resid @ resid)
    dof = t_obs - p
    degenerate = ssr <= (_REL_TOL**2) * sst
    s2 = ssr / dof
    cov_ols = s2 * xtx_inv
    cov_hac = _hac_covariance(x, resid, xtx_inv, lags)
    se_ols = np.sqrt(np.clip(np.diag(cov_ols), 0.0, None))
    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, None))

    # A coefficient is negligible when its contribution to the fitted values is.
    col_scale = np.abs(x).max(axis=0)
    negligible = np.abs(coef) * col_scale <= _REL_TOL * y_scale
    t_hac = [
        _ratio(float(coef[j]), float(se_hac[j]), bool(negligible[j]), degenerate) for j in range(p)
    ]

    alpha = float(coef[0])
    residual_std = math.sqrt(s2)
    appraisal = _ratio(alpha, residual_std, bool(negligible[0]), degenerate)
    r2 = 1.0 - ssr / sst
    adj_r2 = 1.0 - (1.0 - r2) * (t_obs - 1) / dof
    alpha_share = None if mean_y == 0.0 else alpha / mean_y

    frozen_resid = resid.copy()
    frozen_resid.setflags(write=False)
    return SpanningResult(
        alpha=alpha,
        alpha_se=float(se_hac[0]),
        alpha_t=t_hac[0],
        alpha_p=_two_sided_p(t_hac[0]),
        alpha_se_ols=float(se_ols[0]),
        alpha_t_ols=_ratio(alpha, float(se_ols[0]), bool(negligible[0]), degenerate),
        alpha_annual=None if ppy is None else alpha * ppy,
        betas={n: float(coef[j + 1]) for j, n in enumerate(labels)},
        beta_se={n: float(se_hac[j + 1]) for j, n in enumerate(labels)},
        beta_t={n: t_hac[j + 1] for j, n in enumerate(labels)},
        beta_se_ols={n: float(se_ols[j + 1]) for j, n in enumerate(labels)},
        r_squared=r2,
        adj_r_squared=adj_r2,
        mean_return=mean_y,
        alpha_share=alpha_share,
        residual_std=residual_std,
        appraisal_ratio=appraisal,
        appraisal_ratio_annual=None if ppy is None else appraisal * math.sqrt(ppy),
        n_obs=t_obs,
        n_dropped=dropped,
        hac_lags=lags,
        periods_per_year=ppy,
        names=labels,
        residuals=frozen_resid,
    )


# ---------------------------------------------------------------------------
# proxy replication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyResult:
    """How much of a strategy's mean return a single proxy reproduces.

    Attributes
    ----------
    correlation:
        Pearson correlation of strategy and proxy returns.
    beta:
        ``cov(returns, proxy) / var(proxy)``.
    alpha, alpha_t:
        ``mean(returns) - beta * mean(proxy)`` per period and its Newey-West
        t-statistic (the intercept of the one-factor regression).
    mean_return, mean_proxy:
        Mean per-period returns over the rows used.
    reproduced_share:
        ``beta * mean(proxy) / mean(returns)``: the share of the strategy's
        mean return that the proxy reproduces (``1 - alpha_share`` of the
        regression). Near 1 means the proxy explains the edge; negative when
        the exposure cost money. ``None`` when the mean return is exactly
        zero.
    sharpe_strategy, sharpe_proxy:
        Per-period Sharpe ratios (``mean / std``, ``ddof=1``) of the two
        series over the same rows.
    sharpe_strategy_annual, sharpe_proxy_annual:
        The same times ``sqrt(periods_per_year)`` (assumes serially
        uncorrelated returns), or ``None``.
    n_obs, n_dropped, hac_lags, periods_per_year:
        Rows used, NaN rows dropped, HAC lags, annualisation factor.
    """

    correlation: float
    beta: float
    alpha: float
    alpha_t: float
    mean_return: float
    mean_proxy: float
    reproduced_share: float | None
    sharpe_strategy: float
    sharpe_proxy: float
    sharpe_strategy_annual: float | None
    sharpe_proxy_annual: float | None
    n_obs: int
    n_dropped: int
    hac_lags: int
    periods_per_year: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "correlation": self.correlation,
            "beta": self.beta,
            "alpha": self.alpha,
            "alpha_t": self.alpha_t,
            "mean_return": self.mean_return,
            "mean_proxy": self.mean_proxy,
            "reproduced_share": self.reproduced_share,
            "sharpe_strategy": self.sharpe_strategy,
            "sharpe_proxy": self.sharpe_proxy,
            "sharpe_strategy_annual": self.sharpe_strategy_annual,
            "sharpe_proxy_annual": self.sharpe_proxy_annual,
            "n_obs": self.n_obs,
            "n_dropped": self.n_dropped,
            "hac_lags": self.hac_lags,
            "periods_per_year": self.periods_per_year,
        }

    def __str__(self) -> str:
        dropped = f", {self.n_dropped} NaN rows dropped" if self.n_dropped else ""
        if self.sharpe_strategy_annual is not None and self.sharpe_proxy_annual is not None:
            sharpes = (
                f"Sharpe {self.sharpe_strategy_annual:.2f} (strategy) vs "
                f"{self.sharpe_proxy_annual:.2f} (proxy), annualised"
            )
        else:
            sharpes = (
                f"Sharpe {self.sharpe_strategy:.4f} (strategy) vs "
                f"{self.sharpe_proxy:.4f} (proxy), per period"
            )
        return (
            f"Proxy replication (n={self.n_obs}{dropped}): beta * mean(proxy) reproduces "
            f"{_fmt(self.reproduced_share, '.1%')} of the mean return; "
            f"corr {self.correlation:+.3f}, beta {self.beta:+.3f}; "
            f"remaining alpha {self.alpha:.6g} per period, t {self.alpha_t:.2f} (HAC); "
            f"{sharpes}"
        )


def _sharpe(x: FloatArray, name: str) -> float:
    sd = float(x.std(ddof=1))
    if sd <= _REL_TOL * float(np.abs(x).max()):
        raise ValueError(f"{name} is constant; its Sharpe ratio is undefined")
    return float(x.mean()) / sd


def proxy_replication(
    returns: ArrayLike,
    proxy: ArrayLike,
    *,
    hac_lags: int | None = None,
    periods_per_year: float | None = None,
) -> ProxyResult:
    """Does a no-information proxy reproduce the strategy's "edge"?

    What it catches: a strategy whose mean return is a passive exposure in
    disguise. Pick a proxy that needs no skill to hold, such as the market
    itself (for a strategy that is mostly long), a small-minus-big or
    high-minus-low volatility spread (for a screen that tilts towards small
    or volatile names), or an equal-weight basket of the traded universe.
    If ``beta * mean(proxy)`` accounts for most of the strategy's mean
    return, the edge is the proxy.

    Inputs: per-period strategy returns and proxy returns, aligned. Rows
    where either is NaN are dropped and counted; inf raises; either series
    being constant raises. ``hac_lags`` and ``periods_per_year`` are as in
    :func:`spanning_regression`, which this calls with one factor.

    Reading the result: ``reproduced_share`` near 1 with an ``alpha_t``
    close to 0 means the proxy explains the edge; ``reproduced_share`` near
    0 with a large ``alpha_t`` means it does not. Comparing the two Sharpe
    ratios shows whether simply holding the proxy would have done as well
    on a risk-adjusted basis.

    Limitations: one proxy and a linear, constant beta. A proxy that fails
    to explain the edge rules out only that proxy, not every factor, and a
    beta that changes over time (timing the proxy) shows up as alpha.
    """
    y_all = _vector(returns, "returns")
    p_all = _vector(proxy, "proxy")
    if len(p_all) != len(y_all):
        raise ValueError(f"proxy has {len(p_all)} values but returns has {len(y_all)}")
    y, f, dropped = _drop_nan(y_all, p_all[:, None])
    px = f[:, 0]
    if len(y) < 3:
        raise ValueError(
            f"{len(y)} usable rows (after dropping {dropped} with NaN); at least 3 are required"
        )
    sr_y = _sharpe(y, "returns")
    sr_p = _sharpe(px, "proxy")
    reg = spanning_regression(
        y, {"proxy": px}, hac_lags=hac_lags, periods_per_year=periods_per_year
    )
    corr = float(np.corrcoef(y, px)[0, 1])
    beta = reg.betas["proxy"]
    mean_p = float(px.mean())
    share = None if reg.mean_return == 0.0 else beta * mean_p / reg.mean_return
    ppy = reg.periods_per_year
    root = None if ppy is None else math.sqrt(ppy)
    return ProxyResult(
        correlation=corr,
        beta=beta,
        alpha=reg.alpha,
        alpha_t=reg.alpha_t,
        mean_return=reg.mean_return,
        mean_proxy=mean_p,
        reproduced_share=share,
        sharpe_strategy=sr_y,
        sharpe_proxy=sr_p,
        sharpe_strategy_annual=None if root is None else sr_y * root,
        sharpe_proxy_annual=None if root is None else sr_p * root,
        n_obs=reg.n_obs,
        n_dropped=dropped,
        hac_lags=reg.hac_lags,
        periods_per_year=ppy,
    )
