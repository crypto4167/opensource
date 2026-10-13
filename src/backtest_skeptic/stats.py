"""Return-series statistics for judging whether a Sharpe ratio is real.

A backtest's Sharpe ratio is an estimate, and usually an inflated one. This
module implements, from the published formulas, the standard corrections:

* :func:`probabilistic_sharpe_ratio` - the probability that the true Sharpe
  ratio beats a benchmark, given the sample length, skewness and kurtosis
  (Bailey & Lopez de Prado, 2012).
* :func:`deflated_sharpe_ratio` - the same probability against the Sharpe
  ratio you would expect from the best of ``n_trials`` skill-less trials
  (Bailey & Lopez de Prado, 2014).
* :func:`min_track_record_length` and :func:`min_backtest_length` - how much
  data a claimed Sharpe ratio needs.
* :func:`pbo` - the probability of backtest overfitting by combinatorially
  symmetric cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017).
* :func:`newey_west` - a mean and a t-statistic that survive autocorrelated
  or overlapping returns (Newey & West, 1987, 1994; Hansen & Hodrick, 1980).
* :func:`lo_annualized_sharpe` - annualisation that accounts for serial
  correlation (Lo, 2002).
* :func:`adjust_pvalues` and :func:`haircut_sharpe` - multiple-testing
  corrections (Holm, 1979; Benjamini & Hochberg, 1995; Benjamini &
  Yekutieli, 2001; Harvey & Liu, 2015).

Conventions
-----------
* Every Sharpe ratio is **per period** (the sampling frequency of the
  returns, not annualised) unless a parameter name says ``annual``. Use
  :func:`annualize_sharpe` and :func:`deannualize_sharpe` to convert; both
  assume serially uncorrelated returns (see :func:`lo_annualized_sharpe` when
  they are not).
* Kurtosis is always **raw** kurtosis, ``E[(x - mu)^4] / sigma^4``, which is
  3 for a normal distribution. It is never excess kurtosis. Passing excess
  kurtosis (0 for a normal) is the most common way to get these formulas
  wrong, so functions that take a kurtosis reject values that violate
  Pearson's inequality ``kurtosis >= 1 + skew^2``.
* Returns are simple per-period returns or P&L in any consistent unit; R
  multiples from the engine work too.

References
----------
* Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk* 15(2).
* Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  *Journal of Portfolio Management* 40(5).
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2014).
  Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
  Overfitting on Out-of-Sample Performance. *Notices of the AMS* 61(5).
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017).
  The Probability of Backtest Overfitting. *Journal of Computational
  Finance* 20(4).
* Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio
  Management* 42(1).
* Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts
  Journal* 58(4).
* Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica* 55(3).
* Newey, W. K. & West, K. D. (1994). Automatic Lag Selection in Covariance
  Matrix Estimation. *Review of Economic Studies* 61(4).
* Hansen, L. P. & Hodrick, R. J. (1980). Forward Exchange Rates as Optimal
  Predictors of Future Spot Rates. *Journal of Political Economy* 88(5).
* Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
  *Scandinavian Journal of Statistics* 6(2).
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate.
  *Journal of the Royal Statistical Society B* 57(1).
* Benjamini, Y. & Yekutieli, D. (2001). The Control of the False Discovery
  Rate in Multiple Testing under Dependency. *Annals of Statistics* 29(4).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EULER_GAMMA",
    "LoSharpeResult",
    "NeweyWestResult",
    "PBOResult",
    "SharpeHaircut",
    "adjust_pvalues",
    "annualize_sharpe",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "haircut_sharpe",
    "lo_annualized_sharpe",
    "lo_scale_factor",
    "min_backtest_length",
    "min_track_record_length",
    "moments",
    "newey_west",
    "pbo",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sharpe_standard_error",
]

EULER_GAMMA = 0.57721566490153286
"""The Euler-Mascheroni constant used by the expected-maximum approximation."""

_NORMAL = NormalDist()
_REL_TOL = 1e-12

Metric = Callable[[NDArray[np.float64]], ArrayLike]
Kernel = Literal["bartlett", "uniform"]
PMethod = Literal["holm", "bonferroni", "bh", "by"]


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------


def _returns(values: ArrayLike, name: str = "returns") -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return x


def _unit_scale(x: NDArray[np.float64]) -> float:
    """``max |x|`` (1.0 for an all-zero or empty array).

    Dividing by it before squaring keeps scale-free statistics from
    overflowing (values near 1e200) or underflowing (near 1e-200).
    """
    s = float(np.abs(x).max()) if x.size else 0.0
    return s if s > 0 else 1.0


def _std_or_raise(x: NDArray[np.float64], name: str = "returns") -> float:
    if len(x) < 2:
        raise ValueError(f"{name}: at least 2 observations are required, got {len(x)}")
    sd = float(x.std(ddof=1))
    if sd <= _REL_TOL * float(np.abs(x).max()):
        raise ValueError(f"{name}: zero dispersion (a constant series); the statistic is undefined")
    return sd


def _check_finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _check_moments(skew: float, kurtosis: float) -> tuple[float, float]:
    skew = _check_finite("skew", skew)
    kurtosis = _check_finite("kurtosis", kurtosis)
    if kurtosis < 1.0 + skew * skew - 1e-9:
        raise ValueError(
            f"kurtosis={kurtosis!r} with skew={skew!r} violates kurtosis >= 1 + skew^2; "
            "kurtosis must be RAW kurtosis (3 for a normal distribution), not excess kurtosis"
        )
    return skew, kurtosis


def _check_n(n: float) -> float:
    n = _check_finite("n", n)
    if n <= 1:
        raise ValueError(f"n (the number of return observations) must be > 1, got {n!r}")
    return n


def _sr_variance_term(sr: float, skew: float, kurtosis: float) -> float:
    """``1 - skew*SR + (kurtosis - 1)/4 * SR^2``, the non-normal SR variance factor."""
    v = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if v <= 0:
        raise ValueError(
            f"1 - skew*sr + (kurtosis-1)/4*sr^2 = {v!r} is not positive for sr={sr!r}, "
            f"skew={skew!r}, kurtosis={kurtosis!r}"
        )
    return v


# ---------------------------------------------------------------------------
# Sharpe ratio and its sampling distribution
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: ArrayLike, *, periods_per_year: float | None = None) -> float:
    """Sample Sharpe ratio ``mean / std`` with the ``ddof=1`` standard deviation.

    Per period by default. With ``periods_per_year`` the result is multiplied
    by ``sqrt(periods_per_year)``, which is correct only for serially
    uncorrelated returns (see :func:`lo_annualized_sharpe`). Subtract any
    risk-free rate or benchmark from ``returns`` first.

    Raises ``ValueError`` for fewer than two observations, non-finite values or
    zero dispersion (a constant series has no Sharpe ratio).
    """
    x = _returns(returns)
    x = x / _unit_scale(x)
    sd = _std_or_raise(x)
    sr = float(x.mean()) / sd
    if periods_per_year is not None:
        sr = annualize_sharpe(sr, periods_per_year)
    return sr


def annualize_sharpe(sr: float, periods_per_year: float) -> float:
    """Per-period Sharpe ratio to annual: ``sr * sqrt(periods_per_year)`` (IID returns)."""
    ppy = _check_finite("periods_per_year", periods_per_year)
    if ppy <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return float(sr) * math.sqrt(ppy)


def deannualize_sharpe(sharpe_annual: float, periods_per_year: float) -> float:
    """Annual Sharpe ratio to per period: ``sharpe_annual / sqrt(periods_per_year)``.

    A *variance* of annual Sharpe ratios converts by dividing by
    ``periods_per_year`` (not its square root).
    """
    ppy = _check_finite("periods_per_year", periods_per_year)
    if ppy <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    return float(sharpe_annual) / math.sqrt(ppy)


def moments(returns: ArrayLike) -> tuple[float, float]:
    """Biased sample skewness and RAW biased sample kurtosis of ``returns``.

    ``skew = m3 / m2**1.5`` and ``kurtosis = m4 / m2**2`` with ``mk`` the
    k-th central sample moment (divisor ``n``). The kurtosis is raw: about 3
    for normal data, never excess. These are the plug-in estimates used by
    the probabilistic and deflated Sharpe ratio papers.
    """
    x = _returns(returns)
    x = x / _unit_scale(x)
    _std_or_raise(x)
    d = x - x.mean()
    m2 = float(np.mean(d**2))
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    return m3 / m2**1.5, m4 / m2**2


def sharpe_standard_error(sr: float, n: float, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Standard error of a per-period Sharpe ratio estimate.

    ``sqrt((1 - skew*sr + (kurtosis - 1)/4 * sr^2) / (n - 1))``, the
    asymptotic result for IID non-normal returns used in Bailey & Lopez de
    Prado (2012), building on Lo (2002). ``sr`` is per period, ``n`` is the
    number of return observations, and ``kurtosis`` is RAW kurtosis
    (3 for normal returns; the defaults assume normality). Serial
    correlation is not accounted for.
    """
    sr = _check_finite("sr", sr)
    n = _check_n(n)
    skew, kurtosis = _check_moments(skew, kurtosis)
    return math.sqrt(_sr_variance_term(sr, skew, kurtosis) / (n - 1.0))


def _sr_inputs(
    returns: ArrayLike | None,
    sr: float | None,
    n: float | None,
    skew: float | None,
    kurtosis: float | None,
) -> tuple[float, float, float, float]:
    given = {"sr": sr, "n": n, "skew": skew, "kurtosis": kurtosis}
    if returns is not None:
        mixed = [k for k, v in given.items() if v is not None]
        if mixed:
            raise ValueError(
                f"pass either returns or (sr, n, skew, kurtosis), not both; got returns and {mixed}"
            )
        x = _returns(returns)
        s = sharpe_ratio(x)
        g3, g4 = moments(x)
        return s, float(len(x)), g3, g4
    missing = [k for k, v in given.items() if v is None]
    if missing:
        raise ValueError(
            f"without returns, sr, n, skew and kurtosis are all required; missing {missing}"
        )
    assert sr is not None and n is not None and skew is not None and kurtosis is not None
    s = _check_finite("sr", sr)
    nn = _check_n(n)
    g3, g4 = _check_moments(skew, kurtosis)
    return s, nn, g3, g4


def probabilistic_sharpe_ratio(
    returns: ArrayLike | None = None,
    *,
    sr: float | None = None,
    n: float | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
    benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe ratio (PSR): P(true Sharpe > ``benchmark``).

    ``PSR = Phi((sr - benchmark) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurtosis-1)/4*sr^2))``
    from Bailey & Lopez de Prado (2012), eq. (11) there.

    What it catches: a Sharpe ratio that looks high only because the track
    record is short or the returns are negatively skewed and fat-tailed
    (option-like strategies that pick up pennies in front of a steamroller).

    Inputs: either ``returns`` (the moments are estimated from them) or all of
    ``sr`` (per period), ``n`` (number of observations), ``skew`` and
    ``kurtosis`` (RAW, normal = 3). Mixing the two forms raises, and so does
    omitting any of the four, so a normal distribution is never assumed
    silently. ``benchmark`` is a per-period Sharpe ratio.

    Limitations: asymptotic and IID; serial correlation and regime changes
    are ignored, and it does not correct for the number of strategies tried
    (use :func:`deflated_sharpe_ratio`).
    """
    s, nn, g3, g4 = _sr_inputs(returns, sr, n, skew, kurtosis)
    b = _check_finite("benchmark", benchmark)
    z = (s - b) * math.sqrt(nn - 1.0) / math.sqrt(_sr_variance_term(s, g3, g4))
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: float, var_sharpe: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` skill-less trials.

    ``sqrt(V) * ((1 - g) * PhiInv(1 - 1/N) + g * PhiInv(1 - 1/(N e)))`` with
    ``g`` the Euler-Mascheroni constant (Bailey & Lopez de Prado, 2014, eq. 1;
    Bailey et al., 2014). ``var_sharpe`` is the variance of the trials'
    Sharpe ratios in the same units as the Sharpe ratio you will compare
    (per period for :func:`deflated_sharpe_ratio`; divide a variance of
    annual Sharpe ratios by ``periods_per_year``).

    ``n_trials`` is the number of *independent* trials and may be a
    non-integer effective number. ``N = 1`` returns 0.0. The approximation
    is derived for large ``N``; below about 2 it is poor and can even be
    negative. Raises for ``N < 1`` or ``V < 0``.
    """
    n = _check_finite("n_trials", n_trials)
    v = _check_finite("var_sharpe", var_sharpe)
    if n < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    if v < 0:
        raise ValueError(f"var_sharpe must be >= 0, got {var_sharpe!r}")
    if n == 1:
        return 0.0
    # PhiInv(1 - p) = -PhiInv(p): 1 - 1/N loses digits for large N and rounds
    # to 1.0 from about N = 1e16, where PhiInv is undefined.
    z1 = -_NORMAL.inv_cdf(1.0 / n)
    z2 = -_NORMAL.inv_cdf(1.0 / n / math.e)
    return math.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    returns: ArrayLike | None = None,
    *,
    sr: float | None = None,
    n: float | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
    n_trials: float,
    var_trials_sharpe: float,
) -> float:
    """Deflated Sharpe ratio (DSR): PSR against the best-of-``n_trials`` threshold.

    ``DSR = PSR(benchmark = expected_max_sharpe(n_trials, var_trials_sharpe))``
    (Bailey & Lopez de Prado, 2014, eq. 2).

    What it catches: selection bias. Try enough configurations and the best
    one has a high Sharpe ratio by luck alone; DSR asks whether the selected
    strategy beats what luck would have produced, after also accounting for
    sample length, skewness and kurtosis.

    Inputs: the selected strategy's ``returns``, or all of ``sr`` (per
    period), ``n``, ``skew`` and ``kurtosis`` (RAW, normal = 3), exactly as in
    :func:`probabilistic_sharpe_ratio`. ``n_trials`` (number of independent
    trials) and ``var_trials_sharpe`` (variance across the trials' *per-period*
    Sharpe ratios) are required keyword arguments: there is deliberately no
    default, because silently falling back to a plain PSR when they are
    missing reports an undeflated number under a deflated name.

    Converting annual inputs: the paper's example has an annual trial
    variance of 0.5 with 250 observations a year, which is ``0.5 / 250`` per
    period, and an annual Sharpe of 2.5, which is ``2.5 / sqrt(250)``.

    Limitations: needs an honest count of independent trials, which is often
    the hardest input to get; correlated trials count as fewer than their
    number. Inherits PSR's IID asymptotics.
    """
    s, nn, g3, g4 = _sr_inputs(returns, sr, n, skew, kurtosis)
    threshold = expected_max_sharpe(n_trials, var_trials_sharpe)
    return probabilistic_sharpe_ratio(sr=s, n=nn, skew=g3, kurtosis=g4, benchmark=threshold)


def min_track_record_length(
    sr: float,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    prob: float = 0.95,
) -> float:
    """Minimum track record length (MinTRL), in number of return observations.

    ``1 + (1 - skew*sr + (kurtosis-1)/4*sr^2) * (PhiInv(prob) / (sr - benchmark))^2``
    (Bailey & Lopez de Prado, 2012, eq. 13): the sample length at which a
    per-period Sharpe ratio ``sr`` would be significantly above the per-period
    ``benchmark`` with confidence ``prob``. ``kurtosis`` is RAW (normal = 3;
    the defaults assume normal returns). Returns ``math.inf`` when
    ``sr <= benchmark``: no amount of data makes such a track record
    significant. Divide by ``periods_per_year`` for years.

    Limitations: an asymptotic IID result; the authors caution that the
    moments themselves need a long sample to be estimated reliably.
    """
    sr = _check_finite("sr", sr)
    b = _check_finite("benchmark", benchmark)
    skew, kurtosis = _check_moments(skew, kurtosis)
    p = _check_finite("prob", prob)
    if not 0.0 < p < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob!r}")
    if sr <= b:
        return math.inf
    z = _NORMAL.inv_cdf(p)
    return 1.0 + _sr_variance_term(sr, skew, kurtosis) * (z / (sr - b)) ** 2


def min_backtest_length(n_trials: float, target_sharpe: float) -> float:
    """Minimum backtest length (MinBTL), in **years**.

    ``((1 - g) * PhiInv(1 - 1/N) + g * PhiInv(1 - 1/(N e)))^2 / target_sharpe^2``
    (Bailey, Borwein, Lopez de Prado & Zhu, 2014, Theorem 3.1), where
    ``target_sharpe`` is an **annual** Sharpe ratio and ``g`` the
    Euler-Mascheroni constant. Below this length, the best of ``n_trials``
    independent skill-less configurations is expected to reach an in-sample
    annual Sharpe of ``target_sharpe`` by luck alone. The same paper gives
    the simpler upper bound ``2 ln(N) / target_sharpe^2``.

    Example from the paper: with 5 years of data, no more than about 45
    independent configurations should be tried before an in-sample annual
    Sharpe of 1 is expected from noise.

    Limitations: a necessary, not sufficient, condition; assumes independent
    trials with normal, serially uncorrelated returns, and the approximation
    is derived for large ``N`` (``N = 1`` returns 0.0).
    """
    target = _check_finite("target_sharpe", target_sharpe)
    if target <= 0:
        raise ValueError(f"target_sharpe must be a positive annual Sharpe ratio, got {target!r}")
    emax = expected_max_sharpe(n_trials, 1.0)
    return (emax / target) ** 2


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PBOResult:
    """Outcome of combinatorially symmetric cross-validation (CSCV).

    ``pbo`` is the share of splits where the in-sample winner ranks below the
    out-of-sample median (logit < 0; a winner exactly at the median counts as
    half). ``logits`` has one value per split. ``prob_oos_loss`` is the share
    of splits where the winner's out-of-sample metric is below zero.
    ``degradation_slope``/``degradation_intercept`` come from the OLS fit of
    the winner's out-of-sample metric on its in-sample metric across splits
    (``None`` when the in-sample metric does not vary).
    """

    pbo: float
    logits: NDArray[np.float64]
    n_combinations: int
    n_splits: int
    n_configurations: int
    n_periods: int
    prob_oos_loss: float
    degradation_slope: float | None
    degradation_intercept: float | None
    is_metric: NDArray[np.float64]
    oos_metric: NDArray[np.float64]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pbo": self.pbo,
            "logits": self.logits.tolist(),
            "n_combinations": self.n_combinations,
            "n_splits": self.n_splits,
            "n_configurations": self.n_configurations,
            "n_periods": self.n_periods,
            "prob_oos_loss": self.prob_oos_loss,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "is_metric": self.is_metric.tolist(),
            "oos_metric": self.oos_metric.tolist(),
        }

    def __str__(self) -> str:
        slope = "n/a" if self.degradation_slope is None else f"{self.degradation_slope:+.3f}"
        return (
            f"PBO {self.pbo:.3f} over {self.n_combinations} CSCV splits "
            f"(S={self.n_splits}, N={self.n_configurations} configurations, "
            f"T={self.n_periods}); P(OOS loss) {self.prob_oos_loss:.3f}; "
            f"degradation slope {slope}"
        )


def _block_bounds(n_rows: int, n_blocks: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_rows, n_blocks)
    bounds = []
    start = 0
    for b in range(n_blocks):
        size = base + (1 if b < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _cscv_sharpe(
    m: NDArray[np.float64], bounds: list[tuple[int, int]], combos: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Per-period Sharpe of every column over the union of each combination's blocks.

    Pools per-block counts, means and sums of squared deviations (the
    parallel-variance formula), which avoids the cancellation of a
    sum-of-squares shortcut and costs one pass over the data.
    """
    n_cols = m.shape[1]
    counts = np.array([b - a for a, b in bounds], dtype=np.float64)
    means = np.stack([m[a:b].mean(axis=0) for a, b in bounds])
    m2 = np.stack([((m[a:b] - means[i]) ** 2).sum(axis=0) for i, (a, b) in enumerate(bounds)])
    out = np.empty((len(combos), n_cols), dtype=np.float64)
    chunk = max(1, 2_000_000 // max(1, combos.shape[1] * n_cols))
    for lo in range(0, len(combos), chunk):
        idx = combos[lo : lo + chunk]
        cnt = counts[idx]
        n = cnt.sum(axis=1)
        mu_b = means[idx]
        mu = (cnt[:, :, None] * mu_b).sum(axis=1) / n[:, None]
        ss = m2[idx].sum(axis=1) + (cnt[:, :, None] * (mu_b - mu[:, None, :]) ** 2).sum(axis=1)
        sd = np.sqrt(ss / (n[:, None] - 1.0))
        flat = sd <= 1e-10 * np.abs(mu)
        safe = np.where(flat, 1.0, sd)
        out[lo : lo + chunk] = np.where(flat, 0.0, mu / safe)
    return out


def pbo(
    performance: ArrayLike,
    *,
    n_splits: int = 16,
    metric: Metric | None = None,
) -> PBOResult:
    """Probability of backtest overfitting by CSCV (Bailey et al., 2017).

    What it catches: a parameter search whose winner is a fluke. If the
    configuration that looks best in-sample is no better than a typical
    configuration out-of-sample, the search is fitting noise, and the PBO
    approaches or exceeds 0.5.

    Parameters
    ----------
    performance: ``T x N`` matrix, rows are periods and columns are strategy
        configurations, aligned in time (for example daily returns of every
        configuration tried). Every configuration you tried belongs here, not
        just the survivors.
    n_splits: ``S``, the number of contiguous row blocks; even, ``>= 2``, and
        ``T >= 2 * S`` so every block has at least two rows. Blocks differ in
        length by at most one row when ``T`` is not a multiple of ``S``.
        Every choice of ``S/2`` blocks is used once as the in-sample set, with
        the other half out-of-sample, so there are ``C(S, S/2)`` splits
        (12,870 for ``S = 16``).
    metric: function mapping a ``rows x N`` sub-matrix (rows kept in time
        order) to ``N`` performance values. Defaults to the per-period Sharpe
        ratio (``ddof=1``), which is 0 for a column with zero dispersion.

    Method, per split: take the in-sample winner (ties go to the lowest
    column index), compute its out-of-sample rank among all ``N`` columns
    (ascending, 1 = worst, *average* ranks for ties), the relative rank
    ``w = rank / (N + 1)`` and ``logit = ln(w / (1 - w))``. PBO is the share of
    splits with ``logit < 0``, counting ``logit == 0`` (winner exactly at the
    median, possible for odd ``N`` or ties) as half. Average ranks matter:
    ranking ties as "worse" makes identical configurations look overfit, and
    ignoring the exact median biases PBO low.

    Limitations: the columns must be comparable on the same metric, the
    blocks are treated as exchangeable (strong regime changes break this),
    and a low PBO says the selection generalised within this sample, not
    that the winner is profitable (check ``prob_oos_loss``, and the deflated
    Sharpe ratio). Raises if ``N < 2``, ``S`` is odd or too large for ``T``,
    or every column is identical, since PBO is meaningless without
    cross-sectional dispersion.
    """
    m = np.asarray(performance, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"performance must be a 2-D (T x N) matrix, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("performance must be finite (no NaN or inf)")
    t_rows, n_cols = m.shape
    if n_cols < 2:
        raise ValueError(f"PBO needs at least 2 configurations (columns), got {n_cols}")
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise ValueError(f"n_splits must be an even integer, got {n_splits!r}")
    s = int(n_splits)
    if s < 2 or s % 2:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits!r}")
    if t_rows < 2 * s:
        raise ValueError(
            f"T={t_rows} rows is too few for n_splits={s}; need T >= 2 * n_splits = {2 * s}"
        )
    scale = _unit_scale(m)
    if np.all(np.abs(m - m[:, :1]) <= _REL_TOL * scale):
        raise ValueError(
            "all configurations are identical (zero cross-sectional dispersion); PBO is undefined"
        )

    bounds = _block_bounds(t_rows, s)
    combo_list = list(itertools.combinations(range(s), s // 2))
    index = {c: i for i, c in enumerate(combo_list)}
    full = frozenset(range(s))
    complement = np.array(
        [index[tuple(sorted(full.difference(c)))] for c in combo_list], dtype=np.int64
    )
    combos = np.array(combo_list, dtype=np.int64)

    if metric is None:
        stats = _cscv_sharpe(m / scale, bounds, combos)
    else:
        stats = np.empty((len(combo_list), n_cols), dtype=np.float64)
        for i, c in enumerate(combo_list):
            rows = np.concatenate([np.arange(*bounds[b]) for b in c])
            vals = np.asarray(metric(m[rows]), dtype=np.float64)
            if vals.shape != (n_cols,):
                raise ValueError(
                    f"metric must return {n_cols} values (one per column), got shape {vals.shape}"
                )
            if not np.isfinite(vals).all():
                raise ValueError("metric returned a non-finite value")
            stats[i] = vals
    is_stats = stats
    oos_stats = stats[complement]

    rows_ix = np.arange(len(combo_list))
    is_best = is_stats.max(axis=1)
    is_tol = _REL_TOL * np.maximum(1.0, np.abs(is_best))
    winner = np.argmax(is_stats >= (is_best - is_tol)[:, None], axis=1)
    w_is = is_stats[rows_ix, winner]
    w_oos = oos_stats[rows_ix, winner]
    tol = (_REL_TOL * np.maximum(1.0, np.abs(w_oos)))[:, None]
    below = (oos_stats < w_oos[:, None] - tol).sum(axis=1)
    ties = (np.abs(oos_stats - w_oos[:, None]) <= tol).sum(axis=1)
    rank = below + (ties + 1) / 2.0
    omega = rank / (n_cols + 1.0)
    logits = np.log(omega / (1.0 - omega))
    twice = 2.0 * rank
    pbo_value = float(np.mean(twice < n_cols + 1) + 0.5 * np.mean(twice == n_cols + 1))

    x_var = float(np.var(w_is))
    slope: float | None
    intercept: float | None
    if x_var <= (_REL_TOL * max(1.0, float(np.abs(w_is).max()))) ** 2:
        slope = intercept = None
    else:
        slope = float(np.mean((w_is - w_is.mean()) * (w_oos - w_oos.mean())) / x_var)
        intercept = float(w_oos.mean() - slope * w_is.mean())

    for arr in (logits, w_is, w_oos):
        arr.setflags(write=False)
    return PBOResult(
        pbo=pbo_value,
        logits=logits,
        n_combinations=len(combo_list),
        n_splits=s,
        n_configurations=n_cols,
        n_periods=t_rows,
        prob_oos_loss=float(np.mean(w_oos < 0)),
        degradation_slope=slope,
        degradation_intercept=intercept,
        is_metric=w_is,
        oos_metric=w_oos,
    )


# ---------------------------------------------------------------------------
# Serial correlation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeweyWestResult:
    """Mean, HAC standard error and t-statistic of a series.

    Unpacks as ``mean, se, t = newey_west(x)``. ``kernel`` is the kernel
    actually used; ``fallback`` is True when a ``'uniform'`` request produced
    a non-positive variance and Bartlett weights were used instead.
    """

    mean: float
    se: float
    t: float
    lags: int
    kernel: Kernel
    requested_kernel: Kernel
    n: int
    fallback: bool

    def __iter__(self) -> Iterator[float]:
        return iter((self.mean, self.se, self.t))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "se": self.se,
            "t": self.t,
            "lags": self.lags,
            "kernel": self.kernel,
            "requested_kernel": self.requested_kernel,
            "n": self.n,
            "fallback": self.fallback,
        }

    def __str__(self) -> str:
        note = ""
        if self.fallback:
            note = " (uniform-kernel variance was not positive; fell back to Bartlett)"
        return (
            f"mean {self.mean:.6g}, HAC se {self.se:.6g}, t {self.t:.3f} "
            f"({self.kernel} kernel, {self.lags} lags, n={self.n}){note}"
        )


def _autocovariances(d: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """``gamma_j = sum_t d_t d_{t-j} / T`` for ``j = 0..max_lag`` (``d`` demeaned)."""
    t = len(d)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = float(d @ d) / t
    for j in range(1, max_lag + 1):
        out[j] = float(d[j:] @ d[:-j]) / t
    return out


def newey_west(
    x: ArrayLike, *, lags: int | None = None, kernel: Kernel = "bartlett"
) -> NeweyWestResult:
    """Mean of ``x`` with a heteroskedasticity and autocorrelation consistent (HAC) error.

    What it catches: t-statistics inflated by serial correlation, such as
    overlapping trades, multi-bar holding periods, or labels built from
    overlapping forward returns. A plain t-statistic treats each observation
    as independent news; with positive autocorrelation it overstates the
    evidence.

    The long-run variance is ``gamma_0 + 2 * sum_{j=1..L} w_j * gamma_j`` with
    autocovariances ``gamma_j`` (divisor ``T``), ``se = sqrt(lrv / T)`` and
    ``t = mean / se``.

    * ``kernel='bartlett'``: ``w_j = 1 - j / (L + 1)`` (Newey & West, 1987),
      always non-negative.
    * ``kernel='uniform'``: ``w_j = 1`` (Hansen & Hodrick, 1980). For returns
      over overlapping ``h``-period windows use ``lags = h - 1``; Bartlett
      weights with the same lags under-correct that case. If the uniform
      variance is not positive, the Bartlett estimate with the same lags is
      used and the result says so (``fallback=True``, ``kernel='bartlett'``).

    ``lags`` defaults to ``floor(4 * (T / 100) ** (2 / 9))``. Newey & West
    (1994) use this as the pilot lag from which their automatic procedure
    estimates the bandwidth; here it is the lag itself, a common rule of
    thumb. Raises for fewer than 2 observations, lags outside ``0..T-1``, or
    zero dispersion.

    Limitations: asymptotic; with few observations per lag the standard
    error is itself noisy, and a lag choice made after seeing the result is
    another degree of freedom. With strong autocorrelation the default lag
    is too short: on AR(1) data with coefficient 0.6, ``|t| > 1.96``
    happened in 12.55% (T = 250) and 10.0% (T = 1000) of null simulations
    (``docs/calibration.md``); pass a longer ``lags`` when the
    autocorrelation is known to be persistent.
    """
    v = _returns(x, "x")
    t_obs = len(v)
    if t_obs < 2:
        raise ValueError(f"x: at least 2 observations are required, got {t_obs}")
    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"kernel must be 'bartlett' or 'uniform', got {kernel!r}")
    if lags is None:
        lag = math.floor(4.0 * (t_obs / 100.0) ** (2.0 / 9.0))
        lag = min(lag, t_obs - 1)
    else:
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)):
            raise ValueError(f"lags must be a non-negative integer, got {lags!r}")
        lag = int(lags)
        if not 0 <= lag <= t_obs - 1:
            raise ValueError(f"lags must be in 0..{t_obs - 1} for {t_obs} observations, got {lag}")
    scale = _unit_scale(v)
    v = v / scale
    mean = float(v.mean())
    d = v - mean
    gamma = _autocovariances(d, lag)
    if gamma[0] <= (_REL_TOL * float(np.abs(v).max())) ** 2:
        raise ValueError("x has zero dispersion; the standard error is undefined")
    j = np.arange(1, lag + 1, dtype=np.float64)
    bartlett = float(gamma[0] + 2.0 * np.sum((1.0 - j / (lag + 1.0)) * gamma[1:]))
    used: Kernel = kernel
    fallback = False
    if kernel == "uniform":
        lrv = float(gamma[0] + 2.0 * np.sum(gamma[1:]))
        if lrv <= 0:
            lrv, used, fallback = bartlett, "bartlett", True
    else:
        lrv = bartlett
    if lrv <= 0:
        raise ValueError(
            "the HAC long-run variance is not positive; the standard error is undefined"
        )
    se = math.sqrt(lrv / t_obs)
    return NeweyWestResult(
        mean=mean * scale,
        se=se * scale,
        t=mean / se,
        lags=lag,
        kernel=used,
        requested_kernel=kernel,
        n=t_obs,
        fallback=fallback,
    )


@dataclass(frozen=True)
class LoSharpeResult:
    """Sharpe ratio aggregated over ``periods`` with Lo's (2002) correction.

    ``annualized`` is the corrected ratio ``eta * sharpe``; ``naive_annualized``
    is the usual ``sqrt(periods) * sharpe``. ``autocorrelations`` holds
    ``rho_1 .. rho_{periods-1}``.
    """

    sharpe: float
    annualized: float
    naive_annualized: float
    eta: float
    periods: int
    n: int
    autocorrelations: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "annualized": self.annualized,
            "naive_annualized": self.naive_annualized,
            "eta": self.eta,
            "periods": self.periods,
            "n": self.n,
            "autocorrelations": list(self.autocorrelations),
        }

    def __str__(self) -> str:
        return (
            f"Sharpe over {self.periods} periods {self.annualized:.3f} "
            f"(naive sqrt-scaling {self.naive_annualized:.3f}, eta {self.eta:.3f}, "
            f"per period {self.sharpe:.4f}, n={self.n})"
        )


def lo_scale_factor(autocorrelations: ArrayLike, periods: int) -> float:
    """Lo's (2002) scale factor ``eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k)``.

    ``autocorrelations`` must hold at least ``rho_1 .. rho_{q-1}`` (extra lags
    are ignored); ``q = periods``. For uncorrelated returns ``eta = sqrt(q)``.
    Raises when the variance term is not positive, which strongly negative
    autocorrelation estimates can produce.
    """
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError(f"periods must be a positive integer, got {periods!r}")
    q = int(periods)
    rho = np.asarray(autocorrelations, dtype=np.float64).ravel()
    if len(rho) < q - 1:
        raise ValueError(f"need {q - 1} autocorrelations for periods={q}, got {len(rho)}")
    if not np.isfinite(rho[: q - 1]).all():
        raise ValueError("autocorrelations must be finite")
    k = np.arange(1, q, dtype=np.float64)
    denom = q + 2.0 * float(np.sum((q - k) * rho[: q - 1]))
    if denom <= 0:
        raise ValueError(
            f"q + 2*sum((q-k)*rho_k) = {denom!r} is not positive; the autocorrelation "
            "estimates are inconsistent with a valid variance"
        )
    return q / math.sqrt(denom)


def lo_annualized_sharpe(returns: ArrayLike, periods: int) -> LoSharpeResult:
    """Sharpe ratio over ``periods`` periods, corrected for serial correlation (Lo, 2002).

    What it catches: the ``sqrt(12)`` (or ``sqrt(252)``) rule overstates the
    annual Sharpe of positively autocorrelated returns, typical of smoothed
    or illiquid marks and of strategies holding positions across many bars.

    Uses ``eta(q) * SR`` with ``SR`` the per-period Sharpe ratio (``ddof=1``),
    ``q = periods`` (e.g. 12 to annualise monthly returns) and sample
    autocorrelations ``rho_k`` (autocovariances with divisor ``T`` over the
    lag-0 one); see :func:`lo_scale_factor`.

    Limitations: assumes stationarity; autocorrelation estimates at long lags
    are noisy, so ``periods`` should be small relative to the sample (at
    least ``periods + 1`` observations are required).
    """
    x = _returns(returns)
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError(f"periods must be a positive integer, got {periods!r}")
    q = int(periods)
    if len(x) < q + 1:
        raise ValueError(f"need at least periods + 1 = {q + 1} observations, got {len(x)}")
    sr = sharpe_ratio(x)
    x = x / _unit_scale(x)
    gamma = _autocovariances(x - x.mean(), q - 1)
    rho = gamma[1:] / gamma[0]
    eta = lo_scale_factor(rho, q)
    return LoSharpeResult(
        sharpe=sr,
        annualized=eta * sr,
        naive_annualized=math.sqrt(q) * sr,
        eta=eta,
        periods=q,
        n=len(x),
        autocorrelations=tuple(float(r) for r in rho),
    )


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def adjust_pvalues(pvalues: ArrayLike, method: PMethod = "holm") -> NDArray[np.float64]:
    """Multiple-testing adjusted p-values, in the input order and shape.

    * ``'bonferroni'``: ``min(1, m * p)``.
    * ``'holm'``: step-down Holm (1979); controls the family-wise error rate
      under any dependence and is never less powerful than Bonferroni.
    * ``'bh'``: step-up Benjamini & Hochberg (1995); controls the false
      discovery rate for independent or positively dependent tests.
    * ``'by'``: Benjamini & Yekutieli (2001); BH times ``sum_{i=1}^m 1/i``,
      valid under arbitrary dependence.

    Adjusted values are monotone in the raw p-values and capped at 1. Reject
    hypothesis ``i`` at level ``alpha`` when ``adjusted[i] <= alpha``.
    Limitation: ``m`` must count every test run, including the ones that
    were dropped for looking bad.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    shape = p.shape
    flat = p.ravel()
    if not np.isfinite(flat).all() or ((flat < 0) | (flat > 1)).any():
        raise ValueError("p-values must be finite and in [0, 1]")
    if method not in ("holm", "bonferroni", "bh", "by"):
        raise ValueError(f"method must be 'holm', 'bonferroni', 'bh' or 'by', got {method!r}")
    m = len(flat)
    out: NDArray[np.float64]
    if m == 0:
        out = flat.reshape(shape).copy()
        return out
    if method == "bonferroni":
        out = np.minimum(1.0, m * flat).reshape(shape)
        return out
    order = np.argsort(flat, kind="stable")
    sorted_p = flat[order]
    i = np.arange(1, m + 1, dtype=np.float64)
    if method == "holm":
        adj_sorted = np.maximum.accumulate((m - i + 1.0) * sorted_p)
    else:
        factor = float(np.sum(1.0 / i)) if method == "by" else 1.0
        adj_sorted = np.minimum.accumulate((factor * m / i * sorted_p)[::-1])[::-1]
    adj_sorted = np.minimum(1.0, adj_sorted)
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out.reshape(shape)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` by its continued fraction (DLMF 8.17.22)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    log_front = (
        a * math.log(x)
        + b * math.log1p(-x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        - math.log(a)
    )
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    f = d
    for m in range(1, 10_000):
        num = m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m))
        for coeff in (num, -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0))):
            d = 1.0 + coeff * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + coeff / c
            c = c if abs(c) > tiny else tiny
            f *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return math.exp(log_front) * f


def _two_sided_p(t: float, df: float | None) -> float:
    """Two-sided p-value of ``|t|`` under N(0, 1) (``df=None``) or Student t."""
    t = abs(t)
    if df is None:
        return math.erfc(t / math.sqrt(2.0))
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def _two_sided_quantile(p: float, df: float | None) -> float:
    """The ``t >= 0`` whose two-sided p-value is ``p``."""
    if p >= 1.0:
        return 0.0
    if df is None:
        return -_NORMAL.inv_cdf(p / 2.0)
    lo, hi = 0.0, 1.0
    while _two_sided_p(hi, df) > p:
        lo, hi = hi, hi * 2.0
        if hi > 1e300:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _two_sided_p(mid, df) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class SharpeHaircut:
    """A Sharpe ratio before and after the Harvey & Liu (2015) multiple-testing haircut."""

    sharpe_annual: float
    haircut_sharpe: float
    haircut: float
    t_stat: float
    p_single: float
    p_multiple: float
    t_adjusted: float
    n_obs: int
    n_trials: int
    periods_per_year: float
    method: str
    distribution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sharpe_annual": self.sharpe_annual,
            "haircut_sharpe": self.haircut_sharpe,
            "haircut": self.haircut,
            "t_stat": self.t_stat,
            "p_single": self.p_single,
            "p_multiple": self.p_multiple,
            "t_adjusted": self.t_adjusted,
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "periods_per_year": self.periods_per_year,
            "method": self.method,
            "distribution": self.distribution,
        }

    def __str__(self) -> str:
        return (
            f"annual Sharpe {self.sharpe_annual:.3f} -> {self.haircut_sharpe:.3f} after "
            f"{self.n_trials} trials ({self.method}): haircut {self.haircut:.1%}; "
            f"p {self.p_single:.3g} -> {self.p_multiple:.3g}"
        )


def haircut_sharpe(
    sharpe_annual: float,
    n_obs: int,
    n_trials: int,
    *,
    periods_per_year: float,
    method: Literal["bonferroni", "sidak"] = "bonferroni",
    distribution: Literal["normal", "t"] = "t",
) -> SharpeHaircut:
    """Harvey & Liu (2015) haircut of an annual Sharpe ratio for ``n_trials`` tests.

    What it catches: a Sharpe ratio reported as if it were the only strategy
    tested, when it is the best of many.

    Steps: ``t = sharpe_annual * sqrt(n_obs / periods_per_year)`` (the
    per-period Sharpe times ``sqrt(n_obs)``); two-sided p-value ``p_S``;
    multiple-testing p-value ``p_M = min(1, N * p_S)`` (``'bonferroni'``) or
    ``1 - (1 - p_S)^N`` (``'sidak'``, HL's eq. 4 for independent tests); the
    t-statistic with two-sided p-value ``p_M``; and back to an annual Sharpe
    ratio. The haircut is ``(SR - HSR) / SR``. A negative Sharpe ratio is
    haircut towards zero by the same fraction.

    ``n_obs`` is the number of return observations, sampled
    ``periods_per_year`` times a year. ``distribution='t'`` (default) uses
    Student t with ``n_obs - 1`` degrees of freedom, as in HL's eq. (3); it
    reproduces their Exhibit 1, whose ``p^M`` column follows the Sidak form
    even though the caption says Bonferroni. ``'normal'`` uses the
    asymptotic normal distribution, which reproduces the worked example in
    their text (T = 240, SR 0.75, N = 200 gives about 0.32). The two agree
    for long samples; Student t is more conservative for short ones.

    Limitations: this is HL's single-test, independent-trials version; their
    Holm and BHY haircuts need a model of the other trials' t-statistics and
    are not implemented here. Raises for a zero Sharpe ratio, whose haircut
    fraction is undefined, and for t-statistics so large that the p-value
    underflows.
    """
    sr = _check_finite("sharpe_annual", sharpe_annual)
    if sr == 0:
        raise ValueError("sharpe_annual is 0; the haircut fraction of a zero Sharpe is undefined")
    for name, value in (("n_obs", n_obs), ("n_trials", n_trials)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    ppy = _check_finite("periods_per_year", periods_per_year)
    if ppy <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year!r}")
    if method not in ("bonferroni", "sidak"):
        raise ValueError(f"method must be 'bonferroni' or 'sidak', got {method!r}")
    if distribution not in ("normal", "t"):
        raise ValueError(f"distribution must be 'normal' or 't', got {distribution!r}")
    df = None if distribution == "normal" else float(n_obs - 1)
    years = n_obs / ppy
    t_stat = sr * math.sqrt(years)
    p_single = _two_sided_p(t_stat, df)
    if p_single <= 0.0:
        raise ValueError(
            f"t-statistic {t_stat:.1f} is so large that its p-value underflows double "
            "precision; the haircut cannot be computed (and is small)"
        )
    if method == "bonferroni":
        p_multi = min(1.0, n_trials * p_single)
    else:
        p_multi = -math.expm1(n_trials * math.log1p(-p_single)) if p_single < 1 else 1.0
    t_adj = _two_sided_quantile(p_multi, df)
    hsr = math.copysign(t_adj / math.sqrt(years), sr)
    return SharpeHaircut(
        sharpe_annual=sr,
        haircut_sharpe=hsr,
        haircut=(sr - hsr) / sr,
        t_stat=t_stat,
        p_single=p_single,
        p_multiple=p_multi,
        t_adjusted=t_adj,
        n_obs=int(n_obs),
        n_trials=int(n_trials),
        periods_per_year=ppy,
        method=method,
        distribution=distribution,
    )
