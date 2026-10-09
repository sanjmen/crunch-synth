"""
High-Efficiency Volatility Estimators and Online Recursive Variance Filters.

Implements:
1. OHLC resampling from raw tick streams.
2. Realized Volatility (close-to-close).
3. Parkinson Volatility (high-low range).
4. Garman-Klass Volatility (OHLC).
5. Rogers-Satchell Volatility (drift-independent OHLC).
6. Online EWMA Filter (RiskMetrics).
7. Online GARCH(1,1) Recursive Filter with multi-step term structure projection.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def ticks_to_ohlc(
    ticks: Union[List[Tuple[int, float]], pd.DataFrame],
    bar_seconds: int = 300,
) -> pd.DataFrame:
    """
    Resamples raw (timestamp, price) ticks into uniform OHLC candles.

    Parameters
    ----------
    ticks : List[Tuple[int, float]] or pd.DataFrame
        Stream of (timestamp_seconds, price).
    bar_seconds : int
        Bar duration in seconds (default: 300s = 5m).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'count'].
    """
    if isinstance(ticks, list):
        if not ticks:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "count"])
        df = pd.DataFrame(ticks, columns=["timestamp", "price"])
    else:
        df = ticks[["timestamp", "price"]].copy()

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "count"])

    df["bar_ts"] = (df["timestamp"] // bar_seconds) * bar_seconds

    grouped = df.groupby("bar_ts")["price"]
    ohlc = pd.DataFrame({
        "timestamp": grouped.first().index,
        "open": grouped.first().values,
        "high": grouped.max().values,
        "low": grouped.min().values,
        "close": grouped.last().values,
        "count": grouped.count().values,
    }).sort_values("timestamp").reset_index(drop=True)

    return ohlc


def realized_volatility(
    prices: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 288,  # 5-min bars in a crypto year
) -> float:
    """
    Computes classical Close-to-Close Realized Volatility from price series.
    sigma_RV = sqrt( sum(r_i^2) / N )
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 2:
        return 0.0

    # Log returns
    rets = np.diff(np.log(p))
    rv_var = np.mean(rets**2)
    vol = np.sqrt(max(rv_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def parkinson_volatility(
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 288,
) -> float:
    """
    Computes Parkinson (1980) High-Low Range Volatility estimator.
    sigma_P^2 = (1 / (4 * ln(2) * N)) * sum( (ln(H/L))^2 )

    Statistically ~5x more efficient than Close-to-Close volatility.
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)

    if len(h) == 0 or len(l) == 0 or len(h) != len(l):
        return 0.0

    valid = (h > 0) & (l > 0) & (h >= l)
    if not np.any(valid):
        return 0.0

    h_val = h[valid]
    l_val = l[valid]

    hl_ratio = np.log(h_val / l_val)
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * np.mean(hl_ratio**2)
    vol = np.sqrt(max(variance, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def garman_klass_volatility(
    open_: Union[np.ndarray, pd.Series, List[float]],
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    close: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 288,
) -> float:
    """
    Computes Garman-Klass (1980) OHLC Volatility estimator.
    sigma_GK^2 = (1/N) * sum( 0.5*(ln(H/L))^2 - (2*ln(2) - 1)*(ln(C/O))^2 )

    Statistically ~8x more efficient than Close-to-Close volatility.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = len(o)
    if n == 0 or not (len(h) == len(l) == len(c) == n):
        return 0.0

    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    if not np.any(valid):
        return 0.0

    hl = np.log(h[valid] / l[valid])
    co = np.log(c[valid] / o[valid])

    const = 2.0 * np.log(2.0) - 1.0  # ~0.386294
    terms = 0.5 * (hl**2) - const * (co**2)
    variance = np.mean(terms)
    vol = np.sqrt(max(variance, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def rogers_satchell_volatility(
    open_: Union[np.ndarray, pd.Series, List[float]],
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    close: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 288,
) -> float:
    """
    Computes Rogers-Satchell (1991) Volatility estimator.
    Handles non-zero drift:
    sigma_RS^2 = (1/N) * sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = len(o)
    if n == 0 or not (len(h) == len(l) == len(c) == n):
        return 0.0

    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    if not np.any(valid):
        return 0.0

    h_val, l_val, c_val, o_val = h[valid], l[valid], c[valid], o[valid]
    terms = np.log(h_val / c_val) * np.log(h_val / o_val) + np.log(l_val / c_val) * np.log(l_val / o_val)
    variance = np.mean(terms)
    vol = np.sqrt(max(variance, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


class OnlineEWMAVolatility:
    """
    Online recursive Exponentially Weighted Moving Average (RiskMetrics) filter.
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    Runs in O(1) time per incremental observation.
    """

    def __init__(self, decay: float = 0.94, initial_variance: float = 1e-4):
        self.decay = decay
        self.variance = max(initial_variance, 1e-12)
        self.initialized = False

    def warm_up(self, returns: Union[np.ndarray, List[float]]) -> "OnlineEWMAVolatility":
        """Initializes state over an initial batch of returns."""
        rets = np.asarray(returns, dtype=np.float64)
        if len(rets) == 0:
            return self

        # Initialize with sample variance
        self.variance = float(np.var(rets)) if len(rets) > 1 else float(rets[0] ** 2)
        self.variance = max(self.variance, 1e-12)

        # Run recursive filter over warm-up
        for r in rets:
            self.update(float(r))

        self.initialized = True
        return self

    def update(self, return_val: float) -> float:
        """
        Updates the variance estimate with a single new return observation r_t.
        Returns the updated standard deviation sigma_t.
        """
        r2 = return_val**2
        self.variance = self.decay * self.variance + (1.0 - self.decay) * r2
        self.variance = max(self.variance, 1e-12)
        self.initialized = True
        return float(np.sqrt(self.variance))

    @property
    def current_volatility(self) -> float:
        return float(np.sqrt(self.variance))


class OnlineGARCH11:
    """
    Online recursive GARCH(1,1) Volatility Filter with forward term-structure projection.
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters
    ----------
    omega : float
        Base baseline constant (default: 1e-6).
    alpha : float
        Reaction to shocks / ARCH parameter (default: 0.10).
    beta : float
        Persistence / GARCH parameter (default: 0.85).
    initial_variance : float
        Starting variance.
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.10,
        beta: float = 0.85,
        initial_variance: Optional[float] = None,
    ):
        if alpha + beta >= 1.0:
            raise ValueError(f"Stationarity requires alpha + beta < 1 (got {alpha + beta})")

        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.persistence = alpha + beta
        self.unconditional_variance = omega / (1.0 - self.persistence)

        self.variance = initial_variance if initial_variance is not None else self.unconditional_variance
        self.variance = max(self.variance, 1e-12)

    def warm_up(self, returns: Union[np.ndarray, List[float]]) -> "OnlineGARCH11":
        """Initializes recursive state using historical returns."""
        rets = np.asarray(returns, dtype=np.float64)
        if len(rets) == 0:
            return self

        sample_var = float(np.var(rets)) if len(rets) > 1 else self.unconditional_variance
        self.variance = max(sample_var, 1e-12)

        for r in rets:
            self.update(float(r))

        return self

    def update(self, return_val: float) -> float:
        """
        Recursive step: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2.
        Returns the instantaneous volatility sigma_t.
        """
        r2 = return_val**2
        self.variance = self.omega + self.alpha * r2 + self.beta * self.variance
        self.variance = max(self.variance, 1e-12)
        return float(np.sqrt(self.variance))

    @property
    def current_volatility(self) -> float:
        return float(np.sqrt(self.variance))

    def forecast_step_variance(self, k: int) -> float:
        """
        Forecasts expected per-bar variance k steps into the future:
        E[sigma_{t+k}^2] = V_L + (alpha + beta)^k * (sigma_t^2 - V_L)
        """
        if k <= 0:
            return self.variance

        vl = self.unconditional_variance
        decay_factor = self.persistence**k
        forecasted_var = vl + decay_factor * (self.variance - vl)
        return float(max(forecasted_var, 1e-12))

    def forecast_cumulative_volatility(self, k_steps: int) -> float:
        """
        Forecasts total integrated volatility across k steps:
        Sigma(k) = sqrt( sum_{i=1}^k E[sigma_{t+i}^2] )
        """
        if k_steps <= 1:
            return float(np.sqrt(self.variance))

        vl = self.unconditional_variance
        phi = self.persistence
        diff = self.variance - vl

        # Closed-form geometric series summation
        geom_sum = (phi * (1.0 - phi**k_steps)) / (1.0 - phi)
        total_var = k_steps * vl + diff * geom_sum
        return float(np.sqrt(max(total_var, 1e-12)))
