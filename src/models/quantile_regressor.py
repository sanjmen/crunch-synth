"""
Empirical Quantile Regression Forecaster for Density Estimation.

Trains gradient boosted quantile trees (minimizing asymmetric pinball loss)
across multiple quantiles [0.10, 0.25, 0.50, 0.75, 0.90] to estimate
non-parametric empirical return distributions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.volatility import garman_klass_volatility, parkinson_volatility
from src.features.microstructure import corwin_schultz_spread


class EmpiricalQuantileForecaster:
    """
    Fits multi-quantile gradient boosted models to predict empirical quantiles
    of incremental returns and synthesizes density mixture parameters.
    """

    DEFAULT_QUANTILES: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        max_iter: int = 40,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ):
        self.quantiles = sorted(quantiles or self.DEFAULT_QUANTILES)
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.is_fitted = False

    def extract_features_and_targets(
        self,
        ohlc_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds feature matrix X and target y (next-bar price increment).
        """
        if len(ohlc_df) < 15:
            return np.empty((0, 5)), np.empty((0,))

        closes = ohlc_df["close"].to_numpy()
        highs = ohlc_df["high"].to_numpy()
        lows = ohlc_df["low"].to_numpy()
        opens = ohlc_df["open"].to_numpy()

        n = len(closes)
        features = []
        targets = []

        # Use rolling window of 6 bars to compute predictive features
        window = 6
        for i in range(window, n - 1):
            sub_c = closes[i - window : i]
            sub_h = highs[i - window : i]
            sub_l = lows[i - window : i]
            sub_o = opens[i - window : i]

            # Features:
            # 1. Recent GK volatility
            gk = garman_klass_volatility(sub_o, sub_h, sub_l, sub_c)
            # 2. Recent Parkinson volatility
            pv = parkinson_volatility(sub_h, sub_l)
            # 3. Momentum / short trend
            mom = (sub_c[-1] - sub_c[0]) / max(sub_c[0], 1e-4)
            # 4. Immediate prior bar return
            last_ret = (sub_c[-1] - sub_c[-2]) / max(sub_c[-2], 1e-4)
            # 5. Spread estimate
            spread = corwin_schultz_spread(sub_h, sub_l)[-1]

            feat_vec = [gk, pv, mom, last_ret, spread]
            # Target: next-step price increment
            target_inc = closes[i + 1] - closes[i]

            features.append(feat_vec)
            targets.append(target_inc)

        return np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64)

    def fit(self, ohlc_df: pd.DataFrame) -> "EmpiricalQuantileForecaster":
        """
        Fits separate quantile regressors for each target quantile.
        """
        X, y = self.extract_features_and_targets(ohlc_df)
        if len(X) < 25:
            self.is_fitted = False
            return self

        for q in self.quantiles:
            reg = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            reg.fit(X, y)
            self.models[q] = reg

        self.is_fitted = True
        return self

    def predict_quantiles(self, latest_ohlc: pd.DataFrame) -> Dict[float, float]:
        """
        Predicts quantiles for the next forward period.
        """
        if not self.is_fitted or len(latest_ohlc) < 6:
            return {}

        closes = latest_ohlc["close"].to_numpy()
        highs = latest_ohlc["high"].to_numpy()
        lows = latest_ohlc["low"].to_numpy()
        opens = latest_ohlc["open"].to_numpy()

        sub_c = closes[-6:]
        sub_h = highs[-6:]
        sub_l = lows[-6:]
        sub_o = opens[-6:]

        gk = garman_klass_volatility(sub_o, sub_h, sub_l, sub_c)
        pv = parkinson_volatility(sub_h, sub_l)
        mom = (sub_c[-1] - sub_c[0]) / max(sub_c[0], 1e-4)
        last_ret = (sub_c[-1] - sub_c[-2]) / max(sub_c[-2], 1e-4)
        spread = corwin_schultz_spread(sub_h, sub_l)[-1]

        x_latest = np.array([[gk, pv, mom, last_ret, spread]], dtype=np.float64)

        pred_q = {}
        for q, reg in self.models.items():
            val = float(reg.predict(x_latest)[0])
            pred_q[q] = val

        # Ensure monotonicity of quantiles (monotonic sorting fix)
        sorted_qs = sorted(pred_q.keys())
        sorted_vals = sorted([pred_q[q] for q in sorted_qs])
        for q, val in zip(sorted_qs, sorted_vals):
            pred_q[q] = val

        return pred_q

    def quantiles_to_mixture_parameters(
        self,
        predicted_quantiles: Dict[float, float],
        fallback_scale: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Translates predicted quantiles into a 2-component Gaussian mixture representation:
        (w_core, mu_core, sigma_core, w_tail, mu_tail, sigma_tail)
        """
        if not predicted_quantiles or len(predicted_quantiles) < 3:
            w_core, w_tail = 0.75, 0.25
            mu_core, mu_tail = 0.0, 0.0
            sigma_core, sigma_tail = fallback_scale, fallback_scale * 2.2
            return w_core, mu_core, sigma_core, w_tail, mu_tail, sigma_tail

        q10 = predicted_quantiles.get(0.10, -fallback_scale * 1.64)
        q25 = predicted_quantiles.get(0.25, -fallback_scale * 0.67)
        q50 = predicted_quantiles.get(0.50, 0.0)
        q75 = predicted_quantiles.get(0.75, fallback_scale * 0.67)
        q90 = predicted_quantiles.get(0.90, fallback_scale * 1.64)

        # Core body scale from IQR
        iqr = max(q75 - q25, 1e-6)
        sigma_core = max(iqr / 1.349, fallback_scale * 0.5, 1e-4)
        mu_core = float(q50 * 0.1)  # shrink median drift

        # Tail scale from 10-90 spread
        tail_spread = max(q90 - q10, 1e-6)
        sigma_tail = max(tail_spread / 2.56, sigma_core * 1.8, 1e-4)

        # Bowley asymmetry / skew
        bowley_num = (q75 + q25 - 2.0 * q50)
        skew = float(bowley_num / iqr) if iqr > 1e-6 else 0.0
        skew_shift = np.clip(skew * sigma_core * 0.2, -sigma_core, sigma_core)
        mu_tail = float(mu_core + skew_shift)

        w_core = 0.75
        w_tail = 0.25

        return w_core, mu_core, sigma_core, w_tail, mu_tail, sigma_tail
