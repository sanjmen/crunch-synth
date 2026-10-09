"""
Probability Integral Transform (PIT) Calibration and Dispersion Diagnostics.

Implements:
1. Analytical and numerical mixture CDF evaluation: u_t = F(y_t).
2. PIT histogram diagnostics: detects underdispersion, overdispersion, and bias.
3. Empirical scale multiplier optimization to maximize CRPS sharpness and reliability.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy.stats import norm, t as student_t, kstest, uniform
from scipy.optimize import minimize_scalar

from crunch_synth.tracker_evaluator import crps_integral, CRPS_BOUNDS


def evaluate_mixture_cdf(density_dict: dict, y: float) -> float:
    """
    Evaluates cumulative distribution function F(y) for a predictive mixture.

    Parameters
    ----------
    density_dict : dict
        Mixture dictionary complying with densitypdf / CrunchDAO format.
    y : float
        Evaluation point (realized return).

    Returns
    -------
    float
        Cumulative probability F(y) in [0.0, 1.0].
    """
    components = density_dict.get("components", [])
    if not components:
        return 0.5

    total_cdf = 0.0
    total_weight = 0.0

    for comp in components:
        w = comp.get("weight", 1.0)
        density = comp.get("density", {})
        dname = density.get("name", "norm")
        params = density.get("params", {})

        loc = float(params.get("loc", 0.0))
        scale = max(float(params.get("scale", 1.0)), 1e-8)

        if dname == "norm":
            cdf_val = float(norm.cdf(y, loc=loc, scale=scale))
        elif dname == "t":
            df = max(float(params.get("df", 5.0)), 1.1)
            cdf_val = float(student_t.cdf(y, df=df, loc=loc, scale=scale))
        elif dname == "cauchy":
            from scipy.stats import cauchy
            cdf_val = float(cauchy.cdf(y, loc=loc, scale=scale))
        elif dname == "laplace":
            from scipy.stats import laplace
            cdf_val = float(laplace.cdf(y, loc=loc, scale=scale))
        else:
            # Fallback to Gaussian approximation for unknown distributions
            cdf_val = float(norm.cdf(y, loc=loc, scale=scale))

        total_cdf += w * cdf_val
        total_weight += w

    if total_weight <= 0:
        return 0.5

    prob = total_cdf / total_weight
    return float(np.clip(prob, 0.0, 1.0))


class PITCalibrator:
    """
    Evaluates PIT reliability diagnostics and calibrates dispersion parameters
    to achieve uniform coverage and optimal CRPS.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def compute_pit_values(
        self,
        predictions: List[dict],
        realized_returns: Union[np.ndarray, List[float]],
    ) -> np.ndarray:
        """
        Computes PIT values u_t = F_t(y_t) for a series of forecasts and outcomes.
        """
        y = np.asarray(realized_returns, dtype=np.float64)
        n = min(len(predictions), len(y))
        if n == 0:
            return np.empty(0, dtype=np.float64)

        u_values = np.zeros(n, dtype=np.float64)
        for i in range(n):
            u_values[i] = evaluate_mixture_cdf(predictions[i], y[i])

        return u_values

    def diagnose_calibration(self, pit_values: np.ndarray) -> Dict[str, Any]:
        """
        Diagnoses forecast reliability from empirical PIT values:
        - Flat histogram -> Well-calibrated
        - U-shaped -> Underdispersed (forecasts too narrow / overconfident)
        - Hump-shaped -> Overdispersed (forecasts too wide / underconfident)
        - Tilted -> Bias in mean / drift
        """
        u = np.asarray(pit_values, dtype=np.float64)
        n = len(u)
        if n < 10:
            return {
                "status": "INSUFFICIENT_DATA",
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
                "mean_pit": 0.5,
                "tail_ratio": 1.0,
                "center_ratio": 1.0,
            }

        # Kolmogorov-Smirnov test vs Uniform(0, 1)
        ks_res = kstest(u, uniform(0, 1).cdf)

        # Tail proportion (u < 0.1 or u > 0.9) - expected 0.20
        tail_count = np.sum((u < 0.1) | (u > 0.9))
        tail_prop = tail_count / n
        tail_ratio = tail_prop / 0.20  # > 1 means underdispersed (too narrow)

        # Center proportion (0.4 <= u <= 0.6) - expected 0.20
        center_count = np.sum((u >= 0.4) & (u <= 0.6))
        center_prop = center_count / n
        center_ratio = center_prop / 0.20  # > 1 means overdispersed (too wide)

        mean_pit = float(np.mean(u))  # Expected 0.50

        # Diagnosis heuristic
        if abs(mean_pit - 0.50) > 0.08:
            status = "BIASED_DRIFT"
        elif tail_ratio > 1.35:
            status = "UNDERDISPERSED"  # Needs scaling multiplier > 1
        elif center_ratio > 1.35:
            status = "OVERDISPERSED"   # Needs scaling multiplier < 1
        else:
            status = "WELL_CALIBRATED"

        return {
            "status": status,
            "sample_size": n,
            "ks_stat": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
            "mean_pit": round(mean_pit, 4),
            "tail_ratio": round(tail_ratio, 3),
            "center_ratio": round(center_ratio, 3),
        }

    def rescale_density_dict(self, density_dict: dict, multiplier: float) -> dict:
        """
        Applies a multiplicative factor to all scale parameters in the mixture.
        """
        if multiplier <= 0 or abs(multiplier - 1.0) < 1e-4:
            return density_dict

        rescaled = {
            "type": density_dict.get("type", "mixture"),
            "components": [],
        }

        for comp in density_dict.get("components", []):
            new_comp = {
                "weight": comp.get("weight", 1.0),
                "density": {
                    "type": comp.get("density", {}).get("type", "builtin"),
                    "name": comp.get("density", {}).get("name", "norm"),
                    "params": dict(comp.get("density", {}).get("params", {})),
                },
            }
            if "scale" in new_comp["density"]["params"]:
                orig_scale = float(new_comp["density"]["params"]["scale"])
                new_comp["density"]["params"]["scale"] = max(orig_scale * multiplier, 1e-6)

            rescaled["components"].append(new_comp)

        return rescaled

    def optimize_dispersion_multiplier(
        self,
        predictions: List[dict],
        realized_returns: Union[np.ndarray, List[float]],
        asset: str = "BTC",
        step: int = 300,
        bracket: Tuple[float, float] = (0.5, 2.5),
    ) -> float:
        """
        Finds the optimal variance scaling multiplier s* that minimizes average CRPS.
        """
        y = np.asarray(realized_returns, dtype=np.float64)
        n = min(len(predictions), len(y))
        if n == 0:
            return 1.0

        # CRPS bounds for asset
        t_bound = CRPS_BOUNDS["t"].get(asset, 50.0)
        k_scale = np.sqrt(step / CRPS_BOUNDS["base_step"]) if step > CRPS_BOUNDS["base_step"] else 1
        t_min = -k_scale * t_bound
        t_max = k_scale * t_bound
        num_pts = min(CRPS_BOUNDS["num_points"], 400)  # Faster quadrature for optimization

        # Subset up to 50 samples for speed
        if n > 50:
            indices = np.linspace(0, n - 1, 50, dtype=int)
            sub_preds = [predictions[i] for i in indices]
            sub_y = y[indices]
        else:
            sub_preds = predictions[:n]
            sub_y = y[:n]

        def loss_fn(s: float) -> float:
            scores = []
            for pred, target in zip(sub_preds, sub_y):
                scaled_pred = self.rescale_density_dict(pred, s)
                crps_val = crps_integral(
                    density_dict=scaled_pred,
                    x=float(target),
                    t_min=t_min,
                    t_max=t_max,
                    num_points=num_pts,
                )
                scores.append(crps_val)
            return float(np.mean(scores))

        res = minimize_scalar(loss_fn, bounds=bracket, method="bounded")
        if res.success:
            return float(np.clip(res.x, 0.5, 3.0))
        return 1.0
