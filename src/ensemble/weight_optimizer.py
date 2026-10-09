"""
CRPS-Minimizing Ensemble Weight Optimizer and Multi-Expert Mixture Blender.

Optimizes convex combination weights across expert predictive distributions:
    f_ensemble(x) = sum_{m=1}^M w_m * f_m(x),  s.t. sum(w_m) = 1, w_m >= 0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy.optimize import minimize

from crunch_synth.tracker_evaluator import crps_integral, CRPS_BOUNDS


def blend_expert_mixtures(
    expert_mixtures: List[dict],
    weights: Union[np.ndarray, List[float]],
) -> dict:
    """
    Combines multiple predictive mixture distributions into a unified convex mixture.

    Parameters
    ----------
    expert_mixtures : List[dict]
        List of density dictionaries from individual expert models.
    weights : array-like
        Convex weights corresponding to each expert model.

    Returns
    -------
    dict
        Consolidated mixture dictionary conforming to densitypdf specification.
    """
    w_arr = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(w_arr)
    if w_sum > 0:
        w_norm = w_arr / w_sum
    else:
        w_norm = np.ones(len(w_arr)) / len(w_arr)

    combined_components = []

    for expert_dict, expert_w in zip(expert_mixtures, w_norm):
        if expert_w <= 1e-4:
            continue

        comps = expert_dict.get("components", [])
        if not comps:
            continue

        # Sub-weights inside this expert
        sub_w_sum = sum(c.get("weight", 1.0) for c in comps)
        if sub_w_sum <= 0:
            sub_w_sum = 1.0

        for comp in comps:
            comp_w = comp.get("weight", 1.0) / sub_w_sum
            final_weight = float(expert_w * comp_w)

            combined_components.append({
                "weight": final_weight,
                "density": comp.get("density", {}),
            })

    # Normalize final weights
    tot = sum(c["weight"] for c in combined_components)
    if tot > 0:
        for c in combined_components:
            c["weight"] = float(c["weight"] / tot)

    step_val = expert_mixtures[0].get("step", 300) if expert_mixtures else 300

    return {
        "step": step_val,
        "type": "mixture",
        "components": combined_components,
    }


class CRPSWeightOptimizer:
    """
    Optimizes convex ensemble combination weights w* to minimize empirical CRPS loss
    using Sequential Least Squares Programming (SLSQP).
    """

    HORIZON_DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
        # 1-hour horizon: high-frequency, noise-sensitive, favors adaptive vol and Student-t
        "1h": {
            "AdaptiveVolatilityTracker": 0.35,
            "StudentTTracker": 0.30,
            "GaussianMixtureTracker": 0.20,
            "QuantileMixtureTracker": 0.15,
        },
        # 24-hour horizon: macro shifts, regime changes, favors Student-t and GMM
        "24h": {
            "StudentTTracker": 0.40,
            "GaussianMixtureTracker": 0.30,
            "AdaptiveVolatilityTracker": 0.20,
            "QuantileMixtureTracker": 0.10,
        },
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_default_weights(
        self,
        horizon_profile: str,
        model_names: List[str],
    ) -> np.ndarray:
        """
        Retrieves horizon-specific prior weights normalized across requested model names.
        """
        preset = self.HORIZON_DEFAULT_WEIGHTS.get(horizon_profile, self.HORIZON_DEFAULT_WEIGHTS["24h"])
        raw = [preset.get(name, 1.0 / len(model_names)) for name in model_names]
        tot = sum(raw)
        return np.array(raw, dtype=np.float64) / tot

    def optimize_weights(
        self,
        predictions_matrix: List[List[dict]],
        realized_targets: Union[np.ndarray, List[float]],
        asset: str = "BTC",
        step: int = 300,
        init_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Optimizes mixture weights w minimizing empirical CRPS.

        Parameters
        ----------
        predictions_matrix : List[List[dict]]
            predictions_matrix[sample_idx][model_idx] contains predicted mixture dict.
        realized_targets : array-like
            Target realized return for each sample.
        asset : str
            Asset symbol for CRPS integration bounds.
        step : int
            Forecast step in seconds.
        init_weights : Optional[np.ndarray]
            Initial weight vector.
        """
        n_samples = len(predictions_matrix)
        if n_samples == 0:
            return np.array([1.0])

        m_models = len(predictions_matrix[0])
        if m_models <= 1:
            return np.array([1.0])

        y = np.asarray(realized_targets, dtype=np.float64)

        t_bound = CRPS_BOUNDS["t"].get(asset, 50.0)
        k_scale = np.sqrt(step / CRPS_BOUNDS["base_step"]) if step > CRPS_BOUNDS["base_step"] else 1
        t_min = -k_scale * t_bound
        t_max = k_scale * t_bound
        num_pts = CRPS_BOUNDS["num_points"]

        # Downsample if dataset is large for optimizer speed
        max_eval = 40
        if n_samples > max_eval:
            indices = np.linspace(0, n_samples - 1, max_eval, dtype=int)
            sub_preds = [predictions_matrix[i] for i in indices]
            sub_y = y[indices]
        else:
            sub_preds = predictions_matrix
            sub_y = y

        def objective(w: np.ndarray) -> float:
            scores = []
            for sample_preds, target in zip(sub_preds, sub_y):
                blended = blend_expert_mixtures(sample_preds, w)
                crps_val = crps_integral(
                    density_dict=blended,
                    x=float(target),
                    t_min=t_min,
                    t_max=t_max,
                    num_points=num_pts,
                )
                scores.append(crps_val)
            return float(np.mean(scores))

        w0 = init_weights if init_weights is not None else np.ones(m_models) / m_models
        bounds = [(0.02, 0.90) for _ in range(m_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-4},
        )

        if res.success:
            w_opt = res.x
            return np.clip(w_opt / np.sum(w_opt), 0.0, 1.0)

        return w0 / np.sum(w0)
