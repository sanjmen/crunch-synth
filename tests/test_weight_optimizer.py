"""
Unit tests for CRPSWeightOptimizer and blend_expert_mixtures.
"""

import unittest
import numpy as np

from src.ensemble.weight_optimizer import CRPSWeightOptimizer, blend_expert_mixtures


class TestWeightOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = CRPSWeightOptimizer()

        self.mix1 = {
            "step": 300,
            "type": "mixture",
            "components": [
                {
                    "weight": 1.0,
                    "density": {"type": "builtin", "name": "norm", "params": {"loc": 0.0, "scale": 5.0}},
                }
            ],
        }

        self.mix2 = {
            "step": 300,
            "type": "mixture",
            "components": [
                {
                    "weight": 1.0,
                    "density": {"type": "builtin", "name": "t", "params": {"df": 4.0, "loc": 0.0, "scale": 8.0}},
                }
            ],
        }

    def test_blend_expert_mixtures(self):
        blended = blend_expert_mixtures([self.mix1, self.mix2], weights=[0.6, 0.4])
        self.assertEqual(len(blended["components"]), 2)
        total_w = sum(c["weight"] for c in blended["components"])
        self.assertAlmostEqual(total_w, 1.0, places=5)
        self.assertAlmostEqual(blended["components"][0]["weight"], 0.6, places=5)
        self.assertAlmostEqual(blended["components"][1]["weight"], 0.4, places=5)

    def test_get_default_weights(self):
        models = [
            "StudentTTracker",
            "GaussianMixtureTracker",
            "AdaptiveVolatilityTracker",
            "QuantileMixtureTracker",
        ]
        w_1h = self.optimizer.get_default_weights("1h", models)
        w_24h = self.optimizer.get_default_weights("24h", models)

        self.assertEqual(len(w_1h), 4)
        self.assertEqual(len(w_24h), 4)
        self.assertAlmostEqual(np.sum(w_1h), 1.0, places=5)
        self.assertAlmostEqual(np.sum(w_24h), 1.0, places=5)

    def test_optimize_weights_convergence(self):
        # Model 1 has realistic scale (5.0), Model 2 has severe bias and scale (100.0)
        # Optimizer should assign maximal weight to Model 1
        mix_bad = {
            "step": 300,
            "type": "mixture",
            "components": [
                {
                    "weight": 1.0,
                    "density": {"type": "builtin", "name": "norm", "params": {"loc": 50.0, "scale": 100.0}},
                }
            ],
        }
        n = 10
        preds_matrix = [[self.mix1, mix_bad] for _ in range(n)]
        targets = [0.5, -1.0, 1.2, -0.8, 0.2, 0.0, 1.5, -1.2, 0.3, -0.4]

        w_opt = self.optimizer.optimize_weights(
            preds_matrix, targets, asset="BTC", step=300
        )
        self.assertEqual(len(w_opt), 2)
        self.assertAlmostEqual(np.sum(w_opt), 1.0, places=3)
        # Mix 1 should receive higher weight than the overly wide and biased model
        self.assertGreater(w_opt[0], w_opt[1])


if __name__ == "__main__":
    unittest.main()
