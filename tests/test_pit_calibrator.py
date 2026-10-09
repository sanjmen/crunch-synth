"""
Unit tests for PITCalibrator and mixture CDF evaluation.
"""

import unittest
import numpy as np

from src.calibration.pit_calibrator import PITCalibrator, evaluate_mixture_cdf


class TestPITCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = PITCalibrator()

        self.mixture = {
            "type": "mixture",
            "components": [
                {
                    "weight": 0.8,
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": 0.0, "scale": 5.0},
                    },
                },
                {
                    "weight": 0.2,
                    "density": {
                        "type": "builtin",
                        "name": "t",
                        "params": {"df": 4.0, "loc": 0.0, "scale": 10.0},
                    },
                },
            ],
        }

    def test_evaluate_mixture_cdf(self):
        # Symmetry: CDF at loc (0.0) must be 0.5
        cdf_center = evaluate_mixture_cdf(self.mixture, 0.0)
        self.assertAlmostEqual(cdf_center, 0.5, places=5)

        # Monotonicity
        cdf_neg = evaluate_mixture_cdf(self.mixture, -10.0)
        cdf_pos = evaluate_mixture_cdf(self.mixture, 10.0)
        self.assertLess(cdf_neg, 0.5)
        self.assertGreater(cdf_pos, 0.5)
        self.assertGreater(cdf_pos, cdf_neg)

    def test_compute_pit_values(self):
        preds = [self.mixture] * 20
        # Draw from standard normal * 5.0
        y_vals = np.random.normal(0, 5.0, size=20)
        pit_vals = self.calibrator.compute_pit_values(preds, y_vals)

        self.assertEqual(len(pit_vals), 20)
        self.assertTrue((pit_vals >= 0.0).all())
        self.assertTrue((pit_vals <= 1.0).all())

    def test_diagnose_calibration_underdispersed(self):
        # Extreme U-shape: all values in tails
        u_tails = np.array([0.01] * 20 + [0.99] * 20)
        diag = self.calibrator.diagnose_calibration(u_tails)
        self.assertEqual(diag["status"], "UNDERDISPERSED")

    def test_rescale_density_dict(self):
        rescaled = self.calibrator.rescale_density_dict(self.mixture, 2.0)
        orig_scale0 = self.mixture["components"][0]["density"]["params"]["scale"]
        new_scale0 = rescaled["components"][0]["density"]["params"]["scale"]
        self.assertAlmostEqual(new_scale0, orig_scale0 * 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
