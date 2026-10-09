"""
Unit tests for visualization utilities in crunch-synth.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.evaluation.visualization import (
    plot_mixture_density,
    plot_crps_timeline,
    plot_leaderboard_ranks,
)


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_mixture_density(self):
        density_dict = {
            "type": "mixture",
            "components": [
                {
                    "weight": 1.0,
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": 0.0, "scale": 10.0},
                    },
                }
            ],
        }

        fig = plot_mixture_density(density_dict, realized_return=5.0)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # density curve + observed point

        # Test export to HTML
        export_file = Path(self.temp_dir.name) / "test_density.html"
        plot_mixture_density(density_dict, save_path=export_file)
        self.assertTrue(export_file.exists())
        self.assertGreater(export_file.stat().st_size, 0)

    def test_plot_crps_timeline(self):
        df = pd.DataFrame({
            "timestamp": [1700000000 + i * 3600 for i in range(10)],
            "datetime": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
            "score": [1.5, 1.4, 1.6, 1.2, 1.1, 1.3, 1.0, 0.9, 1.1, 0.8],
        })

        fig = plot_crps_timeline(df, asset="BTC")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # score points + rolling mean

    def test_plot_leaderboard_ranks(self):
        df_comp = pd.DataFrame([
            {"asset": "BTC", "my_score": 1.5, "benchmark_score": 1.8},
            {"asset": "ETH", "my_score": 0.8, "benchmark_score": 0.9},
        ])

        fig = plot_leaderboard_ranks(df_comp)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # my_score + benchmark_score


if __name__ == "__main__":
    unittest.main()
