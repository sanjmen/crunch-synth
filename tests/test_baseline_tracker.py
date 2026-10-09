"""
Unit tests for GaussianBaselineTracker.
"""

import time
import pytest
import numpy as np

from src.trackers.gaussian_baseline import GaussianBaselineTracker


def test_tracker_initialization():
    tracker = GaussianBaselineTracker()
    assert tracker.lookback_days == 5
    assert tracker.resolution_seconds == 300
    assert tracker.prices is not None


def test_predict_without_data():
    tracker = GaussianBaselineTracker()
    # Should safely return empty list or fallback when no prices are ingested
    result = tracker.predict("BTC", horizon=3600, step=300)
    assert result == []


def test_predict_with_synthetic_prices():
    tracker = GaussianBaselineTracker()

    base_time = 1700000000
    base_price = 50000.0

    # Ingest 100 ticks of data at 60s intervals
    ticks = []
    current_price = base_price
    for i in range(100):
        current_price += np.random.normal(0, 10.0)
        ticks.append((base_time + i * 60, current_price))

    tracker.tick({"BTC": ticks})

    # Test 1h horizon with 5m step (12 steps)
    start_time = time.time()
    predictions = tracker.predict("BTC", horizon=3600, step=300)
    elapsed = time.time() - start_time

    assert elapsed < 0.1, f"Prediction took too long: {elapsed:.4f}s"
    assert len(predictions) == 12

    for i, pred in enumerate(predictions):
        assert pred["step"] == (i + 1) * 300
        assert pred["type"] == "mixture"
        assert len(pred["components"]) == 1

        comp = pred["components"][0]
        assert comp["weight"] == 1.0
        assert comp["density"]["type"] == "builtin"
        assert comp["density"]["name"] == "norm"
        assert "loc" in comp["density"]["params"]
        assert "scale" in comp["density"]["params"]
        assert comp["density"]["params"]["scale"] > 0


def test_predict_all_integration():
    tracker = GaussianBaselineTracker()
    base_time = 1700000000
    ticks = [(base_time + i * 60, 2000.0 + i * 0.5) for i in range(50)]
    tracker.tick({"ETH": ticks})

    steps = [300, 3600]
    all_preds = tracker.predict_all("ETH", horizon=3600, steps=steps)

    assert 300 in all_preds
    assert 3600 in all_preds
    assert len(all_preds[300]) == 12
    assert len(all_preds[3600]) == 1
