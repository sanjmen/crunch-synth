"""
Production Entrypoint for CrunchDAO Synth Competition.

This file exposes the Tracker class used by CrunchDAO's execution harness.
"""

import os
from pathlib import Path
import sys
from typing import List, Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trackers.gaussian_baseline import GaussianBaselineTracker


class ProductionTracker(GaussianBaselineTracker):
    """
    Primary Tracker entrypoint submitted to CrunchDAO Synth platform.
    Inherits from GaussianBaselineTracker.
    """
    def __init__(self):
        super().__init__(
            lookback_days=5,
            resolution_seconds=300,
            min_samples=5,
            default_scale=1.0,
        )


# CrunchDAO runner compatibility alias
Tracker = ProductionTracker


if __name__ == "__main__":
    import sys
    print("Testing ProductionTracker initialization...")
    tracker = ProductionTracker()
    print("ProductionTracker successfully initialized.")
    sys.exit(0)
