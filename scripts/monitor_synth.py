#!/usr/bin/env python3
"""
CLI script to inspect CrunchDAO Synth live production status and submission records.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.dashboard import SynthProductionDashboard


def main():
    dashboard = SynthProductionDashboard()
    report = dashboard.format_terminal_dashboard()
    print(report)


if __name__ == "__main__":
    main()
