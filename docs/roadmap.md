# Project Roadmap: Crunch-Synth Probabilistic Trading System

This roadmap outlines the phased development of the quantitative modeling and deployment pipeline for the **CrunchDAO Synth** competition (Bittensor Subnet 50).

```mermaid
flowchart LR
    M1["M1: Baseline & Foundations"] --> M2["M2: Validation Harness"]
    M2 --> M3["M3: Volatility Engineering"]
    M3 --> M4["M4: Heavy-Tails & Mixtures"]
    M4 --> M5["M5: Multi-Horizon Meta-Ensemble"]
    M5 --> M6["M6: Production Deployment & Live Mining"]
```

---

## Milestone 1: Foundations, Local Price Engine & Gaussian Step Baseline
*Goal: Establish repo structure, data pipelines, and a fully compliant Gaussian baseline model.*

- **Issue #1**: Setup core repository structure, virtualenv, and dependencies (`crunch-synth`, `densitypdf`, `scipy`).
- **Issue #2**: Implement modular `GaussianBaselineTracker` subclassing `TrackerBase`.
- **Issue #3**: Verify `density_pdf` dictionary format compliance and sub-second inference latency.
- **Issue #4**: Configure production `main.py` entrypoint and initial Git commit.

---

## Milestone 2: Local Validation Harness & Fast CRPS Scoring
*Goal: Build an offline backtesting framework to score models before submitting to CrunchDAO.*

- **Issue #5**: Implement historical price downloader and warm-up loader for all 12 assets.
- **Issue #6**: Build local simulation runner wrapping `TrackerEvaluator` with timeline events (`build_events`).
- **Issue #7**: Implement automated CRPS scoring and leaderboard rank comparator vs historic benchmark.
- **Issue #8**: Add visualization utilities for return distribution vs realized price space.

---

## Milestone 3: Volatility & Order Flow Feature Engineering
*Goal: Extract high-frequency volatility estimators to accurately scale prediction dispersion.*

- **Issue #9**: Implement rolling Realized Volatility, Parkinson Volatility, and Garman-Klass estimators.
- **Issue #10**: Implement online recursive EWMA and GARCH(1,1) volatility filter.
- **Issue #11**: Extract intraday volume, spread, and microstructure indicators.
- **Issue #12**: Develop asset-specific Hurst exponent estimator for sub/super-diffusive step scaling.

---

## Milestone 4: Heavy-Tailed & Mixture Models
*Goal: Move beyond naive Gaussian assumptions to capture fat tails and multi-modal regimes.*

- **Issue #13**: Implement Student-t distribution tracker ($t_\nu$) with calibrated degrees of freedom.
- **Issue #14**: Implement 2-Component Gaussian Mixture Model (Quiet Regime + High Volatility Jump Regime).
- **Issue #15**: Integrate LightGBM / NGBoost quantile regression for empirical density forecasting.
- **Issue #16**: Backtest fat-tailed trackers against baseline on volatile crypto market periods.

---

## Milestone 5: Multi-Horizon Meta-Ensemble & Calibration
*Goal: Fuse specialized sub-trackers for 1h and 24h horizons with probability calibration.*

- **Issue #17**: Implement `MultiHorizonRouter` utilizing `SubTracker` architecture.
- **Issue #18**: Build Probability Integral Transform (PIT) calibration engine to optimize CRPS reliability.
- **Issue #19**: Develop dynamic regime-switching classifier (Trending vs Ranging vs Volatile).
- **Issue #20**: Optimize mixture weights using cross-entropy minimization and CRPS gradient descent.

---

## Milestone 6: Cloud Deployment, Automated Live Tracking & Mining Node Integration
*Goal: Productionize model on CrunchDAO Hub and connect to Subnet 50 rewards.*

- **Issue #21**: Register project on CrunchDAO Hub and configure `.crunchdao/project.json` and credentials.
- **Issue #22**: Build automated submission packaging and push script (`scripts/submit.py`).
- **Issue #23**: Implement checkpoint and leaderboard monitor to track rolling 7-day Anchor CRPS.
- **Issue #24**: Setup background retraining cron jobs with zero-downtime atomic model swapping.
