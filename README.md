# Crunch-Synth: Probabilistic Crypto & Synthetic Asset Forecasting

Autonomous quantitative system for the **CrunchDAO Synth** competition (Bittensor Subnet 50). Generates calibrated real-time continuous probability density forecasts (PDFs) for cryptocurrency and synthetic asset price changes across multiple time horizons.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CrunchDAO Synth](https://img.shields.io/badge/CrunchDAO-Synth%20(Subnet%2050)-green.svg)](https://hub.crunchdao.com/competitions/synth)

---

## 1. Overview & Mechanics

In **Synth**, quantitative models do not output single point forecasts or ordinal asset ranks. Instead, models submit **continuous probability density functions (PDFs)** for the incremental return:

$$r_{t, k} = P_{t + k \cdot \text{step}} - P_{t + (k-1) \cdot \text{step}}$$

Scoring is performed using the **Continuous Ranked Probability Score (CRPS)** evaluated against realized future market ticks after a quarantine period:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(x) - \mathbf{1}\{x \ge y\} \right)^2 dx$$

### Covered Assets & Horizons
- **Assets (12)**: `BTC`, `ETH`, `SOL`, `XRP`, `HYPE`, `XAUT`, `SP500`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`, `WTIOIL`.
- **1-Hour Horizon**: Steps `{1m, 5m, 15m, 30m, 1h}` triggered every 12m (Crypto assets).
- **24-Hour Horizon**: Steps `{5m, 1h, 6h, 24h}` triggered every 1h (All assets).
- **Latency Requirement**: Strict execution limit of **< 40 seconds** per prediction round.

---

## 2. Project Architecture

```
crunch-synth/
├── docs/                      # Research, mathematical formulation & competition guides
│   ├── competition_overview.md
│   ├── mathematical_formulation.md
│   └── roadmap.md
├── src/                       # Core quantitative engine
│   ├── trackers/              # Probabilistic density forecasters (subclassing TrackerBase)
│   ├── features/              # Volatility, momentum, and microstructure features
│   └── evaluation/            # Local CRPS backtesting and benchmarking harness
├── scripts/                   # CLI utilities for backtesting and submissions
├── tests/                     # Unit and integration tests
├── resources/                 # Pretrained weights and checkpoint models
├── main.py                    # Production entrypoint for CrunchDAO runner
├── requirements.txt           # Minimal production dependencies
└── README.md
```

---

## 3. Quickstart

### Installation
```bash
# Clone the repository
git clone git@github.com:sanjmen/crunch-synth.git
cd crunch-synth

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Local Simulation & CRPS Evaluation
```bash
python scripts/evaluate_baseline.py --assets BTC SOL ETH --days 3
```

---

## 4. Development Roadmap

- [x] **Milestone 1**: Foundations, Local Price Engine & Gaussian Step Baseline
- [ ] **Milestone 2**: Local Validation Harness & Fast CRPS Scoring
- [ ] **Milestone 3**: Volatility & Order Flow Feature Engineering (GARCH, Realized Vol, Microstructure)
- [ ] **Milestone 4**: Heavy-Tailed & Mixture Models (Student-t, Gaussian Mixture Models, Quantile Regression)
- [ ] **Milestone 5**: Multi-Horizon Meta-Ensemble & Calibration
- [ ] **Milestone 6**: Cloud Deployment, Automated Live Tracking & Mining Node Integration

For detailed issue breakdowns, see [docs/roadmap.md](docs/roadmap.md).
