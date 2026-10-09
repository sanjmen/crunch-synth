# CrunchDAO Synth Competition: Full Overview & Architecture

## 1. Executive Summary

**CrunchDAO Synth** is a 24/7 real-time probabilistic forecasting challenge operating in connection with **Bittensor Subnet 50** and hosted on the [CrunchDAO Platform](https://hub.crunchdao.com/competitions/synth).

Unlike traditional quantitative modeling competitions (such as `datacrunch-2`) which request point estimates or ordinal cross-sectional asset rankings, **Synth** requires competitors to predict **continuous probability density functions (PDFs)** for future asset price movements across multiple temporal horizons and granularities.

Competitors model high-frequency return distributions for major cryptocurrencies, tokenized commodities, and equities, balancing drift and volatility regimes under strict latency constraints.

---

## 2. Economics, Prizes & Subnet 50 Integration

- **Prize Architecture**:
  - **Fixed Pot**: $30,000 USDC pool.
  - **Dynamic Mining Emissions**: Real-time mining rewards emitted by Subnet 50 on the Bittensor network (historically up to ~$50,000 USD/month).
- **Payout Frequency**: Distributed every **7 + 1 days** at target resolution checkpointing.
- **Horizon Allocation**:
  - 50% of the weekly reward pot is allocated to the **1-Hour Horizon**.
  - 50% of the weekly reward pot is allocated to the **24-Hour Horizon**.
- **Participant Caps**:
  - Each participant can deploy up to **2 active models**.
  - Only the best performing model per participant is eligible for payouts.
  - Top 4 participants per horizon (maximum 8 players total) share the pot, weighted exponentially by leaderboard rank.
  - A benchmark baseline model is excluded from payouts and functions as a qualification cutoff: models must outperform the benchmark to receive emissions.

---

## 3. Asset Coverage

The competition currently covers 12 assets across crypto, tokenized equities, and commodities:

| Asset Class | Symbol | Asset Description | 1h Horizon | 24h Horizon |
| :--- | :--- | :--- | :---: | :---: |
| **Crypto** | `BTC` | Bitcoin | Yes | Yes |
| **Crypto** | `ETH` | Ethereum | Yes | Yes |
| **Crypto** | `SOL` | Solana | Yes | Yes |
| **Crypto** | `XRP` | Ripple | Yes | Yes |
| **Crypto** | `HYPE` | Hyperliquid | Yes | Yes |
| **Tokenized Metals** | `XAUT` | Tether Gold | No | Yes |
| **Tokenized Equities** | `SP500` | S&P 500 Index | No | Yes |
| **Tokenized Equities** | `NVDAX` | NVIDIA Tokenized Stock | No | Yes |
| **Tokenized Equities** | `TSLAX` | Tesla Tokenized Stock | No | Yes |
| **Tokenized Equities** | `AAPLX` | Apple Tokenized Stock | No | Yes |
| **Tokenized Equities** | `GOOGLX` | Alphabet Tokenized Stock | No | Yes |
| **Commodities** | `WTIOIL` | WTI Crude Oil | No | Yes |

---

## 4. Horizons, Resolutions & Cadence

A prediction round is defined by an asset, a forecast horizon, and one or more step resolutions:

### A. 24-Hour Horizon Profile
- **Total Horizon**: 86,400 seconds (24 hours).
- **Execution Interval**: Triggered approximately every 39–60 minutes per asset.
- **Step Resolutions**:
  - `300s` (5 minutes) &rarr; $86400 / 300 = 288$ step distributions.
  - `3,600s` (1 hour) &rarr; $86400 / 3600 = 24$ step distributions.
  - `21,600s` (6 hours) &rarr; $86400 / 21600 = 4$ step distributions.
  - `86,400s` (24 hours) &rarr; $86400 / 86400 = 1$ step distribution.
- **Covered Assets**: All 12 supported assets.

### B. 1-Hour Horizon Profile
- **Total Horizon**: 3,600 seconds (1 hour).
- **Execution Interval**: Triggered approximately every 8–12 minutes per asset.
- **Step Resolutions**:
  - `60s` (1 minute) &rarr; $3600 / 60 = 60$ step distributions.
  - `300s` (5 minutes) &rarr; $3600 / 300 = 12$ step distributions.
  - `900s` (15 minutes) &rarr; $3600 / 900 = 4$ step distributions.
  - `1,800s` (30 minutes) &rarr; $3600 / 1800 = 2$ step distributions.
  - `3,600s` (1 hour) &rarr; $3600 / 3600 = 1$ step distribution.
- **Covered Assets**: `BTC`, `SOL`, `ETH`, `XRP`, `HYPE`.

---

## 5. Target Formulation: Incremental Returns

**Crucial Distinction**: The model does **NOT** predict the raw spot price $P_{t+k}$ nor a point return.

The target is the **distribution of incremental price changes**:
$$r_{t, k} = P_{t + k \cdot \text{step}} - P_{t + (k-1) \cdot \text{step}}$$

This formulation guarantees a stationary target series across assets with widely varying price scales, allowing models to focus directly on drift and volatility dynamics.

### Density Specification (`density_pdf`)

Each step's prediction must comply with the `density_pdf` dictionary format. The engine supports parametric mixture densities:

```python
{
    "step": k * step,  # Time offset in seconds from prediction origin
    "type": "mixture",
    "components": [
        {
            "density": {
                "type": "builtin",  # Optimized C/Cython execution
                "name": "norm",     # e.g., 'norm' (Gaussian) or 't' (Student-t)
                "params": {
                    "loc": float(predicted_drift),
                    "scale": float(predicted_volatility)
                }
            },
            "weight": 1.0
        }
    ]
}
```

---

## 6. Evaluation Metric: CRPS & Quarantine Mechanism

### Continuous Ranked Probability Score (CRPS)
Predictions are scored using the **CRPS**, which generalizes the Mean Absolute Error to probabilistic forecasts:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(x) - \mathbf{1}\{x \ge y\} \right)^2 dx$$

Where:
- $F(x)$ is the cumulative distribution function (CDF) of the submitted predictive density.
- $y$ is the actual realized incremental return observed at that step.
- $\mathbf{1}\{x \ge y\}$ is the Heaviside step function.

A **lower CRPS** indicates superior predictive performance. If a prediction is sharp (low variance) and accurate, the score approaches 0; if it is poorly calibrated, the penalty grows quadratically.

### Relative Ranking & Normalization
For every scoring event across the platform:
- The top-performing model receives a normalized score of **1.0**.
- The bottom 5% receive a normalized score of **0.0**.
- Intermediate participants receive linearly interpolated scores between 0 and 1.
- **Leaderboard Anchor Score**: Evaluated as a 7-day rolling average of relative scores.

### Quarantine Mechanism
Predictions cannot be scored immediately because future real-world price observations must occur first. Every prediction enters a **quarantine buffer** until $t + \text{horizon}$ seconds have elapsed and the ground-truth ticks are recorded.

---

## 7. Operational Constraints & Latency

- **Inference Time Limit**: **Strictly < 40 seconds per round**. In practice, high-frequency prediction rounds should execute in **< 100ms** to guarantee zero timeouts.
- **Tick Frequency**: The framework pushes prices every 60 seconds via `tick(data)`.
- **Model Warm-Start**: Submissions should preload pre-trained weights from `resources/` on startup so initial predictions are immediate while background training tasks execute asynchronously.
