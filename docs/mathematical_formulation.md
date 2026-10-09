# Mathematical Formulation & Modeling Strategy: Crunch-Synth

## 1. Probabilistic Forecasting Formulation

Let $P_t \in \mathbb{R}^+$ denote the price of an asset at timestamp $t$ (in seconds).
We observe an information filtration $\mathcal{F}_t = \sigma(\{P_s\}_{s \le t}, \{X_s\}_{s \le t})$ consisting of past prices and exogenous order-flow / cross-asset features.

For a fixed prediction horizon $H$ and step increment $\Delta t$, the target random variable is the incremental price change:

$$r_{t, k} = P_{t + k \Delta t} - P_{t + (k-1) \Delta t}, \quad k \in \{1, 2, \dots, H/\Delta t\}$$

The objective is to specify the conditional probability density function:

$$f_{t, k}(x) = \frac{d}{dx} \mathbb{P}\left( r_{t, k} \le x \mid \mathcal{F}_t \right)$$

---

## 2. Evaluation Metric: Continuous Ranked Probability Score (CRPS)

The Continuous Ranked Probability Score is a strictly proper scoring rule that penalizes both lack of calibration (systematic bias / misestimated dispersion) and lack of sharpness (excessive uncertainty).

Given a predictive cumulative distribution function $F(x) = \int_{-\infty}^x f(u) du$ and an observed ground-truth realization $y \in \mathbb{R}$:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(x) - \mathbf{1}\{x \ge y\} \right)^2 dx$$

### Closed-Form Solutions for Parametric Distributions

1. **Gaussian Distribution $\mathcal{N}(\mu, \sigma^2)$**:
   $$\text{CRPS}\left(\mathcal{N}(\mu, \sigma^2), y\right) = \sigma \left[ \frac{y - \mu}{\sigma} \left( 2\Phi\left(\frac{y-\mu}{\sigma}\right) - 1 \right) + 2\phi\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right]$$
   Where $\Phi(\cdot)$ is the standard normal CDF and $\phi(\cdot)$ is the standard normal PDF.
   *Note: When $\mu = y$, $\text{CRPS} = \sigma (2/\sqrt{2\pi} - 1/\sqrt{\pi}) \approx 0.2337 \sigma$. The score scales linearly with the forecast spread $\sigma$. Overestimating volatility inflates the penalty.*

2. **Student-t Distribution $t_\nu(\mu, \sigma)$**:
   Financial return time series exhibit heavy tails (leptokurtosis) with excess kurtosis $\kappa > 0$. Modeling with degrees of freedom $\nu \in [3, 7]$ prevents severe CRPS penalties during jump / flash-crash events.

3. **Mixture Distributions**:
   For a mixture $F(x) = \sum_{m=1}^M w_m F_m(x)$ where $\sum w_m = 1$, the mixture allows capturing bimodal distributions (e.g., trend continuation vs mean reversion regime).

---

## 3. Temporal Scaling & Fractal Properties

Under standard geometric Brownian motion (GBM), incremental returns follow:
$$\sigma(\Delta t) = \sigma_0 \sqrt{\frac{\Delta t}{\tau_0}}$$

However, empirical cryptocurrency returns deviate from standard Brownian motion:
$$\sigma(\Delta t) = \sigma_0 \left(\frac{\Delta t}{\tau_0}\right)^H$$

Where $H$ is the **Hurst exponent**:
- $H = 0.5$: Pure random walk (Gaussian diffusion).
- $H < 0.5$: Mean-reverting regime (sub-diffusive), common in short horizons (1m–5m microstructure).
- $H > 0.5$: Momentum / trending regime (super-diffusive), common during liquidation cascades.

Calibrating $H$ dynamically per asset drastically improves multi-step volatility projections across $k=1 \dots K$.

---

## 4. Volatility Modeling Engine

Accurate volatility $\hat{\sigma}_t$ estimation is the single most critical driver of CRPS performance.

### A. EWMA / RiskMetrics
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2, \quad \lambda \in [0.94, 0.98]$$
Fast recursive computation ($O(1)$) suited for real-time tick updates.

### B. GARCH(1,1)
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \alpha + \beta < 1$$
Captures volatility clustering: periods of high turbulence are followed by continued turbulence.

### C. Parkinson & Garman-Klass Realized Volatility
Using high, low, open, close quotes when available:
$$\sigma_{\text{Parkinson}}^2 = \frac{(\ln(H_t) - \ln(L_t))^2}{4 \ln 2}$$
Provides $5\times$ more statistical efficiency than close-to-close returns alone.

---

## 5. Drift Modeling Engine

While volatility $\sigma$ determines the distribution width, drift $\mu$ determines its center of mass:

$$\hat{\mu}_{t, k} = \text{DriftModel}(\mathcal{F}_t) \cdot \frac{\Delta t}{\tau_{\text{base}}}$$

Features for Drift:
- Cross-asset momentum (BTC leading altcoins like SOL, ETH).
- Short-term exponential moving average cross ($EMA_5 - EMA_{20}$).
- Volume-weighted average price (VWAP) deviation.
- Mean reversion z-score against 24h rolling price band.

When signal-to-noise ratio is low, setting $\hat{\mu} \approx 0$ (martingale hypothesis) minimizes CRPS variance, preventing over-confident biased directional bets.
