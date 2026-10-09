"""
Feature engineering and market dynamics package for CrunchDAO Synth.

Modules:
- volatility: Realized, Parkinson, Garman-Klass, Rogers-Satchell, EWMA, and GARCH(1,1).
- microstructure: Corwin-Schultz spread, Roll spread, Amihud illiquidity, liquidity shocks.
- hurst: Hurst exponent estimation (variance-time, R/S) and fractal diffusion step scaling.
"""

from .volatility import (
    ticks_to_ohlc,
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    OnlineEWMAVolatility,
    OnlineGARCH11,
)

from .microstructure import (
    corwin_schultz_spread,
    roll_effective_spread,
    amihud_illiquidity,
    LiquidityRegimeDetector,
)

from .hurst import (
    estimate_hurst_variance_time,
    estimate_hurst_rs,
    AssetHurstCalibrator,
)

from .regime_classifier import (
    MarketRegime,
    MarketRegimeClassifier,
)

__all__ = [
    "ticks_to_ohlc",
    "realized_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "OnlineEWMAVolatility",
    "OnlineGARCH11",
    "corwin_schultz_spread",
    "roll_effective_spread",
    "amihud_illiquidity",
    "LiquidityRegimeDetector",
    "estimate_hurst_variance_time",
    "estimate_hurst_rs",
    "AssetHurstCalibrator",
    "MarketRegime",
    "MarketRegimeClassifier",
]
