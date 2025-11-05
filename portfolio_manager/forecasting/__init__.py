# portfolio_manager/forecasting/__init__.py
"""
Volatility Forecasting Module

Includes:
- GARCH(1,1) - Bollerslev (1986)
- EGARCH (Asymmetric) - Nelson (1991)
"""

from .garch import (
    GARCHVolatilityForecaster,
    forecast_portfolio_volatility,
    compare_sample_vs_garch
)

__all__ = [
    'GARCHVolatilityForecaster',
    'forecast_portfolio_volatility',
    'compare_sample_vs_garch'
]
