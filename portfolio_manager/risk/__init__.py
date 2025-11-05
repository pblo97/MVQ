# portfolio_manager/risk/__init__.py
"""
Risk Analytics Module

CVaR, VaR, stress testing, and portfolio risk attribution.
"""

from .cvar_analysis import (
    calculate_var,
    calculate_cvar,
    calculate_marginal_cvar,
    calculate_component_cvar,
    calculate_percentage_cvar_contribution,
    backtest_var,
    stress_test_scenarios,
    get_default_stress_scenarios,
    calculate_risk_metrics_summary
)

__all__ = [
    'calculate_var',
    'calculate_cvar',
    'calculate_marginal_cvar',
    'calculate_component_cvar',
    'calculate_percentage_cvar_contribution',
    'backtest_var',
    'stress_test_scenarios',
    'get_default_stress_scenarios',
    'calculate_risk_metrics_summary'
]
