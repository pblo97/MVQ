# portfolio_manager/optimization/__init__.py
"""
Portfolio Optimization Module

Includes:
- Parameter Grid Search with Walk-Forward Cross-Validation
- Hyperparameter optimization for Kelly Criterion
- Sensitivity analysis and parameter recommendations
"""

from .parameter_search import (
    walk_forward_cross_validation,
    optimize_kelly_parameters,
    compare_parameter_sets,
    analyze_parameter_sensitivity,
    recommend_parameters,
    ParameterSearchResult
)

__all__ = [
    'walk_forward_cross_validation',
    'optimize_kelly_parameters',
    'compare_parameter_sets',
    'analyze_parameter_sensitivity',
    'recommend_parameters',
    'ParameterSearchResult'
]
