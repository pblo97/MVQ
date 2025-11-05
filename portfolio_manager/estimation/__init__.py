# portfolio_manager/estimation/__init__.py
"""
Robust Covariance Estimation

Includes:
- Ledoit-Wolf Shrinkage (2003, 2004)
- Oracle Approximating Shrinkage (OAS)
- Exponentially Weighted Covariance (RiskMetrics)
"""

from .robust_cov import (
    ledoit_wolf_shrinkage,
    exponentially_weighted_cov,
    oracle_approximating_shrinkage,
    compare_covariance_estimators,
    recommend_covariance_estimator
)

__all__ = [
    'ledoit_wolf_shrinkage',
    'exponentially_weighted_cov',
    'oracle_approximating_shrinkage',
    'compare_covariance_estimators',
    'recommend_covariance_estimator'
]
