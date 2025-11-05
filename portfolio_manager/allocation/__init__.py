# portfolio_manager/allocation/__init__.py
"""
Portfolio Allocation Methods

Includes:
- HRP (Hierarchical Risk Parity) - López de Prado (2016)
- Kelly Vectorial (Multivariate Kelly Criterion) - with robust covariance
- Kelly with Transaction Costs - Gârleanu & Pedersen (2013)
"""

from .hrp import (
    compute_hrp_weights,
    compute_hrp_with_constraints,
    compare_hrp_vs_equal_weight,
    get_hrp_clusters
)

from .kelly_vectorial import (
    kelly_vectorial_weights,
    kelly_vectorial_with_fallback,
    compare_covariance_methods,
    diagnose_covariance_quality,
    effective_number_of_assets,
    concentration_ratio
)

from .kelly_with_costs import (
    kelly_with_transaction_costs,
    compare_with_without_costs,
    optimal_rebalancing_frequency,
    turnover_aware_kelly
)

__all__ = [
    # HRP
    'compute_hrp_weights',
    'compute_hrp_with_constraints',
    'compare_hrp_vs_equal_weight',
    'get_hrp_clusters',
    # Kelly Vectorial
    'kelly_vectorial_weights',
    'kelly_vectorial_with_fallback',
    'compare_covariance_methods',
    'diagnose_covariance_quality',
    'effective_number_of_assets',
    'concentration_ratio',
    # Kelly with Costs
    'kelly_with_transaction_costs',
    'compare_with_without_costs',
    'optimal_rebalancing_frequency',
    'turnover_aware_kelly'
]
