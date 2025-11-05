# portfolio_manager/allocation/__init__.py
"""
Portfolio Allocation Methods

Includes:
- HRP (Hierarchical Risk Parity) - López de Prado (2016)
"""

from .hrp import (
    compute_hrp_weights,
    compute_hrp_with_constraints,
    compare_hrp_vs_equal_weight,
    get_hrp_clusters
)

__all__ = [
    'compute_hrp_weights',
    'compute_hrp_with_constraints',
    'compare_hrp_vs_equal_weight',
    'get_hrp_clusters'
]
