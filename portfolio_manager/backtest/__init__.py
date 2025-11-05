# portfolio_manager/backtest/__init__.py
"""
Backtest Framework

Implements rigorous out-of-sample validation:
- Walk-forward (rolling window)
- Expanding window (cumulative training)

Based on:
- Bailey et al. (2014): "The Probability of Backtest Overfitting"
- Harvey et al. (2016): "... and the Cross-Section of Expected Returns"
- López de Prado (2018): "Advances in Financial Machine Learning"
"""

from .walk_forward import (
    walk_forward_backtest,
    expanding_window_backtest,
    calculate_backtest_metrics,
    plot_backtest_results,
    BacktestResult
)

__all__ = [
    'walk_forward_backtest',
    'expanding_window_backtest',
    'calculate_backtest_metrics',
    'plot_backtest_results',
    'BacktestResult'
]
