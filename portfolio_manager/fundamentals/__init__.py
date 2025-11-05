# portfolio_manager/fundamentals/__init__.py
"""
Fundamental Analysis Modules

- piotroski: Piotroski F-Score calculation and degradation detection
"""

from .piotroski import (
    calculate_piotroski_signals,
    calculate_piotroski_history,
    detect_fundamental_degradation,
    interpret_fscore
)

__all__ = [
    'calculate_piotroski_signals',
    'calculate_piotroski_history',
    'detect_fundamental_degradation',
    'interpret_fscore'
]
