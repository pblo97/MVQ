# portfolio_manager/fundamentals/__init__.py
"""
Fundamental Analysis Modules

- piotroski: Piotroski F-Score calculation and degradation detection (for VALUE stocks)
- mohanram: Mohanram G-Score calculation and degradation detection (for GROWTH stocks)
"""

from .piotroski import (
    calculate_piotroski_signals,
    calculate_piotroski_history,
    detect_fundamental_degradation,
    interpret_fscore
)

from .mohanram import (
    calculate_mohanram_signals,
    calculate_mohanram_history,
    detect_growth_degradation,
    interpret_gscore,
    classify_value_vs_growth
)

__all__ = [
    # Piotroski (Value)
    'calculate_piotroski_signals',
    'calculate_piotroski_history',
    'detect_fundamental_degradation',
    'interpret_fscore',
    # Mohanram (Growth)
    'calculate_mohanram_signals',
    'calculate_mohanram_history',
    'detect_growth_degradation',
    'interpret_gscore',
    'classify_value_vs_growth'
]
