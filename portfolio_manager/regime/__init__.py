# portfolio_manager/regime/__init__.py
"""
Macro Regime Detection

Includes:
- HMM (Hidden Markov Model) - Hamilton (1989), Ang & Bekaert (2002)
"""

from .hmm_regime import (
    HiddenMarkovRegime,
    RegimeState,
    create_macro_features_from_fred
)

__all__ = [
    'HiddenMarkovRegime',
    'RegimeState',
    'create_macro_features_from_fred'
]
