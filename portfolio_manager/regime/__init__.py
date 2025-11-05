# portfolio_manager/regime/__init__.py
"""
Macro Regime Detection

Includes:
- HMM (Hidden Markov Model) - Hamilton (1989), Ang & Bekaert (2002)
- Random Forest Regime Classification - Ballings et al. (2015), Breiman (2001)
"""

from .hmm_regime import (
    HiddenMarkovRegime,
    RegimeState,
    create_macro_features_from_fred
)

from .random_forest_regime import (
    RandomForestRegime,
    compare_regime_models
)

__all__ = [
    'HiddenMarkovRegime',
    'RandomForestRegime',
    'RegimeState',
    'create_macro_features_from_fred',
    'compare_regime_models'
]
