# portfolio_manager/regime/hmm_regime.py
"""
Hidden Markov Model (HMM) for Macro Regime Detection

Implements HMM-based regime classification to replace z-score approach.

States typically represent:
- State 0: CRISIS (high volatility, negative returns)
- State 1: BEAR (moderate negative, risk-off)
- State 2: NEUTRAL (low volatility, range-bound)
- State 3: BULL (positive returns, risk-on)

The HMM learns:
1. Transition probabilities: P(state_t | state_{t-1})
2. Emission probabilities: P(observation_t | state_t)

Academic References:
- Hamilton (1989): A New Approach to the Economic Analysis of Nonstationary Time Series
- Ang & Bekaert (2002): Regime Switches in Interest Rates
- Kim & Nelson (1999): State-Space Models with Regime Switching
- Guidolin & Timmermann (2008): International Asset Allocation under Regime Switching
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import warnings


@dataclass
class RegimeState:
    """Represents a market regime state."""
    state_id: int
    label: str  # "CRISIS", "BEAR", "NEUTRAL", "BULL"
    m_multiplier: float  # Sizing multiplier (0.6 - 1.3)
    beta_cap: float  # Beta cap for this regime
    vol_cap: float  # Position cap for this regime
    description: str


class HiddenMarkovRegime:
    """
    Hidden Markov Model for regime detection.

    Uses hmmlearn library (Gaussian HMM) for regime classification.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (2-4 typical)
                2 states: BEAR/BULL
                3 states: BEAR/NEUTRAL/BULL
                4 states: CRISIS/BEAR/NEUTRAL/BULL
            covariance_type: 'full', 'diag', 'tied', 'spherical'
            n_iter: Maximum EM iterations
            random_state: Random seed
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Import hmmlearn (lazy import)
        try:
            from hmmlearn import hmm
            self.hmm_lib = hmm
            self.model = None
            self.feature_names = None
            self.regime_mapping = None
        except ImportError:
            raise ImportError(
                "hmmlearn not installed. Install with: pip install hmmlearn"
            )

    def fit(
        self,
        features_df: pd.DataFrame,
        labels: Optional[List[str]] = None
    ) -> 'HiddenMarkovRegime':
        """
        Fit HMM to macro features.

        Args:
            features_df: DataFrame with macro indicators (rows=dates, cols=features)
                        Example columns: ['SPY_return', 'VIX', 'HY_OAS', 'Curve_2_10', ...]
            labels: Optional regime labels (if None, auto-generate)

        Returns:
            self (fitted model)
        """
        if features_df.empty or len(features_df) < 100:
            raise ValueError("Need at least 100 observations to fit HMM")

        # Store feature names
        self.feature_names = features_df.columns.tolist()

        # Prepare data (z-score normalization)
        X = features_df.values
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Initialize Gaussian HMM
        self.model = self.hmm_lib.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params='stmc',  # Initialize start, transition, means, covars
            params='stmc'  # Update all parameters
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_normalized)

        # Infer states for training data
        states = self.model.predict(X_normalized)

        # Classify states based on mean returns (if SPY_return available)
        self.regime_mapping = self._classify_states(states, features_df)

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime states for new data.

        Args:
            features_df: DataFrame with same features as training

        Returns:
            Array of regime state IDs (0, 1, 2, ...)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Align features
        if not all(col in features_df.columns for col in self.feature_names):
            raise ValueError(f"Missing features. Expected: {self.feature_names}")

        X = features_df[self.feature_names].values
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        return self.model.predict(X_normalized)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime state probabilities.

        Returns:
            Array of shape (n_samples, n_states) with probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features_df[self.feature_names].values
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        return self.model.predict_proba(X_normalized)

    def get_current_regime(self, features_df: pd.DataFrame) -> RegimeState:
        """
        Get current regime state.

        Args:
            features_df: Recent macro features (use last row for current)

        Returns:
            RegimeState object with label, multipliers, caps
        """
        states = self.predict(features_df)
        current_state_id = states[-1]

        return self.regime_mapping[current_state_id]

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get state transition probability matrix.

        Returns:
            DataFrame (n_states × n_states) with P(j | i)
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        labels = [self.regime_mapping[i].label for i in range(self.n_states)]

        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels
        )

    def _classify_states(
        self,
        states: np.ndarray,
        features_df: pd.DataFrame
    ) -> Dict[int, RegimeState]:
        """
        Classify HMM states into economic regimes based on characteristics.

        Uses SPY returns (if available) to rank states from worst to best.
        """
        # Calculate mean features per state
        state_characteristics = {}

        for state_id in range(self.n_states):
            mask = (states == state_id)
            if mask.sum() == 0:
                continue

            state_data = features_df[mask]

            # Use SPY returns if available
            if 'SPY_return' in state_data.columns or 'returns' in state_data.columns:
                ret_col = 'SPY_return' if 'SPY_return' in state_data.columns else 'returns'
                mean_return = state_data[ret_col].mean()
                volatility = state_data[ret_col].std()
            else:
                # Fallback: use first feature as proxy
                mean_return = state_data.iloc[:, 0].mean()
                volatility = state_data.iloc[:, 0].std()

            state_characteristics[state_id] = {
                'mean_return': mean_return,
                'volatility': volatility
            }

        # Sort states by mean return (worst to best)
        sorted_states = sorted(
            state_characteristics.keys(),
            key=lambda s: state_characteristics[s]['mean_return']
        )

        # Map to regime labels
        regime_mapping = {}

        if self.n_states == 2:
            # BEAR/BULL
            labels = ['BEAR', 'BULL']
            m_multipliers = [0.7, 1.2]
            beta_caps = [0.8, 1.3]
            vol_caps = [0.03, 0.05]

        elif self.n_states == 3:
            # BEAR/NEUTRAL/BULL
            labels = ['BEAR', 'NEUTRAL', 'BULL']
            m_multipliers = [0.7, 1.0, 1.2]
            beta_caps = [0.8, 1.0, 1.3]
            vol_caps = [0.03, 0.04, 0.05]

        elif self.n_states == 4:
            # CRISIS/BEAR/NEUTRAL/BULL
            labels = ['CRISIS', 'BEAR', 'NEUTRAL', 'BULL']
            m_multipliers = [0.6, 0.8, 1.0, 1.3]
            beta_caps = [0.6, 0.8, 1.0, 1.3]
            vol_caps = [0.02, 0.03, 0.04, 0.05]

        else:
            # Generic labels
            labels = [f"STATE_{i}" for i in range(self.n_states)]
            m_multipliers = np.linspace(0.7, 1.2, self.n_states).tolist()
            beta_caps = np.linspace(0.8, 1.3, self.n_states).tolist()
            vol_caps = np.linspace(0.03, 0.05, self.n_states).tolist()

        # Assign labels (sorted by return)
        for rank, state_id in enumerate(sorted_states):
            regime_mapping[state_id] = RegimeState(
                state_id=state_id,
                label=labels[rank],
                m_multiplier=m_multipliers[rank],
                beta_cap=beta_caps[rank],
                vol_cap=vol_caps[rank],
                description=f"{labels[rank]} regime (mean return: {state_characteristics[state_id]['mean_return']:.2%})"
            )

        return regime_mapping

    def analyze_regime_persistence(
        self,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze regime persistence (average duration per regime).

        Returns:
            DataFrame with columns: regime, avg_duration_days, frequency_pct
        """
        states = self.predict(features_df)

        # Calculate durations
        regime_durations = []
        current_regime = states[0]
        duration = 1

        for state in states[1:]:
            if state == current_regime:
                duration += 1
            else:
                regime_durations.append({
                    'regime': self.regime_mapping[current_regime].label,
                    'duration': duration
                })
                current_regime = state
                duration = 1

        # Final regime
        regime_durations.append({
            'regime': self.regime_mapping[current_regime].label,
            'duration': duration
        })

        df = pd.DataFrame(regime_durations)

        # Calculate statistics
        stats = df.groupby('regime')['duration'].agg(['mean', 'count']).reset_index()
        stats.columns = ['regime', 'avg_duration_days', 'n_episodes']
        stats['frequency_pct'] = 100 * stats['n_episodes'] / stats['n_episodes'].sum()

        return stats.sort_values('avg_duration_days', ascending=False)


def create_macro_features_from_fred(
    fred_df: pd.DataFrame,
    spy_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Create HMM features from FRED macro data.

    Args:
        fred_df: DataFrame with FRED indicators (from calculate_macro_zscore_auto_fred)
        spy_returns: Optional SPY returns series

    Returns:
        DataFrame with normalized features for HMM
    """
    features = fred_df.copy()

    # Add SPY returns if provided
    if spy_returns is not None:
        features['SPY_return'] = spy_returns.reindex(features.index)

    # Fill NaNs with forward fill + backfill
    features = features.fillna(method='ffill').fillna(method='bfill')

    # Drop rows with remaining NaNs
    features = features.dropna()

    return features
