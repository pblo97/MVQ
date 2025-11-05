# portfolio_manager/regime/random_forest_regime.py
"""
Random Forest Regime Classification

Supervised ML approach for market regime detection using Random Forest.
Complements HMM (unsupervised) with labeled historical regime learning.

Academic References:
- Ballings et al. (2015): Evaluating Multiple Classifiers for Stock Price Direction Prediction
- Nti et al. (2020): A systematic review of fundamental and technical analysis
- Breiman (2001): Random Forests

Features:
- Macro indicators: GDP, inflation, unemployment, Fed rates
- Technical indicators: VIX, market breadth, momentum
- Sentiment indicators: Put/Call ratio, AAII sentiment

Author: Portfolio Optimization Team
Date: 2025-11
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Import RegimeState from hmm_regime (avoid duplication)
from .hmm_regime import RegimeState


class RandomForestRegime:
    """
    Random Forest classifier for market regime detection

    Uses supervised learning to classify market regimes based on:
    - Macro features: GDP growth, inflation, unemployment, Fed rates
    - Technical features: VIX, market momentum, breadth
    - Sentiment features (optional): Put/Call ratio, AAII sentiment

    Academic Foundation:
    - Breiman (2001): Random Forests - ensemble of decision trees
    - Ballings et al. (2015): Multiple classifiers for stock prediction
    - Feature engineering based on practitioner research (Nti et al. 2020)

    Parameters:
    -----------
    n_estimators : int
        Number of trees in forest (default 100)
    max_depth : int
        Maximum tree depth (default 10)
    min_samples_split : int
        Minimum samples to split node (default 20)
    random_state : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 20,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )

        # Scaler for features
        self.scaler = StandardScaler()

        # Feature names
        self.feature_names: List[str] = []

        # Regime mapping
        self.regime_mapping: Dict[str, RegimeState] = {}

        # Training metadata
        self.is_trained: bool = False
        self.cv_score: Optional[float] = None
        self.feature_importance: Optional[pd.Series] = None

    def prepare_features(
        self,
        macro_df: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare features for Random Forest from macro indicators and returns

        Features Generated:
        - Macro z-scores (from macro_df columns ending in '_z')
        - Rolling returns (1M, 3M, 6M, 12M momentum)
        - Volatility (rolling std)
        - Drawdown (from peak)
        - Trend (SMA crossovers)

        Parameters:
        -----------
        macro_df : pd.DataFrame
            DataFrame with macro z-scores (columns ending in '_z')
        returns_df : pd.DataFrame, optional
            Market returns for technical features

        Returns:
        --------
        features_df : pd.DataFrame
            Feature matrix ready for training/prediction
        """
        features = {}

        # 1. Macro features (z-scores)
        macro_cols = [col for col in macro_df.columns if col.endswith('_z')]
        for col in macro_cols:
            features[col] = macro_df[col]

        # 2. Composite z-score
        if 'composite_z' in macro_df.columns:
            features['composite_z'] = macro_df['composite_z']

        # 3. Technical features from returns (if provided)
        if returns_df is not None and not returns_df.empty:
            # Average returns across assets
            avg_returns = returns_df.mean(axis=1)

            # Momentum features (trailing returns)
            features['momentum_1m'] = avg_returns.rolling(21).sum()
            features['momentum_3m'] = avg_returns.rolling(63).sum()
            features['momentum_6m'] = avg_returns.rolling(126).sum()
            features['momentum_12m'] = avg_returns.rolling(252).sum()

            # Volatility features
            features['volatility_1m'] = avg_returns.rolling(21).std()
            features['volatility_3m'] = avg_returns.rolling(63).std()

            # Drawdown from peak
            cum_returns = (1 + avg_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            features['drawdown'] = drawdown

            # Trend (SMA crossover)
            sma_20 = avg_returns.rolling(20).mean()
            sma_50 = avg_returns.rolling(50).mean()
            features['sma_crossover'] = (sma_20 - sma_50) / sma_50.abs()

        features_df = pd.DataFrame(features, index=macro_df.index)
        features_df = features_df.dropna()

        return features_df

    def create_labeled_regimes(
        self,
        dates: pd.DatetimeIndex,
        crisis_periods: Optional[List[Tuple[str, str]]] = None,
        bear_periods: Optional[List[Tuple[str, str]]] = None,
        bull_periods: Optional[List[Tuple[str, str]]] = None
    ) -> pd.Series:
        """
        Create labeled regime series from known historical periods

        Default periods (if not provided):
        - CRISIS: 2008-09 (GFC), 2020 Q1 (COVID)
        - BEAR: 2001-2002 (Dot-com), 2022 (Rate hikes)
        - BULL: 2009-2019, 2020-2021
        - NEUTRAL: Everything else

        Parameters:
        -----------
        dates : pd.DatetimeIndex
            Dates to label
        crisis_periods : list of tuples (start_date, end_date)
            Crisis regime periods
        bear_periods : list of tuples
            Bear market periods
        bull_periods : list of tuples
            Bull market periods

        Returns:
        --------
        labels : pd.Series
            Regime labels ('CRISIS', 'BEAR', 'NEUTRAL', 'BULL')
        """
        # Default historical periods
        if crisis_periods is None:
            crisis_periods = [
                ('2008-09-01', '2009-03-31'),  # GFC
                ('2020-02-15', '2020-04-15')   # COVID crash
            ]

        if bear_periods is None:
            bear_periods = [
                ('2000-09-01', '2002-10-31'),  # Dot-com bust
                ('2022-01-01', '2022-10-31'),  # 2022 bear market
                ('2015-08-01', '2016-02-29')   # Oil crash
            ]

        if bull_periods is None:
            bull_periods = [
                ('2009-04-01', '2020-02-14'),  # Post-GFC bull
                ('2020-04-16', '2021-12-31'),  # Post-COVID bull
                ('2023-01-01', '2024-12-31')   # 2023-2024 rally
            ]

        # Initialize all as NEUTRAL
        labels = pd.Series('NEUTRAL', index=dates)

        # Label periods (order matters: CRISIS overrides BEAR overrides BULL)
        for start, end in bull_periods:
            mask = (dates >= start) & (dates <= end)
            labels[mask] = 'BULL'

        for start, end in bear_periods:
            mask = (dates >= start) & (dates <= end)
            labels[mask] = 'BEAR'

        for start, end in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            labels[mask] = 'CRISIS'

        return labels

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        cv_folds: int = 5
    ) -> 'RandomForestRegime':
        """
        Train Random Forest on labeled regime data

        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix (rows = dates, columns = features)
        labels : pd.Series
            Regime labels ('CRISIS', 'BEAR', 'NEUTRAL', 'BULL')
        cv_folds : int
            Number of cross-validation folds for evaluation

        Returns:
        --------
        self : RandomForestRegime
            Trained model
        """
        # Align features and labels
        common_idx = features_df.index.intersection(labels.index)
        X = features_df.loc[common_idx]
        y = labels.loc[common_idx]

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need ≥100)")

        # Store feature names
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)

        # Cross-validation score
        self.cv_score = cross_val_score(
            self.model, X_scaled, y, cv=cv_folds, scoring='accuracy'
        ).mean()

        # Feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        # Create regime mapping (compatible with existing system)
        self.regime_mapping = {
            'CRISIS': RegimeState(
                state_id=0,
                label='CRISIS',
                m_multiplier=0.6,
                beta_cap=0.7,
                vol_cap=0.20,
                description='Crisis regime: extreme risk-off, maximum defense',
                probability=1.0
            ),
            'BEAR': RegimeState(
                state_id=1,
                label='BEAR',
                m_multiplier=0.8,
                beta_cap=0.9,
                vol_cap=0.18,
                description='Bear regime: risk-off, defensive positioning',
                probability=1.0
            ),
            'NEUTRAL': RegimeState(
                state_id=2,
                label='NEUTRAL',
                m_multiplier=1.0,
                beta_cap=1.0,
                vol_cap=0.15,
                description='Neutral regime: balanced risk exposure',
                probability=1.0
            ),
            'BULL': RegimeState(
                state_id=3,
                label='BULL',
                m_multiplier=1.2,
                beta_cap=1.3,
                vol_cap=0.12,
                description='Bull regime: risk-on, aggressive positioning',
                probability=1.0
            )
        }

        self.is_trained = True

        return self

    def predict_regime(
        self,
        features_df: pd.DataFrame
    ) -> RegimeState:
        """
        Predict current regime from features

        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix (last row = current state)

        Returns:
        --------
        regime_state : RegimeState
            Predicted regime with probability
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Use last row (most recent)
        X = features_df[self.feature_names].iloc[-1:].values
        X_scaled = self.scaler.transform(X)

        # Predict
        predicted_label = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Get probability for predicted class
        class_idx = list(self.model.classes_).index(predicted_label)
        confidence = probabilities[class_idx]

        # Return RegimeState with probability
        regime = self.regime_mapping[predicted_label]
        regime.probability = confidence

        return regime

    def get_regime_probabilities(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get probabilities for all regimes

        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix (last row = current state)

        Returns:
        --------
        probs : dict
            Regime labels → probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = features_df[self.feature_names].iloc[-1:].values
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[0]

        return dict(zip(self.model.classes_, probabilities))

    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Get top N most important features

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        importance : pd.Series
            Feature importances (sorted descending)
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.feature_importance.head(top_n)

    def evaluate(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data

        Returns accuracy, precision, recall, F1 per class

        Parameters:
        -----------
        features_df : pd.DataFrame
            Test features
        labels : pd.Series
            True labels

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, classification_report

        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Align
        common_idx = features_df.index.intersection(labels.index)
        X = features_df.loc[common_idx][self.feature_names]
        y = labels.loc[common_idx]

        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.model.predict(X_scaled)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'n_samples': len(y)
        }


def compare_regime_models(
    features_df: pd.DataFrame,
    labels: pd.Series,
    models: List[str] = ['rf', 'logistic', 'gradient_boosting']
) -> pd.DataFrame:
    """
    Compare multiple ML models for regime classification

    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    labels : pd.Series
        Regime labels
    models : list
        Models to compare: 'rf', 'logistic', 'gradient_boosting', 'svm'

    Returns:
    --------
    comparison : pd.DataFrame
        Comparison of models (accuracy, precision, recall, F1)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    results = []

    # Align
    common_idx = features_df.index.intersection(labels.index)
    X = features_df.loc[common_idx]
    y = labels.loc[common_idx]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model configs
    model_configs = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42)
    }

    for model_name in models:
        if model_name not in model_configs:
            continue

        model = model_configs[model_name]

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        results.append({
            'model': model_name,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'min_accuracy': cv_scores.min(),
            'max_accuracy': cv_scores.max()
        })

    return pd.DataFrame(results).sort_values('mean_accuracy', ascending=False)
