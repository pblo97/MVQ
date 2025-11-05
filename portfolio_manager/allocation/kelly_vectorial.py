# portfolio_manager/allocation/kelly_vectorial.py
"""
Kelly Criterion - Multivariate Vectorial Implementation

Implements classical Kelly portfolio optimization:
    w* = (1/κ) × Σ^-1 × μ

Where:
- μ: expected returns vector
- Σ: covariance matrix (can use robust estimation)
- κ: risk aversion parameter (fractional Kelly)

Supports:
- Sample covariance (standard)
- Ledoit-Wolf shrinkage (robust)
- Oracle Approximating Shrinkage (OAS)
- Exponentially weighted covariance (RiskMetrics)

Academic References:
- Kelly (1956): A New Interpretation of Information Rate
- MacLean et al. (2011): The Kelly Capital Growth Investment Criterion
- Ledoit & Wolf (2004): Honey, I Shrunk the Sample Covariance Matrix
- Thorp (2006): The Kelly Criterion in Blackjack Sports Betting and the Stock Market
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal


def kelly_vectorial_weights(
    returns_df: pd.DataFrame,
    base_kelly: float = 0.25,
    covariance_method: Literal['sample', 'ledoit_wolf', 'oas', 'ewm'] = 'sample',
    ewm_span: int = 60,
    winsorize_p: float = 0.01,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    long_only: bool = True,
    regularization: float = 1e-8
) -> pd.Series:
    """
    Calculate Kelly optimal weights using multivariate approach.

    Args:
        returns_df: DataFrame with asset returns (rows=dates, cols=assets)
        base_kelly: Fractional Kelly parameter (0 < κ ≤ 1)
        covariance_method: Method for covariance estimation
            - 'sample': Sample covariance
            - 'ledoit_wolf': Ledoit-Wolf shrinkage (robust)
            - 'oas': Oracle Approximating Shrinkage
            - 'ewm': Exponentially weighted (RiskMetrics)
        ewm_span: Span for EWM (if covariance_method='ewm')
        winsorize_p: Winsorization percentile (0.01 = clip at 1st/99th percentile)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        long_only: If True, force all weights ≥ 0
        regularization: Regularization term added to diagonal (numerical stability)

    Returns:
        pd.Series with optimal weights (index = asset symbols)

    Formula:
        w* = (1/κ) × Σ^-1 × μ

    Where:
        - κ = 1/base_kelly (risk aversion)
        - Σ^-1 = inverse covariance matrix
        - μ = expected returns vector

    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': [0.01, -0.02, 0.03, 0.01],
        ...     'GOOGL': [0.02, -0.01, 0.02, 0.01],
        ...     'MSFT': [0.01, -0.01, 0.02, 0.02]
        ... })
        >>> weights = kelly_vectorial_weights(returns, base_kelly=0.25, covariance_method='ledoit_wolf')
        >>> print(weights)
    """
    # Validate inputs
    if returns_df.empty or len(returns_df) < 10:
        # Fallback to equal weight
        n_assets = len(returns_df.columns)
        return pd.Series(1.0 / n_assets, index=returns_df.columns)

    if not (0 < base_kelly <= 1):
        raise ValueError(f"base_kelly must be in (0, 1], got {base_kelly}")

    # Winsorize returns (clip extreme outliers)
    if winsorize_p > 0:
        lower = returns_df.quantile(winsorize_p)
        upper = returns_df.quantile(1 - winsorize_p)
        returns_winsorized = returns_df.clip(lower=lower, upper=upper, axis=1)
    else:
        returns_winsorized = returns_df.copy()

    # Calculate expected returns (μ)
    mu = returns_winsorized.mean().values

    # Calculate covariance matrix (Σ)
    if covariance_method == 'sample':
        # Standard sample covariance
        cov = returns_winsorized.cov().values

    elif covariance_method == 'ledoit_wolf':
        # Ledoit-Wolf shrinkage (robust)
        from portfolio_manager.estimation.robust_cov import ledoit_wolf_shrinkage
        cov_df, shrinkage_intensity = ledoit_wolf_shrinkage(
            returns_winsorized,
            shrinkage_target='constant_correlation'
        )
        cov = cov_df.values

    elif covariance_method == 'oas':
        # Oracle Approximating Shrinkage
        from portfolio_manager.estimation.robust_cov import oracle_approximating_shrinkage
        cov_df, shrinkage_intensity = oracle_approximating_shrinkage(returns_winsorized)
        cov = cov_df.values

    elif covariance_method == 'ewm':
        # Exponentially weighted covariance (RiskMetrics)
        from portfolio_manager.estimation.robust_cov import exponentially_weighted_cov
        cov_df = exponentially_weighted_cov(returns_winsorized, span=ewm_span)
        cov = cov_df.values

    else:
        raise ValueError(f"Unknown covariance_method: {covariance_method}")

    # Add regularization for numerical stability
    cov_reg = cov + np.eye(len(cov)) * regularization

    # Compute Kelly weights: w = (1/κ) × Σ^-1 × μ
    try:
        cov_inv = np.linalg.inv(cov_reg)
        weights_raw = base_kelly * (cov_inv @ mu)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        cov_pinv = np.linalg.pinv(cov_reg)
        weights_raw = base_kelly * (cov_pinv @ mu)

    # Apply constraints
    if long_only:
        weights_raw = np.clip(weights_raw, 0, None)

    weights_raw = np.clip(weights_raw, min_weight, max_weight)

    # Normalize to sum to 1
    if weights_raw.sum() > 0:
        weights_normalized = weights_raw / weights_raw.sum()
    else:
        # If all weights are 0, fallback to equal weight
        weights_normalized = np.ones(len(weights_raw)) / len(weights_raw)

    return pd.Series(weights_normalized, index=returns_df.columns)


def compare_covariance_methods(
    returns_df: pd.DataFrame,
    base_kelly: float = 0.25,
    methods: list[str] = ['sample', 'ledoit_wolf', 'oas', 'ewm']
) -> pd.DataFrame:
    """
    Compare Kelly weights using different covariance estimation methods.

    Args:
        returns_df: DataFrame with asset returns
        base_kelly: Fractional Kelly parameter
        methods: List of covariance methods to compare

    Returns:
        DataFrame with weights for each method (columns = methods, rows = assets)

    Example:
        >>> returns = load_returns()
        >>> comparison = compare_covariance_methods(returns, base_kelly=0.25)
        >>> print(comparison)
    """
    results = {}

    for method in methods:
        try:
            weights = kelly_vectorial_weights(
                returns_df,
                base_kelly=base_kelly,
                covariance_method=method
            )
            results[method] = weights
        except Exception as e:
            print(f"Warning: Failed to compute weights with {method}: {e}")
            results[method] = pd.Series(0.0, index=returns_df.columns)

    comparison_df = pd.DataFrame(results)
    comparison_df.index.name = 'symbol'

    return comparison_df


def diagnose_covariance_quality(
    returns_df: pd.DataFrame,
    method: str = 'sample'
) -> dict:
    """
    Diagnose quality of covariance matrix estimation.

    Returns:
        dict with diagnostics:
            - condition_number: Σ condition number (>1000 = ill-conditioned)
            - min_eigenvalue: Smallest eigenvalue (negative = not positive-definite)
            - max_eigenvalue: Largest eigenvalue
            - rank: Matrix rank
            - is_positive_definite: Boolean
            - shrinkage_intensity: Shrinkage δ (if applicable)

    Academic Note:
        - Condition number = λ_max / λ_min
        - High condition number → unstable inverse → unreliable weights
        - Ledoit-Wolf shrinkage reduces condition number
    """
    from portfolio_manager.estimation.robust_cov import (
        ledoit_wolf_shrinkage,
        oracle_approximating_shrinkage,
        exponentially_weighted_cov
    )

    # Compute covariance
    if method == 'sample':
        cov = returns_df.cov().values
        shrinkage = None
    elif method == 'ledoit_wolf':
        cov_df, shrinkage = ledoit_wolf_shrinkage(returns_df)
        cov = cov_df.values
    elif method == 'oas':
        cov_df, shrinkage = oracle_approximating_shrinkage(returns_df)
        cov = cov_df.values
    elif method == 'ewm':
        cov_df = exponentially_weighted_cov(returns_df)
        cov = cov_df.values
        shrinkage = None
    else:
        raise ValueError(f"Unknown method: {method}")

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    min_eig = float(eigenvalues.min())
    max_eig = float(eigenvalues.max())

    # Condition number
    if min_eig > 1e-12:
        condition_number = max_eig / min_eig
    else:
        condition_number = np.inf

    # Rank
    rank = np.linalg.matrix_rank(cov)

    # Positive definite check
    is_positive_definite = min_eig > 0

    return {
        'method': method,
        'condition_number': float(condition_number),
        'min_eigenvalue': float(min_eig),
        'max_eigenvalue': float(max_eig),
        'rank': int(rank),
        'dimension': cov.shape[0],
        'is_positive_definite': bool(is_positive_definite),
        'shrinkage_intensity': float(shrinkage) if shrinkage is not None else None,
        'recommendation': _get_recommendation(condition_number, is_positive_definite, shrinkage)
    }


def _get_recommendation(condition_number: float, is_positive_definite: bool, shrinkage: Optional[float]) -> str:
    """Generate diagnostic recommendation."""
    if not is_positive_definite:
        return "❌ Matrix not positive-definite. Use robust estimator (Ledoit-Wolf, OAS)."

    if condition_number > 1000:
        if shrinkage is not None and shrinkage > 0.5:
            return f"⚠️ High condition number ({condition_number:.0f}), but shrinkage ({shrinkage:.2f}) helps. Consider OAS."
        else:
            return f"❌ Ill-conditioned matrix (κ={condition_number:.0f}). Use Ledoit-Wolf shrinkage."

    if condition_number > 100:
        return f"⚠️ Moderate condition number ({condition_number:.0f}). Robust estimator recommended."

    return f"✅ Well-conditioned matrix (κ={condition_number:.0f}). Sample covariance OK."


# ============================================================================
# Integration with existing Kelly (backwards compatibility)
# ============================================================================

def kelly_vectorial_with_fallback(
    returns_df: pd.DataFrame,
    base_kelly: float = 0.25,
    covariance_method: str = 'sample',
    fallback_to_equal: bool = True,
    **kwargs
) -> pd.Series:
    """
    Kelly vectorial with automatic fallback to equal weight if optimization fails.

    This function is designed for production use where robustness is critical.

    Args:
        returns_df: Asset returns DataFrame
        base_kelly: Fractional Kelly (0 < κ ≤ 1)
        covariance_method: Covariance estimation method
        fallback_to_equal: If True, fallback to equal weight on error
        **kwargs: Additional arguments for kelly_vectorial_weights

    Returns:
        pd.Series with weights (guaranteed to sum to 1)
    """
    try:
        weights = kelly_vectorial_weights(
            returns_df,
            base_kelly=base_kelly,
            covariance_method=covariance_method,
            **kwargs
        )

        # Validation
        if weights.sum() < 0.01 or weights.isna().any():
            raise ValueError("Invalid weights produced")

        return weights

    except Exception as e:
        if fallback_to_equal:
            print(f"Warning: Kelly vectorial failed ({e}), falling back to equal weight")
            n_assets = len(returns_df.columns)
            return pd.Series(1.0 / n_assets, index=returns_df.columns)
        else:
            raise


# ============================================================================
# Utility functions
# ============================================================================

def effective_number_of_assets(weights: pd.Series) -> float:
    """
    Calculate effective number of assets (Herfindahl index).

    N_eff = 1 / Σ(w_i^2)

    Interpretation:
        - N_eff = 1: All weight on one asset (max concentration)
        - N_eff = N: Equal weight across N assets (max diversification)

    Example:
        >>> w = pd.Series([0.5, 0.3, 0.2])  # 3 assets
        >>> N_eff = effective_number_of_assets(w)
        >>> print(f"N_eff = {N_eff:.2f}")  # ~2.63
    """
    return float(1.0 / np.sum(weights.values ** 2))


def concentration_ratio(weights: pd.Series, top_n: int = 5) -> float:
    """
    Calculate concentration ratio (sum of top N weights).

    CR_5 = w_1 + w_2 + w_3 + w_4 + w_5 (sorted descending)

    Interpretation:
        - CR_5 = 1.0: All weight in top 5 assets
        - CR_5 = 0.5: Half the weight in top 5 assets
    """
    sorted_weights = weights.sort_values(ascending=False)
    return float(sorted_weights.head(top_n).sum())
