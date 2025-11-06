# portfolio_manager/optimization/parameter_search.py
"""
Parameter Grid Search with Walk-Forward Cross-Validation

Optimizes hyperparameters using time-series cross-validation to avoid overfitting.

Parameters typically optimized:
- Kelly fraction (base_kelly)
- Training window size (train_window)
- Test window size (test_window)
- Correlation penalty (lambda_corr)
- Transaction cost penalty (cost_penalty_lambda)
- Robust covariance method (cov_method)
- Rebalancing frequency

Academic References:
- Bergmeir & Benítez (2012): On the use of cross-validation for time series predictor evaluation
- Hsu et al. (2003): A Practical Guide to Support Vector Classification
- Harvey et al. (2016): ... and the Cross-Section of Expected Returns
- López de Prado (2018): Advances in Financial Machine Learning (Ch. 7 - Cross-Validation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple, Optional
from itertools import product
from dataclasses import dataclass
import warnings


@dataclass
class ParameterSearchResult:
    """Results from parameter grid search."""
    best_params: Dict[str, any]
    best_score: float
    cv_results: pd.DataFrame  # All parameter combinations and scores
    scoring_metric: str
    n_folds: int
    total_evaluations: int
    errors: List[str] = None  # Captured error messages for debugging


def walk_forward_cross_validation(
    returns_df: pd.DataFrame,
    strategy_func: Callable,
    param_grid: Dict[str, List],
    scoring: str = 'sharpe',
    n_splits: int = 5,
    train_size: int = 504,  # 2 years daily
    test_size: int = 126,   # 6 months daily
    min_train_obs: int = 252,
    verbose: bool = True
) -> ParameterSearchResult:
    """
    Perform walk-forward cross-validation with grid search.

    Args:
        returns_df: Asset returns DataFrame
        strategy_func: Strategy function that takes (returns, **params) and returns weights
        param_grid: Dictionary of parameter lists to search
            Example: {'base_kelly': [0.1, 0.25, 0.5], 'lambda_corr': [0.1, 0.25, 0.5]}
        scoring: Scoring metric ('sharpe', 'sortino', 'information_ratio', 'calmar')
        n_splits: Number of walk-forward splits
        train_size: Training window size (trading days)
        test_size: Test window size (trading days)
        min_train_obs: Minimum observations for training
        verbose: Print progress

    Returns:
        ParameterSearchResult with best parameters and CV results
    """
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    if verbose:
        print(f"Grid Search: {len(param_combinations)} parameter combinations × {n_splits} folds = {len(param_combinations) * n_splits} evaluations")

    # Walk-forward splits
    splits = _generate_walk_forward_splits(
        len(returns_df),
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        min_train_obs=min_train_obs
    )

    # Evaluate each parameter combination
    results = []
    errors = []  # Capture unique errors for debugging

    for param_values_tuple in param_combinations:
        params = dict(zip(param_names, param_values_tuple))

        # Cross-validate this parameter set
        fold_scores = []

        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            train_data = returns_df.iloc[train_indices]
            test_data = returns_df.iloc[test_indices]

            try:
                # Train strategy with these parameters
                weights = strategy_func(train_data, **params)

                # Evaluate on test set
                strategy_returns = (test_data.values * weights.values).sum(axis=1)

                # Calculate score
                score = _calculate_score(strategy_returns, scoring)
                fold_scores.append(score)

            except Exception as e:
                error_msg = f"Fold {fold_idx}, params {params}: {type(e).__name__}: {str(e)}"
                if verbose:
                    print(f"Warning: {error_msg}")
                # Store first 20 unique error types
                if len(errors) < 20:
                    error_type = f"{type(e).__name__}: {str(e)[:100]}"
                    if error_type not in errors:
                        errors.append(error_msg)
                fold_scores.append(np.nan)

        # Aggregate scores across folds
        mean_score = float(np.nanmean(fold_scores))
        std_score = float(np.nanstd(fold_scores))

        results.append({
            **params,
            f'{scoring}_mean': mean_score,
            f'{scoring}_std': std_score,
            'n_valid_folds': int(np.sum(~np.isnan(fold_scores)))
        })

        if verbose:
            print(f"Params: {params} → {scoring}={mean_score:.3f} ± {std_score:.3f}")

    # Create results DataFrame
    cv_results = pd.DataFrame(results).sort_values(
        f'{scoring}_mean',
        ascending=False
    ).reset_index(drop=True)

    # Best parameters
    best_params = cv_results.iloc[0][param_names].to_dict()
    best_score = cv_results.iloc[0][f'{scoring}_mean']

    return ParameterSearchResult(
        best_params=best_params,
        best_score=float(best_score),
        cv_results=cv_results,
        scoring_metric=scoring,
        n_folds=n_splits,
        total_evaluations=len(param_combinations) * n_splits,
        errors=errors if errors else None
    )


def _generate_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    train_size: int,
    test_size: int,
    min_train_obs: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward split indices.

    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    step_size = test_size  # Step forward by test window

    for i in range(n_splits):
        # Training window
        train_start = i * step_size
        train_end = train_start + train_size

        # Test window
        test_start = train_end
        test_end = test_start + test_size

        # Check bounds
        if test_end > n_samples:
            break

        if train_end - train_start < min_train_obs:
            continue

        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)

        splits.append((train_indices, test_indices))

    return splits


def _calculate_score(returns: np.ndarray, scoring: str) -> float:
    """Calculate scoring metric from returns."""
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return np.nan

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if scoring == 'sharpe':
        # Sharpe ratio (annualized)
        if std_ret > 0:
            return float((mean_ret / std_ret) * np.sqrt(252))
        else:
            return 0.0

    elif scoring == 'sortino':
        # Sortino ratio (annualized, downside deviation)
        downside_ret = returns[returns < 0]
        if len(downside_ret) > 0:
            downside_std = np.std(downside_ret, ddof=1)
            if downside_std > 0:
                return float((mean_ret / downside_std) * np.sqrt(252))
        return 0.0

    elif scoring == 'information_ratio':
        # Information ratio (assumes benchmark = 0)
        if std_ret > 0:
            return float(mean_ret / std_ret)
        else:
            return 0.0

    elif scoring == 'calmar':
        # Calmar ratio (return / max drawdown)
        cum_ret = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_dd = np.abs(np.min(drawdown))

        if max_dd > 0:
            total_ret = cum_ret[-1] - 1
            return float(total_ret / max_dd)
        else:
            return 0.0

    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")


def optimize_kelly_parameters(
    returns_df: pd.DataFrame,
    param_ranges: Optional[Dict[str, List]] = None,
    scoring: str = 'sharpe',
    n_splits: int = 5,
    verbose: bool = True
) -> ParameterSearchResult:
    """
    Optimize Kelly Criterion parameters using walk-forward CV.

    Args:
        returns_df: Asset returns
        param_ranges: Parameter grid (if None, use defaults)
        scoring: Scoring metric
        n_splits: Number of CV folds
        verbose: Print progress

    Returns:
        ParameterSearchResult
    """
    # Default parameter grid
    if param_ranges is None:
        param_ranges = {
            'base_kelly': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            'lambda_corr': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            'winsor_p': [0.005, 0.01, 0.02, 0.05]
        }

    # Simple Kelly strategy
    def kelly_strategy(train_returns, base_kelly=0.25, lambda_corr=0.25, winsor_p=0.01):
        # Winsorize
        train_w = train_returns.clip(
            lower=train_returns.quantile(winsor_p),
            upper=train_returns.quantile(1 - winsor_p)
        )

        # Kelly weights
        mu = train_w.mean()
        cov = train_w.cov()

        try:
            cov_inv = np.linalg.inv(cov.values + np.eye(len(cov)) * 1e-8)
            weights_raw = base_kelly * (cov_inv @ mu.values)
            weights_raw = np.clip(weights_raw, 0, None)

            if weights_raw.sum() > 0:
                weights = weights_raw / weights_raw.sum()
            else:
                weights = np.ones(len(mu)) / len(mu)
        except:
            weights = np.ones(len(mu)) / len(mu)

        return pd.Series(weights, index=train_returns.columns)

    # Run grid search
    return walk_forward_cross_validation(
        returns_df,
        kelly_strategy,
        param_ranges,
        scoring=scoring,
        n_splits=n_splits,
        verbose=verbose
    )


def compare_parameter_sets(
    results: ParameterSearchResult,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Compare top N parameter sets from grid search.

    Args:
        results: ParameterSearchResult from grid search
        top_n: Number of top results to show

    Returns:
        DataFrame with top N parameter sets
    """
    return results.cv_results.head(top_n)


def analyze_parameter_sensitivity(
    results: ParameterSearchResult,
    param_name: str
) -> pd.DataFrame:
    """
    Analyze sensitivity to a specific parameter.

    Shows how scoring metric varies with parameter value (averaging over other params).

    Args:
        results: ParameterSearchResult
        param_name: Parameter to analyze

    Returns:
        DataFrame with mean score per parameter value
    """
    if param_name not in results.cv_results.columns:
        raise ValueError(f"Parameter '{param_name}' not in results")

    score_col = f'{results.scoring_metric}_mean'

    sensitivity = results.cv_results.groupby(param_name)[score_col].agg(['mean', 'std', 'count'])
    sensitivity = sensitivity.sort_index().reset_index()

    return sensitivity


def recommend_parameters(
    returns_df: pd.DataFrame,
    strategy_type: str = 'kelly',
    risk_tolerance: str = 'moderate',
    verbose: bool = True
) -> Dict[str, any]:
    """
    Recommend optimal parameters based on strategy type and risk tolerance.

    Args:
        returns_df: Historical returns
        strategy_type: 'kelly', 'hrp', or 'kelly_with_costs'
        risk_tolerance: 'conservative', 'moderate', 'aggressive'
        verbose: Print recommendations

    Returns:
        Dict with recommended parameters
    """
    # Define search spaces based on risk tolerance
    if risk_tolerance == 'conservative':
        param_ranges = {
            'base_kelly': [0.10, 0.15, 0.20],
            'lambda_corr': [0.5, 0.75, 1.0],
            'winsor_p': [0.01, 0.02]
        }
        scoring = 'sortino'  # Favor downside protection

    elif risk_tolerance == 'moderate':
        param_ranges = {
            'base_kelly': [0.15, 0.20, 0.25, 0.30],
            'lambda_corr': [0.25, 0.5, 0.75],
            'winsor_p': [0.01, 0.02]
        }
        scoring = 'sharpe'

    else:  # aggressive
        param_ranges = {
            'base_kelly': [0.25, 0.30, 0.40, 0.50],
            'lambda_corr': [0.0, 0.1, 0.25],
            'winsor_p': [0.005, 0.01]
        }
        scoring = 'sharpe'

    # Run optimization
    results = optimize_kelly_parameters(
        returns_df,
        param_ranges=param_ranges,
        scoring=scoring,
        n_splits=5,
        verbose=verbose
    )

    if verbose:
        print(f"\n✅ Recommended Parameters ({risk_tolerance}, {strategy_type}):")
        for param, value in results.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nCross-Validated {scoring.capitalize()}: {results.best_score:.3f}")

    return results.best_params
