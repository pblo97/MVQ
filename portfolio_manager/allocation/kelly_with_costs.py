# portfolio_manager/allocation/kelly_with_costs.py
"""
Kelly Criterion with Transaction Costs

Implements Kelly portfolio optimization with transaction costs integrated
into the optimization objective:

    max E[log(1 + R)] - cost × turnover

Where:
- E[log(1 + R)]: Expected log return (Kelly objective)
- cost: Transaction cost rate (bps)
- turnover: ||w_new - w_old||_1 (sum of absolute weight changes)

This formulation trades off between:
1. Growth maximization (Kelly)
2. Turnover minimization (cost reduction)

Academic References:
- Gârleanu & Pedersen (2013): Dynamic Trading with Predictable Returns and Transaction Costs
- Liu & Loewenstein (2002): Optimal Portfolio Selection with Transaction Costs
- DeMiguel et al. (2009): Optimal Versus Naive Diversification
- Kozak et al. (2020): Shrinking the Cross-Section
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict
import warnings


def kelly_with_transaction_costs(
    returns_df: pd.DataFrame,
    current_weights: Optional[pd.Series] = None,
    base_kelly: float = 0.25,
    transaction_cost_bps: float = 10.0,
    cost_penalty_lambda: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 0.5,
    long_only: bool = True,
    method: str = 'SLSQP',
    max_iter: int = 500
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Optimize Kelly portfolio with transaction costs.

    Objective:
        max_w E[log(1 + w^T × r)] - λ × cost × ||w - w_old||_1

    Where:
        - w: new portfolio weights
        - r: asset returns
        - w_old: current portfolio weights
        - cost: transaction cost rate (bps / 10000)
        - λ: penalty multiplier for cost sensitivity

    Args:
        returns_df: DataFrame with asset returns (rows=dates, cols=assets)
        current_weights: Current portfolio weights (for turnover calculation)
                        If None, assumes starting from cash (all 0s)
        base_kelly: Fractional Kelly parameter (scaling factor)
        transaction_cost_bps: Transaction cost in basis points (e.g., 10 bps = 0.1%)
        cost_penalty_lambda: Multiplier for cost penalty (higher = more cost-averse)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        long_only: If True, force all weights ≥ 0
        method: Optimization method ('SLSQP', 'trust-constr')
        max_iter: Maximum iterations

    Returns:
        (optimal_weights, diagnostics)
        - optimal_weights: pd.Series with optimal weights
        - diagnostics: Dict with objective value, turnover, cost, etc.

    Example:
        >>> returns = pd.DataFrame({'AAPL': [...], 'GOOGL': [...]})
        >>> current_w = pd.Series({'AAPL': 0.6, 'GOOGL': 0.4})
        >>> new_w, diag = kelly_with_transaction_costs(
        ...     returns, current_w, transaction_cost_bps=10
        ... )
        >>> print(f"Turnover: {diag['turnover']:.2%}, Cost: {diag['cost_pct']:.2%}")
    """
    # Validate inputs
    if returns_df.empty or len(returns_df) < 10:
        raise ValueError("Insufficient returns data (need at least 10 observations)")

    n_assets = len(returns_df.columns)

    # Initialize current weights (if None, start from cash)
    if current_weights is None:
        current_weights = pd.Series(0.0, index=returns_df.columns)
    else:
        # Ensure alignment
        current_weights = current_weights.reindex(returns_df.columns, fill_value=0.0)

    w_old = current_weights.values

    # Convert cost to decimal
    cost_rate = transaction_cost_bps / 10000.0

    # Expected returns and covariance
    mu = returns_df.mean().values
    returns_matrix = returns_df.values

    # Objective function: -E[log(1 + R)] + λ × cost × turnover
    def objective(w):
        # Expected log return (Kelly objective)
        portfolio_returns = returns_matrix @ w

        # Avoid log(negative) by clipping
        portfolio_returns_shifted = 1.0 + portfolio_returns
        portfolio_returns_shifted = np.clip(portfolio_returns_shifted, 1e-10, None)

        expected_log_return = np.mean(np.log(portfolio_returns_shifted))

        # Turnover cost
        turnover = np.sum(np.abs(w - w_old))
        transaction_cost = cost_penalty_lambda * cost_rate * turnover

        # Negative because we want to maximize (scipy minimizes)
        return -(expected_log_return - transaction_cost)

    # Gradient (optional, for faster convergence)
    def gradient(w):
        portfolio_returns = returns_matrix @ w
        portfolio_returns_shifted = 1.0 + portfolio_returns
        portfolio_returns_shifted = np.clip(portfolio_returns_shifted, 1e-10, None)

        # Gradient of E[log(1 + R)]
        grad_kelly = np.mean(returns_matrix / portfolio_returns_shifted[:, np.newaxis], axis=0)

        # Gradient of turnover penalty
        grad_cost = cost_penalty_lambda * cost_rate * np.sign(w - w_old)

        return -(grad_kelly - grad_cost)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
    ]

    # Bounds
    if long_only:
        bounds = [(max(min_weight, 0.0), max_weight) for _ in range(n_assets)]
    else:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

    # Initial guess (start from current weights or equal weight)
    if np.sum(w_old) > 0.01:
        w0 = w_old / np.sum(w_old)  # Normalize
    else:
        w0 = np.ones(n_assets) / n_assets

    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective,
            w0,
            method=method,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )

    if not result.success:
        warnings.warn(f"Optimization did not converge: {result.message}")

    # Extract optimal weights
    w_optimal = result.x

    # Calculate diagnostics
    portfolio_returns_opt = returns_matrix @ w_optimal
    expected_return = np.mean(portfolio_returns_opt)
    expected_log_return = np.mean(np.log(1.0 + np.clip(portfolio_returns_opt, -0.999, None)))

    turnover = np.sum(np.abs(w_optimal - w_old))
    transaction_cost_pct = cost_rate * turnover

    diagnostics = {
        'expected_return': float(expected_return),
        'expected_log_return': float(expected_log_return),
        'turnover': float(turnover),
        'cost_rate_bps': float(transaction_cost_bps),
        'cost_pct': float(transaction_cost_pct),
        'cost_penalty_lambda': float(cost_penalty_lambda),
        'objective_value': float(-result.fun),  # Negate back
        'optimization_success': bool(result.success),
        'n_iterations': int(result.nit) if hasattr(result, 'nit') else 0,
        'method': method
    }

    return pd.Series(w_optimal, index=returns_df.columns), diagnostics


def compare_with_without_costs(
    returns_df: pd.DataFrame,
    current_weights: Optional[pd.Series] = None,
    transaction_cost_bps: float = 10.0,
    base_kelly: float = 0.25,
    **kwargs
) -> pd.DataFrame:
    """
    Compare Kelly weights with and without transaction costs.

    Returns:
        DataFrame with columns: ['no_costs', 'with_costs', 'difference']
    """
    # Without costs (vanilla Kelly)
    from portfolio_manager.allocation.kelly_vectorial import kelly_vectorial_weights
    w_no_costs = kelly_vectorial_weights(
        returns_df,
        base_kelly=base_kelly,
        covariance_method='sample',
        **kwargs
    )

    # With costs
    w_with_costs, _ = kelly_with_transaction_costs(
        returns_df,
        current_weights=current_weights,
        base_kelly=base_kelly,
        transaction_cost_bps=transaction_cost_bps,
        **kwargs
    )

    comparison = pd.DataFrame({
        'no_costs': w_no_costs,
        'with_costs': w_with_costs,
        'difference': w_with_costs - w_no_costs
    })

    return comparison


def optimal_rebalancing_frequency(
    returns_df: pd.DataFrame,
    current_weights: pd.Series,
    frequencies: list = [1, 5, 21, 63, 126],  # Daily, weekly, monthly, quarterly, semi-annual
    transaction_cost_bps: float = 10.0,
    **kwargs
) -> pd.DataFrame:
    """
    Determine optimal rebalancing frequency by simulating different frequencies.

    Args:
        returns_df: Historical returns
        current_weights: Current portfolio weights
        frequencies: List of rebalancing frequencies (in days)
        transaction_cost_bps: Transaction cost (bps)
        **kwargs: Additional arguments for kelly_with_transaction_costs

    Returns:
        DataFrame with columns: frequency, net_return, turnover, cost
    """
    results = []

    for freq in frequencies:
        # Simulate rebalancing at this frequency
        n_periods = len(returns_df) // freq

        total_return = 0.0
        total_turnover = 0.0
        total_cost = 0.0

        w_current = current_weights.copy()

        for i in range(n_periods):
            start_idx = i * freq
            end_idx = min((i + 1) * freq, len(returns_df))

            period_returns = returns_df.iloc[start_idx:end_idx]

            # Optimize for this period
            w_new, diag = kelly_with_transaction_costs(
                period_returns,
                current_weights=w_current,
                transaction_cost_bps=transaction_cost_bps,
                **kwargs
            )

            # Calculate period return
            period_ret = (period_returns.values @ w_new.values).mean()

            total_return += period_ret
            total_turnover += diag['turnover']
            total_cost += diag['cost_pct']

            # Update for next period (account for drift)
            # Simplified: just use new weights
            w_current = w_new

        net_return = total_return - total_cost

        results.append({
            'frequency_days': freq,
            'frequency_label': _freq_label(freq),
            'gross_return': total_return,
            'total_turnover': total_turnover,
            'total_cost': total_cost,
            'net_return': net_return,
            'n_rebalances': n_periods
        })

    return pd.DataFrame(results).sort_values('net_return', ascending=False)


def _freq_label(days: int) -> str:
    """Convert days to frequency label."""
    if days == 1:
        return "Daily"
    elif days <= 7:
        return "Weekly"
    elif days <= 30:
        return "Monthly"
    elif days <= 90:
        return "Quarterly"
    elif days <= 180:
        return "Semi-Annual"
    else:
        return "Annual"


def turnover_aware_kelly(
    returns_df: pd.DataFrame,
    current_weights: pd.Series,
    target_turnover: float = 0.10,
    transaction_cost_bps: float = 10.0,
    **kwargs
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Kelly optimization with target turnover constraint.

    Instead of penalty, directly constrains turnover to be ≤ target.

    Args:
        returns_df: Asset returns
        current_weights: Current weights
        target_turnover: Maximum allowed turnover (e.g., 0.10 = 10%)
        transaction_cost_bps: Transaction cost (bps)
        **kwargs: Additional arguments

    Returns:
        (optimal_weights, diagnostics)
    """
    n_assets = len(returns_df.columns)
    w_old = current_weights.reindex(returns_df.columns, fill_value=0.0).values

    mu = returns_df.mean().values
    returns_matrix = returns_df.values
    cost_rate = transaction_cost_bps / 10000.0

    # Objective: E[log(1 + R)]
    def objective(w):
        portfolio_returns = returns_matrix @ w
        portfolio_returns_shifted = np.clip(1.0 + portfolio_returns, 1e-10, None)
        return -np.mean(np.log(portfolio_returns_shifted))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: target_turnover - np.sum(np.abs(w - w_old))}  # turnover ≤ target
    ]

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}
    )

    w_optimal = result.x
    turnover = np.sum(np.abs(w_optimal - w_old))
    cost_pct = cost_rate * turnover

    diagnostics = {
        'turnover': float(turnover),
        'cost_pct': float(cost_pct),
        'target_turnover': float(target_turnover),
        'turnover_utilized': float(turnover / target_turnover) if target_turnover > 0 else 0.0,
        'optimization_success': bool(result.success)
    }

    return pd.Series(w_optimal, index=returns_df.columns), diagnostics
