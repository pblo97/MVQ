# portfolio_manager/risk/cvar_analysis.py
"""
CVaR (Conditional Value at Risk) and VaR Analysis

Based on academic literature:
- Rockafellar & Uryasev (2000, 2002): CVaR optimization
- Acerbi & Tasche (2002): Expected Shortfall properties
- Jorion (2007): Value at Risk methods

Implements:
1. VaR and CVaR calculation (historical, parametric, cornish-fisher)
2. Marginal CVaR contributions (portfolio risk attribution)
3. Component CVaR (additive decomposition)
4. VaR backtesting (Kupiec, Christoffersen tests)
5. Stress testing scenarios
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List


# ========== VAR & CVAR CALCULATION ==========

def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level (default 95%)
        method: 'historical', 'parametric', or 'cornish_fisher'

    Returns:
        VaR (positive number, represents loss)
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    alpha = 1 - confidence_level

    if method == 'historical':
        # Historical simulation
        var = -np.percentile(returns, alpha * 100)

    elif method == 'parametric':
        # Parametric (normal distribution)
        mu = returns.mean()
        sigma = returns.std()
        var = -(mu + stats.norm.ppf(alpha) * sigma)

    elif method == 'cornish_fisher':
        # Cornish-Fisher expansion (accounts for skew and kurtosis)
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        z = stats.norm.ppf(alpha)
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = -(mu + z_cf * sigma)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(var)


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR = expected return given that return is worse than VaR.

    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level (default 95%)
        method: 'historical', 'parametric', or 'cornish_fisher'

    Returns:
        CVaR (positive number, represents expected tail loss)
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    alpha = 1 - confidence_level

    if method == 'historical':
        # Historical CVaR: mean of returns below VaR
        var = calculate_var(returns, confidence_level, method='historical')
        tail_returns = returns[returns <= -var]
        cvar = -tail_returns.mean() if not tail_returns.empty else var

    elif method == 'parametric':
        # Parametric (normal distribution)
        mu = returns.mean()
        sigma = returns.std()
        z_alpha = stats.norm.ppf(alpha)
        cvar = -(mu - sigma * stats.norm.pdf(z_alpha) / alpha)

    elif method == 'cornish_fisher':
        # Use historical CVaR with Cornish-Fisher VaR
        var = calculate_var(returns, confidence_level, method='cornish_fisher')
        tail_returns = returns[returns <= -var]
        cvar = -tail_returns.mean() if not tail_returns.empty else var

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(cvar)


# ========== MARGINAL CVAR ==========

def calculate_marginal_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate Marginal CVaR contributions (∂CVaR/∂w_i).

    Marginal CVaR = sensitivity of portfolio CVaR to small change in asset weight.

    Based on: Tasche (2002) "Expected Shortfall and Beyond"

    Args:
        returns_df: DataFrame with asset returns (columns = assets)
        weights: Portfolio weights (array)
        confidence_level: Confidence level (default 95%)
        method: VaR/CVaR calculation method

    Returns:
        Series of marginal CVaR by asset
    """
    returns_df = returns_df.dropna()
    if returns_df.empty or len(weights) != len(returns_df.columns):
        return pd.Series(np.nan, index=returns_df.columns)

    # Portfolio returns
    portfolio_returns = (returns_df * weights).sum(axis=1)

    # Portfolio CVaR
    port_var = calculate_var(portfolio_returns, confidence_level, method)

    # Tail conditional expectation (E[R_i | R_p < -VaR])
    tail_mask = portfolio_returns <= -port_var

    if tail_mask.sum() == 0:
        # No tail events
        return pd.Series(0.0, index=returns_df.columns)

    # Marginal CVaR = E[R_i | R_p in tail]
    marginal_cvar = returns_df[tail_mask].mean()

    return marginal_cvar


def calculate_component_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate Component CVaR (additive risk decomposition).

    Component CVaR_i = w_i × Marginal CVaR_i

    Property: Σ Component CVaR_i = Portfolio CVaR (Euler decomposition)

    Args:
        returns_df: DataFrame with asset returns
        weights: Portfolio weights
        confidence_level: Confidence level
        method: VaR/CVaR calculation method

    Returns:
        Series of component CVaR by asset
    """
    marginal_cvar = calculate_marginal_cvar(returns_df, weights, confidence_level, method)
    component_cvar = marginal_cvar * weights

    return component_cvar


def calculate_percentage_cvar_contribution(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate percentage CVaR contribution (component / total).

    Args:
        returns_df: DataFrame with asset returns
        weights: Portfolio weights
        confidence_level: Confidence level
        method: VaR/CVaR calculation method

    Returns:
        Series of % CVaR contribution by asset (sums to 100%)
    """
    component_cvar = calculate_component_cvar(returns_df, weights, confidence_level, method)
    total_cvar = component_cvar.sum()

    if total_cvar == 0:
        return pd.Series(0.0, index=returns_df.columns)

    pct_contribution = (component_cvar / total_cvar) * 100

    return pct_contribution


# ========== VAR BACKTESTING ==========

def backtest_var(
    returns: pd.Series,
    var_forecast: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, any]:
    """
    Backtest VaR forecasts using standard tests.

    Tests:
    1. Kupiec Test (POF): Tests if exception rate matches expected rate
    2. Christoffersen Test: Tests for independence of exceptions

    Args:
        returns: Realized returns
        var_forecast: VaR forecasts (positive = loss)
        confidence_level: Confidence level used for VaR

    Returns:
        Dict with test results
    """
    returns = returns.dropna()
    var_forecast = var_forecast.reindex(returns.index).dropna()

    # Align
    common_idx = returns.index.intersection(var_forecast.index)
    returns = returns.loc[common_idx]
    var_forecast = var_forecast.loc[common_idx]

    if len(returns) == 0:
        return {'error': 'No data for backtesting'}

    # Exceptions (violations): realized loss > VaR
    exceptions = (returns < -var_forecast).astype(int)
    n_exceptions = exceptions.sum()
    n_obs = len(returns)

    expected_exceptions = n_obs * (1 - confidence_level)
    exception_rate = n_exceptions / n_obs

    # 1. Kupiec POF Test (Proportion of Failures)
    # H0: exception rate = expected rate
    # LR = -2 * log(L(p)/L(p_hat))
    p = 1 - confidence_level
    p_hat = exception_rate

    if p_hat == 0 or p_hat == 1:
        kupiec_lr = np.nan
        kupiec_pvalue = np.nan
    else:
        try:
            kupiec_lr = -2 * (
                n_exceptions * np.log(p) +
                (n_obs - n_exceptions) * np.log(1 - p) -
                n_exceptions * np.log(p_hat) -
                (n_obs - n_exceptions) * np.log(1 - p_hat)
            )
            kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_lr, df=1)
        except:
            kupiec_lr = np.nan
            kupiec_pvalue = np.nan

    # 2. Christoffersen Independence Test
    # H0: exceptions are independent (no clustering)
    transitions = pd.DataFrame({
        'current': exceptions,
        'next': exceptions.shift(-1)
    }).dropna()

    if len(transitions) < 10:
        christ_lr = np.nan
        christ_pvalue = np.nan
    else:
        n00 = ((transitions['current'] == 0) & (transitions['next'] == 0)).sum()
        n01 = ((transitions['current'] == 0) & (transitions['next'] == 1)).sum()
        n10 = ((transitions['current'] == 1) & (transitions['next'] == 0)).sum()
        n11 = ((transitions['current'] == 1) & (transitions['next'] == 1)).sum()

        # Transition probabilities
        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / len(transitions)

        try:
            if pi_0 > 0 and pi_1 > 0 and pi > 0:
                christ_lr = -2 * (
                    n00 * np.log(1 - pi) + n01 * np.log(pi) +
                    n10 * np.log(1 - pi) + n11 * np.log(pi) -
                    n00 * np.log(1 - pi_0) - n01 * np.log(pi_0) -
                    n10 * np.log(1 - pi_1) - n11 * np.log(pi_1)
                )
                christ_pvalue = 1 - stats.chi2.cdf(christ_lr, df=1)
            else:
                christ_lr = np.nan
                christ_pvalue = np.nan
        except:
            christ_lr = np.nan
            christ_pvalue = np.nan

    return {
        'n_obs': n_obs,
        'n_exceptions': int(n_exceptions),
        'expected_exceptions': expected_exceptions,
        'exception_rate': exception_rate,
        'expected_rate': p,
        'kupiec_lr': kupiec_lr,
        'kupiec_pvalue': kupiec_pvalue,
        'kupiec_reject_h0': kupiec_pvalue < 0.05 if not np.isnan(kupiec_pvalue) else None,
        'christoffersen_lr': christ_lr,
        'christoffersen_pvalue': christ_pvalue,
        'christoffersen_reject_h0': christ_pvalue < 0.05 if not np.isnan(christ_pvalue) else None
    }


# ========== STRESS TESTING ==========

def stress_test_scenarios(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    scenarios: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Stress test portfolio against defined scenarios.

    Args:
        returns_df: DataFrame with asset returns
        weights: Portfolio weights
        scenarios: Dict of {scenario_name: {asset: shock}}
                   If None, uses default scenarios

    Returns:
        DataFrame with stress test results
    """
    if scenarios is None:
        # Default scenarios (academic stress tests)
        scenarios = get_default_stress_scenarios(returns_df)

    results = []

    for scenario_name, shocks in scenarios.items():
        # Apply shocks to assets
        shocked_returns = pd.Series(index=returns_df.columns, dtype=float)

        for asset in returns_df.columns:
            shock = shocks.get(asset, 0.0)  # 0 if asset not in scenario
            shocked_returns[asset] = shock

        # Portfolio impact
        portfolio_impact = (shocked_returns * weights).sum()

        results.append({
            'scenario': scenario_name,
            'portfolio_loss_pct': portfolio_impact * 100,
            **{f"{asset}_shock": shocks.get(asset, 0.0) * 100
               for asset in returns_df.columns[:5]}  # First 5 assets
        })

    return pd.DataFrame(results)


def get_default_stress_scenarios(returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Get default stress scenarios based on historical events.

    Scenarios based on:
    - 2008 Financial Crisis
    - 2020 COVID Crash
    - 2022 Rate Hike Shock
    - Plus statistical scenarios (3-sigma, 5-sigma)

    Args:
        returns_df: DataFrame with asset returns

    Returns:
        Dict of scenarios
    """
    # Calculate historical statistics
    mean_returns = returns_df.mean()
    std_returns = returns_df.std()

    scenarios = {}

    # 1. Financial Crisis 2008 (Oct 2008: -15% to -40%)
    scenarios['2008_financial_crisis'] = {
        asset: -0.30 for asset in returns_df.columns  # -30% uniform shock
    }

    # 2. COVID Crash 2020 (Mar 2020: -30% to -40%)
    scenarios['2020_covid_crash'] = {
        asset: -0.35 for asset in returns_df.columns  # -35% uniform shock
    }

    # 3. Rate Hike 2022 (Growth stocks -20% to -40%, value -10%)
    # Proxy: tech/growth more sensitive
    scenarios['2022_rate_hike'] = {
        asset: -0.25 for asset in returns_df.columns  # -25% average
    }

    # 4. Statistical: 3-sigma event
    scenarios['3_sigma_down'] = {
        asset: float(mean_returns[asset] - 3 * std_returns[asset])
        for asset in returns_df.columns
    }

    # 5. Statistical: 5-sigma event (tail risk)
    scenarios['5_sigma_down'] = {
        asset: float(mean_returns[asset] - 5 * std_returns[asset])
        for asset in returns_df.columns
    }

    # 6. Correlation breakdown (all assets down together)
    scenarios['correlation_one'] = {
        asset: -0.20 for asset in returns_df.columns  # All down 20%
    }

    return scenarios


# ========== PORTFOLIO RISK SUMMARY ==========

def calculate_risk_metrics_summary(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> pd.DataFrame:
    """
    Calculate comprehensive risk metrics summary.

    Args:
        returns_df: DataFrame with asset returns
        weights: Portfolio weights
        confidence_levels: List of confidence levels for VaR/CVaR

    Returns:
        DataFrame with risk metrics
    """
    portfolio_returns = (returns_df * weights).sum(axis=1)

    results = []

    for cl in confidence_levels:
        var_hist = calculate_var(portfolio_returns, cl, method='historical')
        cvar_hist = calculate_cvar(portfolio_returns, cl, method='historical')
        var_param = calculate_var(portfolio_returns, cl, method='parametric')
        cvar_param = calculate_cvar(portfolio_returns, cl, method='parametric')

        results.append({
            'confidence_level': f"{cl*100:.0f}%",
            'VaR_historical': var_hist * 100,
            'CVaR_historical': cvar_hist * 100,
            'VaR_parametric': var_param * 100,
            'CVaR_parametric': cvar_param * 100
        })

    return pd.DataFrame(results)
