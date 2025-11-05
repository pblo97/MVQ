# portfolio_manager/estimation/robust_cov.py
"""
Robust Covariance Estimation

Implements shrinkage estimators to address estimation error in sample covariance matrices.

Methods:
1. Ledoit-Wolf (2004): Optimal shrinkage towards constant correlation matrix
2. Ledoit-Wolf (2003): Optimal shrinkage towards identity matrix
3. Exponentially Weighted (RiskMetrics): Time-decay weights for recent data

Academic Foundation:
- Ledoit & Wolf (2003): "Improved Estimation of the Covariance Matrix..."
- Ledoit & Wolf (2004): "Honey, I Shrunk the Sample Covariance Matrix"
- Ledoit & Wolf (2004): "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"

Problem: Sample covariance matrix Σ̂ is noisy, especially when n ~ p (obs ~ assets)
Solution: Shrink towards structured estimator F: Σ_shrunk = δF + (1-δ)Σ̂
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.covariance import LedoitWolf, OAS


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    shrinkage_target: str = 'constant_correlation'
) -> Tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf optimal shrinkage covariance estimator.

    Shrinks sample covariance towards a structured target.

    Args:
        returns: DataFrame of asset returns
        shrinkage_target: 'constant_correlation' (Ledoit-Wolf 2004)
                          'identity' (Ledoit-Wolf 2003)
                          'auto' (sklearn implementation)

    Returns:
        (Shrunk covariance matrix, shrinkage intensity δ)
    """
    returns = returns.dropna()

    if returns.empty or len(returns) < 2:
        # Fallback to sample covariance
        return returns.cov(), 0.0

    X = returns.values
    n, p = X.shape

    if shrinkage_target == 'auto':
        # Use sklearn's implementation (fast, automatic)
        lw = LedoitWolf()
        lw.fit(X)
        cov_shrunk = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        delta = lw.shrinkage_
        return cov_shrunk, delta

    # Sample covariance
    S = returns.cov().values

    if shrinkage_target == 'constant_correlation':
        # Target: constant correlation matrix (Ledoit-Wolf 2004)
        # F_ij = sqrt(S_ii * S_jj) * rho_avg if i != j else S_ii

        # Average pairwise correlation
        std_devs = np.sqrt(np.diag(S))
        corr = S / np.outer(std_devs, std_devs)
        np.fill_diagonal(corr, 0)  # Exclude diagonal
        rho_avg = corr.sum() / (p * (p - 1))

        # Target matrix F
        F = np.outer(std_devs, std_devs) * rho_avg
        np.fill_diagonal(F, np.diag(S))  # Keep variances

    elif shrinkage_target == 'identity':
        # Target: scaled identity matrix (Ledoit-Wolf 2003)
        # F = trace(S)/p * I
        trace_S = np.trace(S)
        F = (trace_S / p) * np.eye(p)

    else:
        raise ValueError(f"Unknown shrinkage_target: {shrinkage_target}")

    # Optimal shrinkage intensity (Ledoit-Wolf formula)
    delta = _compute_shrinkage_intensity(X, S, F)

    # Shrunk covariance
    Sigma_shrunk = delta * F + (1 - delta) * S

    cov_shrunk = pd.DataFrame(Sigma_shrunk, index=returns.columns, columns=returns.columns)

    return cov_shrunk, delta


def _compute_shrinkage_intensity(X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
    """
    Compute optimal shrinkage intensity δ using Ledoit-Wolf (2004) formula.

    δ* = min(1, max(0, (π̂ - ρ̂) / γ̂))

    Where:
    - π̂: sum of asymptotic variances of sample covariance entries
    - ρ̂: sum of asymptotic covariances between S and F
    - γ̂: distance between F and S

    Args:
        X: Data matrix (n × p)
        S: Sample covariance (p × p)
        F: Target covariance (p × p)

    Returns:
        Optimal shrinkage intensity δ ∈ [0, 1]
    """
    n, p = X.shape

    # Center data
    X_centered = X - X.mean(axis=0)

    # Gamma: ||F - S||^2
    gamma = np.linalg.norm(F - S, 'fro') ** 2

    if gamma < 1e-10:
        # F and S are already very close
        return 0.0

    # Pi: sum of asymptotic variances
    # Approximation: Σ_ij Var(s_ij) ≈ 1/n * Σ_t (x_t x_t' - S)^2
    pi = 0.0
    for t in range(n):
        x_t = X_centered[t, :].reshape(-1, 1)
        outer = x_t @ x_t.T
        pi += np.linalg.norm(outer - S, 'fro') ** 2
    pi /= n

    # Rho: Σ_ij Cov(s_ij, f_ij)
    # For constant correlation target, this simplifies (see Ledoit-Wolf 2004 Appendix)
    # Approximation: rho ≈ 0 for many target structures
    rho = 0.0  # Conservative approximation

    # Optimal delta
    delta = (pi - rho) / gamma
    delta = max(0.0, min(1.0, delta))  # Clamp to [0, 1]

    return delta


def exponentially_weighted_cov(
    returns: pd.DataFrame,
    span: int = 60
) -> pd.DataFrame:
    """
    Exponentially weighted covariance matrix (RiskMetrics approach).

    More recent data gets higher weight: w_t = (1-λ) * λ^(T-t)

    Args:
        returns: DataFrame of asset returns
        span: Span for exponential weighting (default 60 days ≈ λ=0.94)

    Returns:
        Exponentially weighted covariance matrix
    """
    returns = returns.dropna()

    if returns.empty or len(returns) < 2:
        return returns.cov()

    # Convert span to decay factor λ
    # span = 2/(1-λ) - 1  =>  λ = 1 - 2/(span+1)
    lam = 1 - 2 / (span + 1)

    # Weights
    n = len(returns)
    weights = np.array([(1 - lam) * lam ** (n - 1 - t) for t in range(n)])
    weights /= weights.sum()  # Normalize

    # Weighted mean
    X = returns.values
    mu = (X.T @ weights).reshape(-1, 1)

    # Weighted covariance
    X_centered = X - mu.T
    cov_weighted = (X_centered.T * weights) @ X_centered

    return pd.DataFrame(cov_weighted, index=returns.columns, columns=returns.columns)


def oracle_approximating_shrinkage(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Oracle Approximating Shrinkage (OAS) estimator.

    Alternative to Ledoit-Wolf, sometimes better for smaller samples.

    Based on Chen et al. (2010): "Shrinkage Algorithms for MMSE Covariance Estimation"

    Args:
        returns: DataFrame of asset returns

    Returns:
        (Shrunk covariance matrix, shrinkage intensity)
    """
    returns = returns.dropna()

    if returns.empty or len(returns) < 2:
        return returns.cov(), 0.0

    X = returns.values

    oas = OAS()
    oas.fit(X)

    cov_shrunk = pd.DataFrame(oas.covariance_, index=returns.columns, columns=returns.columns)

    return cov_shrunk, oas.shrinkage_


def compare_covariance_estimators(
    returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare different covariance estimators.

    Metrics:
    - Condition number (lower = more stable)
    - Frobenius norm
    - Shrinkage intensity

    Args:
        returns: DataFrame of asset returns

    Returns:
        DataFrame comparing estimators
    """
    results = []

    # Sample covariance
    cov_sample = returns.cov().values
    results.append({
        'Estimator': 'Sample',
        'Condition Number': np.linalg.cond(cov_sample),
        'Frobenius Norm': np.linalg.norm(cov_sample, 'fro'),
        'Shrinkage': 0.0
    })

    # Ledoit-Wolf (constant correlation)
    cov_lw_cc, delta_cc = ledoit_wolf_shrinkage(returns, 'constant_correlation')
    results.append({
        'Estimator': 'Ledoit-Wolf (CC)',
        'Condition Number': np.linalg.cond(cov_lw_cc.values),
        'Frobenius Norm': np.linalg.norm(cov_lw_cc.values, 'fro'),
        'Shrinkage': delta_cc
    })

    # Ledoit-Wolf (identity)
    cov_lw_id, delta_id = ledoit_wolf_shrinkage(returns, 'identity')
    results.append({
        'Estimator': 'Ledoit-Wolf (Identity)',
        'Condition Number': np.linalg.cond(cov_lw_id.values),
        'Frobenius Norm': np.linalg.norm(cov_lw_id.values, 'fro'),
        'Shrinkage': delta_id
    })

    # OAS
    cov_oas, delta_oas = oracle_approximating_shrinkage(returns)
    results.append({
        'Estimator': 'OAS',
        'Condition Number': np.linalg.cond(cov_oas.values),
        'Frobenius Norm': np.linalg.norm(cov_oas.values, 'fro'),
        'Shrinkage': delta_oas
    })

    # Exponential weighting
    cov_ew = exponentially_weighted_cov(returns)
    results.append({
        'Estimator': 'Exp Weighted',
        'Condition Number': np.linalg.cond(cov_ew.values),
        'Frobenius Norm': np.linalg.norm(cov_ew.values, 'fro'),
        'Shrinkage': np.nan
    })

    return pd.DataFrame(results)


def recommend_covariance_estimator(
    returns: pd.DataFrame,
    n_assets: Optional[int] = None
) -> str:
    """
    Recommend best covariance estimator based on data characteristics.

    Rules of thumb:
    - n >> p: Sample covariance OK
    - n ~ p: Ledoit-Wolf or OAS
    - n < p: Strong shrinkage (identity target)
    - Time series: Exponential weighting

    Args:
        returns: DataFrame of asset returns
        n_assets: Number of assets (if None, uses returns.shape[1])

    Returns:
        Recommended estimator name
    """
    n_obs = len(returns)
    p = n_assets if n_assets else len(returns.columns)

    ratio = n_obs / p

    if ratio > 10:
        return "Sample (sufficient observations)"
    elif ratio > 5:
        return "Ledoit-Wolf (constant correlation)"
    elif ratio > 2:
        return "OAS (Oracle Approximating Shrinkage)"
    else:
        return "Ledoit-Wolf (identity) - high shrinkage needed"
