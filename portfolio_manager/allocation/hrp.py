# portfolio_manager/allocation/hrp.py
"""
Hierarchical Risk Parity (HRP)

Based on López de Prado (2016):
"Building Diversified Portfolios that Outperform Out-of-Sample"
Journal of Portfolio Management, 42(4), 59-69.

HRP addresses three problems of Markowitz optimization:
1. Instability from ill-conditioned covariance matrices
2. Concentration in a few assets
3. Poor out-of-sample performance

Algorithm:
1. Tree Clustering: Group similar assets hierarchically (dendrogram)
2. Quasi-Diagonalization: Reorder covariance matrix
3. Recursive Bisection: Allocate weights top-down

Benefits vs Markowitz:
- No matrix inversion → numerically stable
- Diversification by construction → no extreme weights
- Outperforms out-of-sample (empirically validated)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from typing import Optional, List


def compute_hrp_weights(
    returns: pd.DataFrame,
    linkage_method: str = 'single'
) -> pd.Series:
    """
    Calculate HRP (Hierarchical Risk Parity) weights.

    Args:
        returns: DataFrame of asset returns (columns = assets)
        linkage_method: 'single', 'complete', 'average', 'ward'
                       (López de Prado recommends 'single')

    Returns:
        Series of HRP weights (sums to 1.0)
    """
    returns = returns.dropna()

    if returns.empty or len(returns.columns) < 2:
        # Fallback to equal weight
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    # Step 1: Compute correlation matrix and distance matrix
    corr = returns.corr()

    # Distance = sqrt(0.5 * (1 - correlation))
    # This is a proper metric (satisfies triangle inequality)
    dist = np.sqrt(0.5 * (1 - corr))

    # Convert to condensed distance matrix for scipy
    dist_condensed = squareform(dist.values, checks=False)

    # Step 2: Hierarchical clustering (build dendrogram)
    link = hierarchy.linkage(dist_condensed, method=linkage_method)

    # Get optimal leaf ordering (makes dendrogram more interpretable)
    sort_ix = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link, dist_condensed))

    # Reorder columns by dendrogram
    sorted_columns = returns.columns[sort_ix].tolist()

    # Step 3: Quasi-diagonalization (reorder covariance matrix)
    cov = returns.cov()
    cov_sorted = cov.loc[sorted_columns, sorted_columns]

    # Step 4: Recursive bisection to allocate weights
    weights = _recursive_bisection(cov_sorted)

    return pd.Series(weights, index=cov_sorted.columns)


def _recursive_bisection(cov: pd.DataFrame) -> np.ndarray:
    """
    Recursive bisection algorithm for HRP weight allocation.

    At each step:
    1. Split cluster into two sub-clusters
    2. Allocate weight between sub-clusters inversely proportional to variance
    3. Recurse on each sub-cluster

    Args:
        cov: Covariance matrix (already sorted by dendrogram)

    Returns:
        Array of weights
    """
    weights = pd.Series(1.0, index=cov.columns)
    clusters = [cov.columns.tolist()]  # Start with all assets in one cluster

    while len(clusters) > 0:
        # Pop first cluster
        cluster = clusters.pop(0)

        if len(cluster) == 1:
            # Single asset, nothing to split
            continue

        # Split cluster in half (already sorted by similarity)
        split_point = len(cluster) // 2
        cluster_0 = cluster[:split_point]
        cluster_1 = cluster[split_point:]

        # Compute variance of each sub-cluster
        # Variance of equal-weighted portfolio in cluster
        cov_0 = cov.loc[cluster_0, cluster_0]
        cov_1 = cov.loc[cluster_1, cluster_1]

        # Equal-weighted portfolio variance
        w_equal_0 = np.ones(len(cluster_0)) / len(cluster_0)
        w_equal_1 = np.ones(len(cluster_1)) / len(cluster_1)

        var_0 = w_equal_0 @ cov_0.values @ w_equal_0
        var_1 = w_equal_1 @ cov_1.values @ w_equal_1

        # Inverse variance allocation
        # alpha = var_1 / (var_0 + var_1)
        # cluster_0 gets alpha, cluster_1 gets (1 - alpha)
        alpha = 1.0 - var_0 / (var_0 + var_1)

        # Update weights
        weights[cluster_0] *= alpha
        weights[cluster_1] *= (1 - alpha)

        # Add sub-clusters to queue
        clusters.extend([cluster_0, cluster_1])

    return weights.values


def compute_hrp_with_constraints(
    returns: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    linkage_method: str = 'single'
) -> pd.Series:
    """
    HRP with box constraints (min/max weights per asset).

    Note: Constraints are applied POST-HRP, not during optimization.
    This may violate some HRP properties but ensures practical constraints.

    Args:
        returns: DataFrame of asset returns
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        linkage_method: Clustering method

    Returns:
        Series of constrained HRP weights
    """
    # Calculate unconstrained HRP
    weights = compute_hrp_weights(returns, linkage_method)

    # Apply constraints
    weights = weights.clip(lower=min_weight, upper=max_weight)

    # Renormalize to sum to 1
    weights = weights / weights.sum()

    return weights


def compare_hrp_vs_equal_weight(
    returns: pd.DataFrame,
    periods: int = 252
) -> pd.DataFrame:
    """
    Compare HRP vs Equal Weight performance metrics.

    Args:
        returns: DataFrame of asset returns
        periods: Number of periods for annualization (252 for daily)

    Returns:
        DataFrame comparing metrics
    """
    # HRP weights
    hrp_weights = compute_hrp_weights(returns)
    hrp_returns = (returns * hrp_weights).sum(axis=1)

    # Equal weight
    ew_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    ew_returns = (returns * ew_weights).sum(axis=1)

    # Metrics
    def calc_metrics(ret):
        total_ret = (1 + ret).prod() - 1
        ann_ret = (1 + total_ret) ** (periods / len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(periods)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (ret.cumsum() - ret.cumsum().cummax()).min()
        return {
            'Total Return': total_ret * 100,
            'Annual Return': ann_ret * 100,
            'Annual Vol': ann_vol * 100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd * 100
        }

    results = pd.DataFrame({
        'HRP': calc_metrics(hrp_returns),
        'Equal Weight': calc_metrics(ew_returns)
    }).T

    # Add improvement
    results['vs EW (%)'] = ((results['Sharpe Ratio'] / results.loc['Equal Weight', 'Sharpe Ratio'] - 1) * 100)

    return results


def get_hrp_clusters(
    returns: pd.DataFrame,
    linkage_method: str = 'single',
    n_clusters: Optional[int] = None
) -> pd.Series:
    """
    Get cluster assignments from HRP dendrogram.

    Useful for understanding asset groupings.

    Args:
        returns: DataFrame of asset returns
        linkage_method: Clustering method
        n_clusters: Number of clusters (if None, uses dendrogram structure)

    Returns:
        Series mapping assets to cluster IDs
    """
    returns = returns.dropna()

    if returns.empty or len(returns.columns) < 2:
        return pd.Series(0, index=returns.columns)

    # Correlation and distance
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    dist_condensed = squareform(dist.values, checks=False)

    # Linkage
    link = hierarchy.linkage(dist_condensed, method=linkage_method)

    # Cut dendrogram
    if n_clusters is None:
        # Use inconsistency method to auto-determine clusters
        n_clusters = max(2, len(returns.columns) // 5)

    cluster_labels = hierarchy.fcluster(link, n_clusters, criterion='maxclust')

    return pd.Series(cluster_labels, index=returns.columns)
