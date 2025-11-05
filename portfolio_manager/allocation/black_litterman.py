# portfolio_manager/allocation/black_litterman.py
"""
Black-Litterman Portfolio Optimization

Bayesian approach combining market equilibrium with investor views.
Developed by Fischer Black and Robert Litterman at Goldman Sachs (1992).

Key Innovation:
- Market cap weights as prior (CAPM equilibrium)
- Investor views as Bayesian update
- Posterior expected returns more stable than sample mean

Academic References:
- Black & Litterman (1992): Global Portfolio Optimization
- He & Litterman (1999): The Intuition Behind Black-Litterman
- Idzorek (2005): A step-by-step guide to the Black-Litterman model

Author: Portfolio Optimization Team
Date: 2025-11
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization with investor views

    Mathematical Framework:
    1. Implied Equilibrium Returns (Reverse Optimization):
       π = λ × Σ × w_mkt
       where λ = risk aversion coefficient (default 2.5)

    2. Bayesian Update (Incorporating Views):
       Posterior returns: μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1π + P'Ω^-1Q]
       Posterior covariance: Σ_BL = Σ + [(τΣ)^-1 + P'Ω^-1P]^-1

    3. Optimization:
       Maximize: w'μ_BL - (λ/2)w'Σ_BL w
       Subject to: Σw = 1, w ≥ 0 (long-only)

    Parameters:
    -----------
    risk_aversion : float
        Risk aversion coefficient λ (default 2.5, market standard)
    tau : float
        Uncertainty in prior (default 0.025, He & Litterman 1999)
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.025
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Stored data
        self.market_cap_weights: Optional[pd.Series] = None
        self.equilibrium_returns: Optional[pd.Series] = None
        self.posterior_returns: Optional[pd.Series] = None
        self.posterior_cov: Optional[pd.DataFrame] = None

    def implied_equilibrium_returns(
        self,
        cov: pd.DataFrame,
        market_weights: pd.Series
    ) -> pd.Series:
        """
        Calculate implied equilibrium returns via reverse optimization

        Formula: π = λ × Σ × w_mkt

        Interpretation: Market cap weights imply expected returns that justify
        those weights under CAPM equilibrium.

        Parameters:
        -----------
        cov : pd.DataFrame
            Covariance matrix of returns
        market_weights : pd.Series
            Market capitalization weights (normalized to sum=1)

        Returns:
        --------
        pi : pd.Series
            Implied equilibrium returns (annualized)
        """
        # Align
        common_assets = cov.index.intersection(market_weights.index)
        cov_aligned = cov.loc[common_assets, common_assets]
        weights_aligned = market_weights.loc[common_assets]

        # Normalize weights
        weights_aligned = weights_aligned / weights_aligned.sum()

        # π = λ × Σ × w_mkt
        pi = self.risk_aversion * (cov_aligned @ weights_aligned)

        self.equilibrium_returns = pi
        self.market_cap_weights = weights_aligned

        return pi

    def create_view_matrix(
        self,
        view_dict: Dict[str, Tuple[List[str], List[float], float]],
        assets: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create P, Q, Ω matrices from investor views

        View Types:
        1. Absolute: "Asset A will return 5%"
           → P = [1, 0, 0, ...], Q = [0.05]

        2. Relative: "Asset A will outperform Asset B by 3%"
           → P = [1, -1, 0, ...], Q = [0.03]

        Parameters:
        -----------
        view_dict : dict
            Views in format:
            {
                'view1': ([assets], [weights], expected_return),
                'view2': ([assets], [weights], expected_return),
                ...
            }
            Example:
            {
                'AAPL_outperform': (['AAPL', 'SPY'], [1, -1], 0.05),  # AAPL beats SPY by 5%
                'MSFT_absolute': (['MSFT'], [1], 0.10)  # MSFT returns 10%
            }

        assets : list
            List of all asset names (for ordering)

        confidence : float
            Confidence level (0-1). Higher = tighter view
            Used to construct Ω: lower confidence → higher variance

        Returns:
        --------
        P : np.ndarray (k × n)
            Pick matrix (k views, n assets)
        Q : np.ndarray (k × 1)
            View returns
        Omega : np.ndarray (k × k)
            View uncertainty covariance matrix
        """
        n_assets = len(assets)
        n_views = len(view_dict)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))

        asset_to_idx = {asset: i for i, asset in enumerate(assets)}

        for view_idx, (view_name, (view_assets, view_weights, view_return)) in enumerate(view_dict.items()):
            # Fill P matrix
            for asset, weight in zip(view_assets, view_weights):
                if asset in asset_to_idx:
                    P[view_idx, asset_to_idx[asset]] = weight

            # Fill Q vector
            Q[view_idx] = view_return

        # Omega: uncertainty in views
        # Idzorek (2005): Ω = τ × P × Σ × P'
        # We'll compute this after getting Σ

        return P, Q, Omega

    def calculate_omega(
        self,
        P: np.ndarray,
        cov: pd.DataFrame,
        confidence_levels: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Calculate view uncertainty matrix Ω

        Two approaches:
        1. Proportional to variance (Idzorek 2005):
           Ω = τ × P × Σ × P'

        2. Confidence-based (He & Litterman 1999):
           Ω_ii = (1/confidence_i) × τ × (P × Σ × P')_ii

        Parameters:
        -----------
        P : np.ndarray
            Pick matrix
        cov : pd.DataFrame
            Covariance matrix
        confidence_levels : list of float, optional
            Confidence per view (0-1). If None, uses proportional method.

        Returns:
        --------
        Omega : np.ndarray
            View uncertainty covariance matrix
        """
        n_views = P.shape[0]
        cov_np = cov.values

        # Base: τ × P × Σ × P'
        base_omega = self.tau * (P @ cov_np @ P.T)

        if confidence_levels is None:
            # Proportional method (Idzorek 2005)
            return base_omega
        else:
            # Confidence-based (He & Litterman 1999)
            # Higher confidence → lower uncertainty
            Omega = np.diag(np.diag(base_omega))
            for i, conf in enumerate(confidence_levels):
                if conf > 0:
                    Omega[i, i] = Omega[i, i] / conf
            return Omega

    def posterior_distribution(
        self,
        cov: pd.DataFrame,
        pi: pd.Series,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute posterior (Black-Litterman) returns and covariance

        Formulas (He & Litterman 1999):

        μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1π + P'Ω^-1Q]

        Σ_BL = Σ + [(τΣ)^-1 + P'Ω^-1P]^-1

        Parameters:
        -----------
        cov : pd.DataFrame
            Prior covariance matrix
        pi : pd.Series
            Prior (equilibrium) expected returns
        P : np.ndarray
            Pick matrix
        Q : np.ndarray
            View returns
        Omega : np.ndarray
            View uncertainty

        Returns:
        --------
        mu_bl : pd.Series
            Posterior expected returns
        sigma_bl : pd.DataFrame
            Posterior covariance matrix
        """
        # Convert to numpy
        assets = cov.index.tolist()
        Sigma = cov.values
        pi_vec = pi.values.reshape(-1, 1)
        Q_vec = Q.reshape(-1, 1)

        # Inverse of prior covariance (scaled by τ)
        tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)

        # Inverse of view uncertainty
        Omega_inv = np.linalg.inv(Omega)

        # Posterior precision matrix
        M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
        M = np.linalg.inv(M_inv)

        # Posterior expected returns
        mu_bl_vec = M @ (tau_Sigma_inv @ pi_vec + P.T @ Omega_inv @ Q_vec)
        mu_bl = pd.Series(mu_bl_vec.flatten(), index=assets)

        # Posterior covariance
        Sigma_bl = Sigma + M
        sigma_bl = pd.DataFrame(Sigma_bl, index=assets, columns=assets)

        self.posterior_returns = mu_bl
        self.posterior_cov = sigma_bl

        return mu_bl, sigma_bl

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        cov: pd.DataFrame,
        target_return: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Optimize portfolio weights using Black-Litterman returns

        Objective: Maximize Sharpe ratio (or target return with min variance)

        w* = argmax w'μ - (λ/2)w'Σw
        subject to: Σw = 1, w ≥ 0

        Parameters:
        -----------
        expected_returns : pd.Series
            Expected returns (posterior)
        cov : pd.DataFrame
            Covariance matrix (posterior)
        target_return : float, optional
            If provided, minimizes variance subject to target return
        constraints : dict, optional
            Additional constraints (min/max weights)

        Returns:
        --------
        weights : pd.Series
            Optimal portfolio weights
        """
        n = len(expected_returns)
        mu = expected_returns.values
        Sigma = cov.values

        # Objective: -Sharpe ratio (minimize negative)
        def objective(w):
            port_return = w @ mu
            port_variance = w @ Sigma @ w
            # Sharpe = return / std = return / sqrt(variance)
            sharpe = port_return / np.sqrt(port_variance)
            return -sharpe  # Minimize negative = maximize Sharpe

        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1

        if target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda w: w @ mu - target_return})

        # Bounds: long-only
        bounds = [(0, 1) for _ in range(n)]

        if constraints is not None:
            if 'min_weights' in constraints:
                min_w = constraints['min_weights']
                bounds = [(min_w.get(asset, 0), 1) for asset in expected_returns.index]
            if 'max_weights' in constraints:
                max_w = constraints['max_weights']
                bounds = [(bounds[i][0], max_w.get(asset, 1))
                         for i, asset in enumerate(expected_returns.index)]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        weights = pd.Series(result.x, index=expected_returns.index)

        return weights

    def compare_returns(
        self,
        sample_returns: pd.Series,
        equilibrium_returns: pd.Series,
        posterior_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Compare sample mean vs equilibrium vs Black-Litterman returns

        Parameters:
        -----------
        sample_returns : pd.Series
            Historical sample mean returns
        equilibrium_returns : pd.Series
            Implied equilibrium returns (π)
        posterior_returns : pd.Series
            Black-Litterman posterior returns (μ_BL)

        Returns:
        --------
        comparison : pd.DataFrame
            Comparison table with differences
        """
        comparison = pd.DataFrame({
            'Sample Mean': sample_returns,
            'Equilibrium (π)': equilibrium_returns,
            'BL Posterior (μ_BL)': posterior_returns
        })

        comparison['Sample - Equilibrium'] = comparison['Sample Mean'] - comparison['Equilibrium (π)']
        comparison['BL - Equilibrium'] = comparison['BL Posterior (μ_BL)'] - comparison['Equilibrium (π)']

        return comparison


def run_black_litterman(
    returns_df: pd.DataFrame,
    market_cap_weights: pd.Series,
    views: Dict[str, Tuple[List[str], List[float], float]],
    confidence_levels: Optional[List[float]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.025
) -> Dict:
    """
    Complete Black-Litterman workflow

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Historical returns (assets × time)
    market_cap_weights : pd.Series
        Market capitalization weights
    views : dict
        Investor views (see create_view_matrix for format)
    confidence_levels : list of float, optional
        Confidence per view (0-1)
    risk_aversion : float
        Risk aversion coefficient (default 2.5)
    tau : float
        Prior uncertainty (default 0.025)

    Returns:
    --------
    results : dict
        {
            'bl_weights': Optimal BL weights,
            'equilibrium_returns': π,
            'posterior_returns': μ_BL,
            'comparison': Comparison table,
            'optimizer': BlackLittermanOptimizer instance
        }
    """
    bl = BlackLittermanOptimizer(risk_aversion=risk_aversion, tau=tau)

    # 1. Calculate covariance and sample mean
    cov = returns_df.cov()
    sample_mean = returns_df.mean()

    # 2. Implied equilibrium returns
    pi = bl.implied_equilibrium_returns(cov, market_cap_weights)

    # 3. Create view matrices
    assets = list(returns_df.columns)
    P, Q, _ = bl.create_view_matrix(views, assets)
    Omega = bl.calculate_omega(P, cov, confidence_levels)

    # 4. Posterior distribution
    mu_bl, sigma_bl = bl.posterior_distribution(cov, pi, P, Q, Omega)

    # 5. Optimize portfolio
    bl_weights = bl.optimize_portfolio(mu_bl, sigma_bl)

    # 6. Comparison
    comparison = bl.compare_returns(sample_mean, pi, mu_bl)

    return {
        'bl_weights': bl_weights,
        'equilibrium_returns': pi,
        'posterior_returns': mu_bl,
        'posterior_cov': sigma_bl,
        'comparison': comparison,
        'optimizer': bl
    }
