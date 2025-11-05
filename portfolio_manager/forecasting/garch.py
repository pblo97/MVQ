# portfolio_manager/forecasting/garch.py
"""
GARCH Volatility Forecasting

Generalized AutoRegressive Conditional Heteroskedasticity models for
forecasting time-varying volatility.

Key Innovation:
- Models volatility clustering: high volatility follows high volatility
- Adaptive forecasts: responds to regime changes
- Better than sample variance in crisis periods

Academic References:
- Engle (1982): Autoregressive Conditional Heteroskedasticity (ARCH)
- Bollerslev (1986): Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
- Nelson (1991): Conditional Heteroskedasticity in Asset Returns: A New Approach (EGARCH)
- Engle (2001): GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics

Model:
GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Where:
- σ²_t: Conditional variance at time t
- ε²_{t-1}: Squared residual (shock) at t-1
- ω: Constant term (> 0)
- α: ARCH coefficient (≥ 0) - reaction to shocks
- β: GARCH coefficient (≥ 0) - persistence
- α + β < 1 for stationarity

Author: Portfolio Optimization Team
Date: 2025-11
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings


class GARCHVolatilityForecaster:
    """
    GARCH volatility forecasting for portfolio optimization

    Uses arch library (Kevin Sheppard) for GARCH estimation.

    Parameters:
    -----------
    p : int
        ARCH order (default 1)
    q : int
        GARCH order (default 1)
    model_type : str
        'GARCH' (default) or 'EGARCH' (asymmetric)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model_type: str = 'GARCH'
    ):
        self.p = p
        self.q = q
        self.model_type = model_type

        # Model and results
        self.model = None
        self.fitted_model = None
        self.model_params: Optional[Dict] = None

    def fit(
        self,
        returns: pd.Series,
        rescale: bool = True
    ) -> 'GARCHVolatilityForecaster':
        """
        Fit GARCH model to return series

        Parameters:
        -----------
        returns : pd.Series
            Return series (should be demeaned or have mean specified)
        rescale : bool
            If True, rescales returns by 100 for numerical stability

        Returns:
        --------
        self : GARCHVolatilityForecaster
            Fitted forecaster
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch library required. Install with: pip install arch")

        # Clean data
        returns_clean = returns.dropna()

        if len(returns_clean) < 100:
            raise ValueError(f"Insufficient data for GARCH: {len(returns_clean)} obs (need ≥100)")

        # Rescale for numerical stability (optional but recommended)
        if rescale:
            returns_clean = returns_clean * 100

        # Fit GARCH model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = arch_model(
                returns_clean,
                vol=self.model_type,
                p=self.p,
                q=self.q,
                rescale=False  # We already rescaled manually
            )

            self.fitted_model = self.model.fit(disp='off', show_warning=False)

        # Store parameters
        params = self.fitted_model.params
        self.model_params = {
            'omega': params.get('omega', np.nan),
            'alpha[1]': params.get('alpha[1]', np.nan),
            'beta[1]': params.get('beta[1]', np.nan)
        }

        # Calculate persistence (α + β)
        alpha = self.model_params['alpha[1]']
        beta = self.model_params['beta[1]']
        self.model_params['persistence'] = alpha + beta

        return self

    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> pd.Series:
        """
        Forecast volatility for next periods

        Parameters:
        -----------
        horizon : int
            Forecast horizon (days ahead)
        method : str
            'analytic' (closed-form) or 'simulation' (Monte Carlo)

        Returns:
        --------
        forecast : pd.Series
            Forecasted volatilities (annualized if returns were daily)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate forecast
        forecast_obj = self.fitted_model.forecast(horizon=horizon, method=method)

        # Extract variance forecast
        variance_forecast = forecast_obj.variance.iloc[-1]

        # Convert to volatility (std dev)
        volatility_forecast = np.sqrt(variance_forecast)

        return volatility_forecast

    def conditional_volatility(self) -> pd.Series:
        """
        Get in-sample conditional volatility (fitted values)

        Returns:
        --------
        cond_vol : pd.Series
            Conditional volatility series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.conditional_volatility

    def get_params(self) -> Dict[str, float]:
        """
        Get estimated GARCH parameters

        Returns:
        --------
        params : dict
            {'omega', 'alpha[1]', 'beta[1]', 'persistence'}
        """
        if self.model_params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_params

    def diagnostics(self) -> Dict[str, any]:
        """
        Get model diagnostics and quality metrics

        Returns:
        --------
        diagnostics : dict
            {
                'params': GARCH parameters,
                'loglikelihood': Log-likelihood,
                'AIC': Akaike Information Criterion,
                'BIC': Bayesian Information Criterion,
                'persistence': α + β,
                'stationarity': Whether α + β < 1
            }
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        persistence = self.model_params['persistence']

        return {
            'params': self.model_params,
            'loglikelihood': self.fitted_model.loglikelihood,
            'AIC': self.fitted_model.aic,
            'BIC': self.fitted_model.bic,
            'persistence': persistence,
            'is_stationary': persistence < 1.0,
            'mean': self.fitted_model.params.get('mu', 0.0),
            'n_obs': self.fitted_model.nobs
        }


def forecast_portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    horizon: int = 1,
    method: str = 'sample'
) -> float:
    """
    Forecast portfolio volatility using GARCH or sample variance

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns (assets × time)
    weights : pd.Series, optional
        Portfolio weights. If None, uses equal weight.
    horizon : int
        Forecast horizon (days ahead)
    method : str
        'sample' (sample variance) or 'garch' (GARCH forecast)

    Returns:
    --------
    portfolio_vol : float
        Forecasted portfolio volatility (annualized)
    """
    if weights is None:
        weights = pd.Series(1 / len(returns_df.columns), index=returns_df.columns)

    # Align
    common_assets = returns_df.columns.intersection(weights.index)
    returns_aligned = returns_df[common_assets]
    weights_aligned = weights.loc[common_assets]
    weights_aligned = weights_aligned / weights_aligned.sum()

    if method == 'sample':
        # Sample covariance
        cov = returns_aligned.cov()
        portfolio_variance = weights_aligned @ cov @ weights_aligned
        portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualize

    elif method == 'garch':
        # GARCH forecast for each asset, then combine
        asset_vols = {}

        for asset in common_assets:
            try:
                forecaster = GARCHVolatilityForecaster(p=1, q=1)
                forecaster.fit(returns_aligned[asset], rescale=True)

                # Forecast variance
                vol_forecast = forecaster.forecast(horizon=horizon)
                asset_vols[asset] = vol_forecast.iloc[0] / 100  # Unscale

            except Exception as e:
                # Fallback to sample std
                asset_vols[asset] = returns_aligned[asset].std()

        # Portfolio volatility with GARCH forecasts
        # Simplified: use sample correlation, GARCH for individual vols
        corr = returns_aligned.corr()

        # Build covariance from GARCH vols + sample correlation
        vols_series = pd.Series(asset_vols)
        cov_garch = corr * np.outer(vols_series, vols_series)

        portfolio_variance = weights_aligned @ cov_garch @ weights_aligned
        portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualize

    else:
        raise ValueError(f"Unknown method: {method}. Use 'sample' or 'garch'.")

    return float(portfolio_vol)


def compare_sample_vs_garch(
    returns: pd.Series,
    rolling_window: int = 252
) -> pd.DataFrame:
    """
    Compare sample volatility vs GARCH forecast

    Useful for visualizing GARCH adaptation to volatility regimes.

    Parameters:
    -----------
    returns : pd.Series
        Return series
    rolling_window : int
        Window for sample volatility

    Returns:
    --------
    comparison : pd.DataFrame
        Columns: ['sample_vol', 'garch_vol']
    """
    # Sample volatility (rolling)
    sample_vol = returns.rolling(rolling_window).std() * np.sqrt(252)

    # GARCH volatility (conditional)
    try:
        forecaster = GARCHVolatilityForecaster(p=1, q=1)
        forecaster.fit(returns, rescale=True)
        garch_cond_vol = forecaster.conditional_volatility() / 100 * np.sqrt(252)  # Annualize

        comparison = pd.DataFrame({
            'sample_vol': sample_vol,
            'garch_vol': garch_cond_vol
        })

    except Exception as e:
        # Fallback: return only sample vol
        comparison = pd.DataFrame({
            'sample_vol': sample_vol,
            'garch_vol': np.nan
        })

    return comparison


def garch_with_kelly(
    returns_df: pd.DataFrame,
    base_kelly: float = 0.25,
    use_garch: bool = True
) -> pd.Series:
    """
    Calculate Kelly weights using GARCH volatility forecasts

    Replaces sample variance with GARCH forecast in Kelly formula.

    Kelly formula (single-asset): f* = μ / σ²

    With GARCH: f* = μ / σ²_GARCH

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    base_kelly : float
        Kelly fraction multiplier
    use_garch : bool
        If True, uses GARCH forecast. If False, uses sample variance.

    Returns:
    --------
    weights : pd.Series
        Kelly weights
    """
    weights = {}

    for asset in returns_df.columns:
        returns_asset = returns_df[asset].dropna()

        # Expected return (sample mean)
        mu = returns_asset.mean()

        if use_garch:
            try:
                # GARCH variance forecast
                forecaster = GARCHVolatilityForecaster(p=1, q=1)
                forecaster.fit(returns_asset, rescale=True)
                vol_forecast = forecaster.forecast(horizon=1).iloc[0] / 100
                variance = vol_forecast ** 2

            except Exception:
                # Fallback to sample variance
                variance = returns_asset.var()
        else:
            # Sample variance
            variance = returns_asset.var()

        # Kelly weight
        if variance > 0 and mu > 0:
            f = base_kelly * (mu / variance)
            weights[asset] = max(0, f)  # Long-only
        else:
            weights[asset] = 0.0

    weights_series = pd.Series(weights)

    # Normalize to sum = 1
    if weights_series.sum() > 0:
        weights_series = weights_series / weights_series.sum()
    else:
        # Equal weight fallback
        weights_series = pd.Series(1 / len(weights_series), index=weights_series.index)

    return weights_series
