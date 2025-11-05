# portfolio_manager/data/orchestrator_enhanced.py
"""
Enhanced Portfolio Orchestrator con Quality Caps dinámicos.
Extiende qvm_trend.pm.orchestrator con:
- Quality Score 3D → position caps individuales
- λ_quality penalty (activos de baja calidad reciben menor peso)
- Integration con transaction cost model
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

# Import orchestrator original
from qvm_trend.pm.orchestrator import build_portfolio as build_portfolio_base
from qvm_trend.data_io import load_prices_panel
from qvm_trend.macro.macro_score import z_to_regime

# Import nuevos módulos
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from portfolio_manager.quality.composite import compute_quality_batch, QualityResult
from portfolio_manager.execution.cost_model import estimate_transaction_cost


def build_portfolio_with_quality_caps(
    symbols: List[str],
    bench: str,
    start: str,
    end: str,
    *,
    # Kelly pro (pasa al base)
    base_kelly: float = 0.25,
    winsor_p: float = 0.01,
    min_months: int = 36,
    costs_per_period: float = 0.0005,
    shrink_kappa: int = 12,
    ewm_span: int = 14,
    lambda_corr: float = 0.25,
    # Macro
    macro_z: float = 0.0,
    beta_cap_user: float = 1.2,
    allow_new_when_z_below: float = -1.0,
    current_holdings: Optional[List[str]] = None,
    # Quality 3D (NUEVO)
    use_quality_caps: bool = True,
    quality_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),  # liq, fund, tech
    lambda_quality_range: tuple[float, float] = (0.3, 1.0),  # penalty range por quality
    fundamentals_df: Optional[pd.DataFrame] = None,
    # Compatibility
    quality_df: Optional[pd.DataFrame] = None,  # legacy
    alpha_off: float = 0.40,
    alpha_neu: float = 0.25,
    alpha_on: float = 0.15,
    enforce_sum1: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced portfolio builder con quality caps dinámicos.

    Returns:
        (portfolio_df, quality_df)
        - portfolio_df: DataFrame con pesos y métricas Kelly
        - quality_df: DataFrame con quality scores 3D
    """

    # 1) Cargar precios para quality score
    price_panel = load_prices_panel(symbols + [bench], start, end, cache_key="pm_quality_panel")
    benchmark_df = price_panel.get(bench)

    # 2) Calcular Quality Score 3D
    if use_quality_caps:
        quality_scores_df = compute_quality_batch(
            symbols=symbols,
            price_panel=price_panel,
            fundamentals_df=fundamentals_df,
            benchmark_df=benchmark_df,
            weights=quality_weights
        )
    else:
        # Usa quality_df legacy si se provee
        quality_scores_df = quality_df if quality_df is not None else pd.DataFrame()

    # 3) Construye portfolio base usando orchestrator original
    # Pasa quality_scores_df como quality_df para quality tilt
    portfolio_df = build_portfolio_base(
        symbols=symbols,
        bench=bench,
        start=start,
        end=end,
        base_kelly=base_kelly,
        winsor_p=winsor_p,
        min_months=min_months,
        costs_per_period=costs_per_period,
        shrink_kappa=shrink_kappa,
        ewm_span=ewm_span,
        lambda_corr=lambda_corr,
        macro_z=macro_z,
        quality_df=quality_scores_df if not quality_scores_df.empty else quality_df,
        alpha_off=alpha_off,
        alpha_neu=alpha_neu,
        alpha_on=alpha_on,
        enforce_sum1=False,  # aplicaremos caps custom después
        pos_cap=1.0,  # sin cap global, usaremos caps individuales
        beta_cap_user=beta_cap_user,
        allow_new_when_z_below=allow_new_when_z_below,
        current_holdings=current_holdings
    )

    if portfolio_df.empty:
        return portfolio_df, quality_scores_df

    # 4) Aplica Quality Caps dinámicos
    if use_quality_caps and not quality_scores_df.empty:
        # Merge quality caps
        quality_map = dict(zip(
            quality_scores_df['symbol'].str.upper(),
            quality_scores_df['position_cap']
        ))
        portfolio_df['quality_cap'] = portfolio_df['symbol'].str.upper().map(quality_map).fillna(0.05)

        # Aplica λ_quality penalty (reduce peso según quality score)
        quality_score_map = dict(zip(
            quality_scores_df['symbol'].str.upper(),
            quality_scores_df['quality_score']
        ))
        portfolio_df['quality_score'] = portfolio_df['symbol'].str.upper().map(quality_score_map).fillna(50.0)

        # λ_quality: mapea quality_score 0-100 → λ_quality_range
        # quality 100 → λ = 1.0 (sin penalty)
        # quality 0 → λ = 0.3 (fuerte penalty)
        lambda_min, lambda_max = lambda_quality_range
        portfolio_df['lambda_quality'] = (
            lambda_min + (lambda_max - lambda_min) * (portfolio_df['quality_score'] / 100.0)
        )

        # Aplica penalty a pesos base
        portfolio_df['weight_pre_cap'] = portfolio_df['weight'].copy()
        portfolio_df['weight'] = portfolio_df['weight'] * portfolio_df['lambda_quality']

        # Clip individual caps
        portfolio_df['weight'] = np.minimum(
            portfolio_df['weight'],
            portfolio_df['quality_cap']
        )

        # Re-normalize si enforce_sum1
        if enforce_sum1:
            total = portfolio_df['weight'].sum()
            if total > 0:
                portfolio_df['weight'] = portfolio_df['weight'] / total

        # Aplica beta cap (régimen macro)
        reg = z_to_regime(macro_z)
        beta_cap_eff = min(beta_cap_user, reg.beta_cap)

        betas = portfolio_df['beta'].fillna(1.0).values
        weights = portfolio_df['weight'].values
        beta_total = float((betas * weights).sum())

        if beta_total > beta_cap_eff and beta_total > 0:
            scale_factor = beta_cap_eff / beta_total
            portfolio_df['weight'] = portfolio_df['weight'] * scale_factor

        # Re-calc beta_w
        portfolio_df['beta_w'] = portfolio_df['beta'].fillna(1.0) * portfolio_df['weight']

    else:
        # Sin quality caps, agrega columnas vacías para consistencia
        portfolio_df['quality_cap'] = np.nan
        portfolio_df['quality_score'] = np.nan
        portfolio_df['lambda_quality'] = 1.0

    # 5) Sort y return
    portfolio_df = portfolio_df.sort_values('weight', ascending=False).reset_index(drop=True)

    return portfolio_df, quality_scores_df


def estimate_rebalance_costs(
    current_portfolio: pd.DataFrame,
    target_portfolio: pd.DataFrame,
    portfolio_value: float,
    price_panel: Dict[str, pd.DataFrame],
    quality_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Estima costos de rebalanceo usando transaction cost model.

    Args:
        current_portfolio: DataFrame con pesos actuales
        target_portfolio: DataFrame con pesos objetivo (de build_portfolio_with_quality_caps)
        portfolio_value: valor total del portfolio (USD)
        price_panel: Dict con DataFrames de precios
        quality_df: DataFrame con quality scores (para ADV, volatility)

    Returns:
        DataFrame con breakdown de costos por trade
    """
    from portfolio_manager.execution.cost_model import estimate_portfolio_rebalance_cost

    # Extract current & target weights
    curr_w = pd.Series(0.0, index=target_portfolio['symbol'])
    if not current_portfolio.empty and 'weight' in current_portfolio.columns:
        curr_w_temp = current_portfolio.set_index('symbol')['weight']
        curr_w.update(curr_w_temp)

    tgt_w = target_portfolio.set_index('symbol')['weight']

    # Extract prices (latest close)
    prices = pd.Series(dtype=float)
    for sym in tgt_w.index:
        if sym in price_panel and 'close' in price_panel[sym].columns:
            close = pd.to_numeric(price_panel[sym]['close'], errors='coerce').dropna()
            if not close.empty:
                prices[sym] = close.iloc[-1]

    # Extract ADV & volatilities from quality_df
    ADVs = pd.Series(1_000_000, index=tgt_w.index)  # default conservador
    vols = pd.Series(0.25, index=tgt_w.index)  # default 25%

    if not quality_df.empty:
        adv_map = dict(zip(quality_df['symbol'], quality_df['ADV']))
        ADVs.update(pd.Series(adv_map))

        # Vol estimate (desde price panel)
        for sym in tgt_w.index:
            if sym in price_panel and 'close' in price_panel[sym].columns:
                close = pd.to_numeric(price_panel[sym]['close'], errors='coerce').dropna()
                if len(close) > 60:
                    ret = close.pct_change().dropna().tail(60)
                    vol_ann = ret.std() * np.sqrt(252)
                    vols[sym] = vol_ann

    # Calcula costos
    cost_df = estimate_portfolio_rebalance_cost(
        current_weights=curr_w,
        target_weights=tgt_w,
        portfolio_value=portfolio_value,
        prices=prices,
        ADVs=ADVs,
        volatilities=vols,
        impact_exponent=1.5
    )

    return cost_df
