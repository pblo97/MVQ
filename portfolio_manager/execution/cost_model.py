# portfolio_manager/execution/cost_model.py
"""
Transaction Cost Model: Spread + Market Impact (Almgren-Chriss style)
Estima costos realistas de ejecución para optimizar net returns.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransactionCostResult:
    """Resultado de estimación de costos"""
    symbol: str
    shares: float
    price: float
    notional_usd: float
    spread_cost_usd: float
    impact_cost_usd: float
    total_cost_usd: float
    total_cost_bps: float
    pct_ADV: float

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'shares': round(self.shares, 2),
            'price': round(self.price, 2),
            'notional_usd': round(self.notional_usd, 2),
            'spread_cost_usd': round(self.spread_cost_usd, 2),
            'impact_cost_usd': round(self.impact_cost_usd, 2),
            'total_cost_usd': round(self.total_cost_usd, 2),
            'total_cost_bps': round(self.total_cost_bps, 2),
            'pct_ADV': round(self.pct_ADV * 100, 2)
        }


def _estimate_spread_bps(ADV: float, volatility: float) -> float:
    """
    Estima bid-ask spread en bps usando liquidez y volatilidad.

    Tiered model:
    - ADV > $50M: mega liquid (3-5 bps)
    - ADV > $10M: highly liquid (5-10 bps)
    - ADV > $1M: liquid (10-20 bps)
    - ADV < $1M: illiquid (20-50+ bps)

    Ajusta por volatilidad: vol alta → spread más amplio
    """
    if ADV > 50_000_000:
        base_spread = 4
    elif ADV > 10_000_000:
        base_spread = 7
    elif ADV > 1_000_000:
        base_spread = 15
    elif ADV > 100_000:
        base_spread = 30
    else:
        base_spread = 50

    # Ajuste por volatilidad (vol anualizada)
    vol_factor = 1 + (volatility - 0.20) * 2  # vol 20% = neutral, 40% = 1.4×
    vol_factor = max(0.5, min(vol_factor, 3.0))  # clip a [0.5, 3.0]

    spread_bps = base_spread * vol_factor
    return float(np.clip(spread_bps, 1, 200))  # min 1 bps, max 200 bps


def _estimate_market_impact(
    shares: float,
    ADV: float,
    volatility: float,
    price: float,
    impact_exponent: float = 1.5
) -> float:
    """
    Estima market impact cost usando modelo non-linear (Almgren-Chriss style).

    Formula:
    impact_cost = k × (shares / ADV)^impact_exponent × volatility × notional

    donde:
    - k = 0.1 (constante de ajuste empírica)
    - impact_exponent = 1.5 (no-linearidad: grandes órdenes tienen impacto desproporcionado)
    - volatility: vol anualizada del activo

    Intuición:
    - 5% del ADV → impacto bajo (~5-10 bps)
    - 20% del ADV → impacto significativo (~30-50 bps)
    - 50%+ del ADV → impacto muy alto (100+ bps)
    """
    if ADV <= 0 or shares == 0:
        return 0.0

    pct_ADV = abs(shares) / ADV
    notional = abs(shares) * price

    # Constante de impacto (ajustable por mercado/estilo)
    k = 0.1

    # Impact cost (USD)
    impact_cost = k * (pct_ADV ** impact_exponent) * volatility * notional

    return float(impact_cost)


def estimate_transaction_cost(
    symbol: str,
    shares: float,
    price: float,
    ADV: float,
    volatility: float,
    impact_exponent: float = 1.5
) -> TransactionCostResult:
    """
    Estima costos de transacción para una orden.

    Args:
        symbol: ticker del activo
        shares: número de acciones a transar (positivo = compra, negativo = venta)
        price: precio de ejecución estimado (USD)
        ADV: Average Daily Volume (USD, no shares)
        volatility: volatilidad anualizada (e.g., 0.25 = 25%)
        impact_exponent: exponente de no-linearidad del impacto (default 1.5)

    Returns:
        TransactionCostResult con breakdown de costos
    """
    if shares == 0 or price <= 0:
        return TransactionCostResult(
            symbol=symbol,
            shares=0,
            price=price,
            notional_usd=0,
            spread_cost_usd=0,
            impact_cost_usd=0,
            total_cost_usd=0,
            total_cost_bps=0,
            pct_ADV=0
        )

    notional = abs(shares) * price
    pct_ADV = abs(shares * price) / max(ADV, 1)

    # 1) Spread cost (fijo por cada acción transada)
    spread_bps = _estimate_spread_bps(ADV, volatility)
    spread_cost = (spread_bps / 10_000) * notional

    # 2) Market impact cost (non-linear en tamaño)
    impact_cost = _estimate_market_impact(shares, ADV, volatility, price, impact_exponent)

    # Total cost
    total_cost = spread_cost + impact_cost
    total_cost_bps = (total_cost / notional) * 10_000 if notional > 0 else 0

    return TransactionCostResult(
        symbol=symbol,
        shares=float(shares),
        price=float(price),
        notional_usd=float(notional),
        spread_cost_usd=float(spread_cost),
        impact_cost_usd=float(impact_cost),
        total_cost_usd=float(total_cost),
        total_cost_bps=float(total_cost_bps),
        pct_ADV=float(pct_ADV)
    )


def estimate_portfolio_rebalance_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    ADVs: pd.Series,
    volatilities: pd.Series,
    impact_exponent: float = 1.5
) -> pd.DataFrame:
    """
    Estima costos de rebalanceo completo del portfolio.

    Args:
        current_weights: Series con pesos actuales (symbol → weight)
        target_weights: Series con pesos objetivo (symbol → weight)
        portfolio_value: valor total del portfolio (USD)
        prices: Series con precios actuales (symbol → price)
        ADVs: Series con average daily volumes (symbol → ADV in USD)
        volatilities: Series con volatilidades anualizadas (symbol → vol)
        impact_exponent: exponente de no-linearidad

    Returns:
        DataFrame con breakdown de costos por símbolo + totales
    """
    symbols = list(set(current_weights.index) | set(target_weights.index))

    results = []
    for sym in symbols:
        w_curr = current_weights.get(sym, 0.0)
        w_tgt = target_weights.get(sym, 0.0)
        delta_w = w_tgt - w_curr

        if abs(delta_w) < 1e-6:  # skip si cambio insignificante
            continue

        price = prices.get(sym, np.nan)
        adv = ADVs.get(sym, 1_000_000)  # default conservador
        vol = volatilities.get(sym, 0.25)  # default 25%

        if pd.isna(price) or price <= 0:
            continue

        # Calcula shares a transar
        delta_usd = delta_w * portfolio_value
        shares_to_trade = delta_usd / price

        # Estima costos
        cost_result = estimate_transaction_cost(
            symbol=sym,
            shares=shares_to_trade,
            price=price,
            ADV=adv,
            volatility=vol,
            impact_exponent=impact_exponent
        )

        row = cost_result.to_dict()
        row['delta_weight'] = delta_w
        row['action'] = 'BUY' if shares_to_trade > 0 else 'SELL'
        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Agrega fila de totales
    totals = {
        'symbol': 'TOTAL',
        'shares': df['shares'].sum(),
        'notional_usd': df['notional_usd'].sum(),
        'spread_cost_usd': df['spread_cost_usd'].sum(),
        'impact_cost_usd': df['impact_cost_usd'].sum(),
        'total_cost_usd': df['total_cost_usd'].sum(),
        'total_cost_bps': (df['total_cost_usd'].sum() / df['notional_usd'].sum() * 10_000) if df['notional_usd'].sum() > 0 else 0,
        'pct_ADV': np.nan,
        'action': '—'
    }

    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df
