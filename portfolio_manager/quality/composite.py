# portfolio_manager/quality/composite.py
"""
Quality Score 3D: Liquidity + Fundamental + Technical
Genera un score 0-100 por activo y sugiere position caps dinámicos.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class QualityResult:
    """Resultado de quality score por activo"""
    symbol: str
    quality_score: float      # 0-100 composite
    liq_score: float          # 0-100 liquidity
    fund_score: float         # 0-100 fundamental
    tech_score: float         # 0-100 technical
    position_cap: float       # cap sugerido (0.02-0.10)
    ADV: float                # average daily volume (USD)
    spread_bps: float         # bid-ask spread estimate
    days_to_liquidate: float  # posición típica / ADV

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quality_score': round(self.quality_score, 2),
            'liq_score': round(self.liq_score, 2),
            'fund_score': round(self.fund_score, 2),
            'tech_score': round(self.tech_score, 2),
            'position_cap': round(self.position_cap, 4),
            'ADV': round(self.ADV, 2),
            'spread_bps': round(self.spread_bps, 2),
            'days_to_liquidate': round(self.days_to_liquidate, 2)
        }


def _percentile_score(value: float, universe_values: pd.Series, higher_is_better: bool = True) -> float:
    """
    Convierte un valor a percentile score 0-100.
    higher_is_better: True si mayor valor = mejor (e.g., ADV)
                      False si menor valor = mejor (e.g., spread)
    """
    if pd.isna(value) or universe_values.empty:
        return 50.0  # neutral si no hay datos

    pct = (universe_values < value).sum() / len(universe_values) * 100
    return pct if higher_is_better else (100 - pct)


def _liquidity_score(
    symbol: str,
    price_df: pd.DataFrame,
    universe_stats: Optional[Dict] = None
) -> tuple[float, dict]:
    """
    Calcula liquidity score 0-100.

    Componentes:
    - ADV (average daily volume USD)
    - Bid-ask spread estimate (vol como proxy)
    - Volume stability (CV del volumen)
    - Market cap proxy (precio × volumen promedio)
    """
    if price_df is None or price_df.empty or 'close' not in price_df.columns:
        return 50.0, {}

    # Datos de precios y volumen
    close = pd.to_numeric(price_df['close'], errors='coerce').dropna()
    volume = pd.to_numeric(price_df.get('volume', pd.Series()), errors='coerce').dropna()

    if close.empty:
        return 50.0, {}

    # ADV (últimos 60 días)
    if not volume.empty:
        vol_60d = volume.tail(60)
        price_60d = close.reindex(vol_60d.index).ffill()
        adv_usd = (vol_60d * price_60d).mean()

        # Volume stability (CV = std/mean, menor es mejor)
        vol_cv = vol_60d.std() / vol_60d.mean() if vol_60d.mean() > 0 else 2.0
    else:
        # Sin datos de volumen, asumimos liquidez baja
        adv_usd = 100_000  # placeholder conservador
        vol_cv = 1.0

    # Spread estimate (usando volatility como proxy: vol alta = spread alto)
    ret_60d = close.pct_change().tail(60).dropna()
    volatility = ret_60d.std() if len(ret_60d) > 20 else 0.02
    spread_bps = min(100, volatility * 10000 / np.sqrt(252))  # anualizada a bps diario

    # Days to liquidate (para una posición "típica" de $100k)
    typical_position_usd = 100_000
    days_to_liq = typical_position_usd / max(adv_usd, 1000) if adv_usd > 0 else 10

    # Scores individuales
    if universe_stats:
        adv_score = _percentile_score(adv_usd, pd.Series(universe_stats.get('ADV', [adv_usd])), higher_is_better=True)
        spread_score = _percentile_score(spread_bps, pd.Series(universe_stats.get('spread_bps', [spread_bps])), higher_is_better=False)
        cv_score = _percentile_score(vol_cv, pd.Series(universe_stats.get('vol_cv', [vol_cv])), higher_is_better=False)
    else:
        # Heurísticas sin universo
        adv_score = min(100, (adv_usd / 10_000_000) * 100)  # 10M = 100 pts
        spread_score = max(0, 100 - spread_bps * 2)  # 50 bps = 0 pts
        cv_score = max(0, 100 - vol_cv * 50)  # CV=2 → 0 pts

    # Composite liquidity (weighted)
    liq_score = (
        0.50 * adv_score +
        0.30 * spread_score +
        0.20 * cv_score
    )

    metadata = {
        'ADV': adv_usd,
        'spread_bps': spread_bps,
        'vol_cv': vol_cv,
        'days_to_liquidate': days_to_liq
    }

    return float(np.clip(liq_score, 0, 100)), metadata


def _fundamental_score(
    symbol: str,
    fundamentals_df: Optional[pd.DataFrame] = None,
    universe_stats: Optional[Dict] = None
) -> float:
    """
    Calcula fundamental score 0-100 desde datos FMP.

    Componentes:
    - Market cap tier (large > mid > small)
    - Debt/Equity ratio (bajo es mejor)
    - ROE consistency (alto y estable es mejor)
    - Sector defensiveness (utilities, healthcare > tech, energy)
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return 50.0  # neutral si no hay datos

    # Buscar símbolo (case-insensitive)
    mask = fundamentals_df.get('symbol', pd.Series(dtype=str)).str.upper() == symbol.upper()
    if not mask.any():
        return 50.0

    row = fundamentals_df.loc[mask].iloc[0]

    # Market Cap (mayor = mejor liquidez/estabilidad)
    mcap = pd.to_numeric(row.get('marketCap', np.nan), errors='coerce')
    if pd.notna(mcap):
        if mcap > 100_000_000_000:  # >100B = mega cap
            mcap_score = 100
        elif mcap > 10_000_000_000:  # >10B = large cap
            mcap_score = 80
        elif mcap > 2_000_000_000:  # >2B = mid cap
            mcap_score = 60
        else:
            mcap_score = 40  # small cap
    else:
        mcap_score = 50

    # Debt/Equity (bajo es mejor)
    de_ratio = pd.to_numeric(row.get('debtToEquity', np.nan), errors='coerce')
    if pd.notna(de_ratio):
        de_score = max(0, 100 - de_ratio * 20)  # D/E=5 → 0 pts
    else:
        de_score = 50

    # ROE (alto es mejor, pero castigamos extremos)
    roe = pd.to_numeric(row.get('returnOnEquity', np.nan), errors='coerce')
    if pd.notna(roe):
        roe_pct = roe * 100 if roe < 1 else roe  # normaliza
        roe_score = min(100, max(0, roe_pct * 5))  # 20% ROE = 100 pts
    else:
        roe_score = 50

    # Sector defensiveness (si está disponible)
    sector = str(row.get('sector', '')).lower()
    sector_map = {
        'healthcare': 90,
        'utilities': 85,
        'consumer staples': 80,
        'consumer defensive': 80,
        'financials': 70,
        'industrials': 65,
        'technology': 60,
        'communication services': 60,
        'consumer cyclical': 55,
        'real estate': 55,
        'materials': 50,
        'energy': 45
    }
    sector_score = 60  # default neutral
    for key, val in sector_map.items():
        if key in sector:
            sector_score = val
            break

    # Composite fundamental
    fund_score = (
        0.30 * mcap_score +
        0.25 * de_score +
        0.25 * roe_score +
        0.20 * sector_score
    )

    return float(np.clip(fund_score, 0, 100))


def _technical_score(
    symbol: str,
    price_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    universe_stats: Optional[Dict] = None
) -> float:
    """
    Calcula technical score 0-100.

    Componentes:
    - Volatility regime (baja vs histórica = mejor)
    - Correlation stability (baja = diversificación)
    - Drawdown depth (menor = resiliencia)
    - Momentum consistency (menos whipsaws = mejor)
    """
    if price_df is None or price_df.empty or 'close' not in price_df.columns:
        return 50.0

    close = pd.to_numeric(price_df['close'], errors='coerce').dropna()
    if close.empty or len(close) < 60:
        return 50.0

    ret = close.pct_change().dropna()

    # 1) Volatility regime (últimos 60d vs 252d)
    vol_60d = ret.tail(60).std() * np.sqrt(252)
    vol_252d = ret.tail(252).std() * np.sqrt(252) if len(ret) >= 252 else vol_60d
    vol_ratio = vol_60d / max(vol_252d, 0.01)
    vol_score = max(0, 100 - abs(vol_ratio - 1.0) * 100)  # cerca de 1.0 = estable

    # 2) Correlation stability (con benchmark si existe)
    if benchmark_df is not None and 'close' in benchmark_df.columns:
        bench_close = pd.to_numeric(benchmark_df['close'], errors='coerce').dropna()
        bench_ret = bench_close.pct_change().reindex(ret.index).dropna()
        common = ret.index.intersection(bench_ret.index)
        if len(common) > 60:
            corr_60d = ret.loc[common].tail(60).corr(bench_ret.loc[common].tail(60))
            # Correlación moderada es buena (0.5-0.7), muy alta o muy baja es riesgo
            corr_score = 100 - abs(corr_60d - 0.6) * 150  # óptimo en 0.6
            corr_score = np.clip(corr_score, 0, 100)
        else:
            corr_score = 50
    else:
        corr_score = 60  # sin benchmark, asumimos moderado

    # 3) Drawdown depth (menor es mejor)
    cum_ret = (1 + ret).cumprod()
    running_max = cum_ret.cummax()
    dd = (cum_ret / running_max - 1.0)
    max_dd = abs(dd.min())
    dd_score = max(0, 100 - max_dd * 200)  # 50% DD = 0 pts

    # 4) Momentum consistency (menos cambios de signo = mejor)
    if len(ret) >= 21:
        mom_12m = close.pct_change(252).iloc[-1] if len(close) >= 252 else 0
        mom_3m = close.pct_change(63).iloc[-1] if len(close) >= 63 else 0
        mom_1m = close.pct_change(21).iloc[-1] if len(close) >= 21 else 0

        # Consistencia: si todos tienen mismo signo = mejor
        signs = [np.sign(x) for x in [mom_12m, mom_3m, mom_1m] if not np.isnan(x)]
        if signs:
            consistency = abs(sum(signs)) / len(signs)  # 1.0 = todos igual signo
            mom_score = consistency * 100
        else:
            mom_score = 50
    else:
        mom_score = 50

    # Composite technical
    tech_score = (
        0.30 * vol_score +
        0.25 * corr_score +
        0.25 * dd_score +
        0.20 * mom_score
    )

    return float(np.clip(tech_score, 0, 100))


def compute_asset_quality_3d(
    symbol: str,
    price_df: pd.DataFrame,
    fundamentals_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    universe_stats: Optional[Dict] = None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> QualityResult:
    """
    Calcula quality score 3D para un activo.

    Args:
        symbol: ticker del activo
        price_df: DataFrame con columnas ['close', 'volume'] (opcional)
        fundamentals_df: DataFrame con datos FMP (marketCap, debtToEquity, etc.)
        benchmark_df: DataFrame con precios del benchmark (para correlación)
        universe_stats: Dict con estadísticas del universo (para percentiles)
        weights: (liq_weight, fund_weight, tech_weight) - default (0.4, 0.3, 0.3)

    Returns:
        QualityResult con scores y metadata
    """
    # Calcula 3 dimensiones
    liq_score, liq_meta = _liquidity_score(symbol, price_df, universe_stats)
    fund_score = _fundamental_score(symbol, fundamentals_df, universe_stats)
    tech_score = _technical_score(symbol, price_df, benchmark_df, universe_stats)

    # Composite score (weighted)
    w_liq, w_fund, w_tech = weights
    composite = w_liq * liq_score + w_fund * fund_score + w_tech * tech_score
    composite = float(np.clip(composite, 0, 100))

    # Map composite score → position cap
    if composite >= 80:
        pos_cap = 0.10  # excelente
    elif composite >= 60:
        pos_cap = 0.06  # bueno
    elif composite >= 40:
        pos_cap = 0.04  # fair
    else:
        pos_cap = 0.02  # pobre (o considerar exclude)

    return QualityResult(
        symbol=symbol,
        quality_score=composite,
        liq_score=liq_score,
        fund_score=fund_score,
        tech_score=tech_score,
        position_cap=pos_cap,
        ADV=liq_meta.get('ADV', 0.0),
        spread_bps=liq_meta.get('spread_bps', 0.0),
        days_to_liquidate=liq_meta.get('days_to_liquidate', 0.0)
    )


def compute_quality_batch(
    symbols: list[str],
    price_panel: Dict[str, pd.DataFrame],
    fundamentals_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> pd.DataFrame:
    """
    Calcula quality scores para un batch de símbolos.

    Returns:
        DataFrame con columnas: symbol, quality_score, liq_score, fund_score,
                                tech_score, position_cap, ADV, spread_bps, days_to_liquidate
    """
    # Primero calcula universe stats para percentiles
    universe_stats = {
        'ADV': [],
        'spread_bps': [],
        'vol_cv': []
    }

    # Pre-scan para estadísticas del universo
    for sym in symbols:
        if sym not in price_panel:
            continue
        _, liq_meta = _liquidity_score(sym, price_panel[sym], universe_stats=None)
        universe_stats['ADV'].append(liq_meta.get('ADV', 0))
        universe_stats['spread_bps'].append(liq_meta.get('spread_bps', 0))
        universe_stats['vol_cv'].append(liq_meta.get('vol_cv', 1.0))

    # Convierte a Series para percentiles
    for k in universe_stats:
        universe_stats[k] = pd.Series(universe_stats[k])

    # Calcula quality para cada activo
    results = []
    for sym in symbols:
        if sym not in price_panel:
            continue

        try:
            result = compute_asset_quality_3d(
                symbol=sym,
                price_df=price_panel[sym],
                fundamentals_df=fundamentals_df,
                benchmark_df=benchmark_df,
                universe_stats=universe_stats,
                weights=weights
            )
            results.append(result.to_dict())
        except Exception as e:
            # Si falla, agrega con score neutral
            results.append({
                'symbol': sym,
                'quality_score': 50.0,
                'liq_score': 50.0,
                'fund_score': 50.0,
                'tech_score': 50.0,
                'position_cap': 0.04,
                'ADV': 0.0,
                'spread_bps': 0.0,
                'days_to_liquidate': 0.0
            })

    df = pd.DataFrame(results)
    return df.sort_values('quality_score', ascending=False).reset_index(drop=True)
