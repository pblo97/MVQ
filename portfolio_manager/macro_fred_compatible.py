# portfolio_manager/macro_fred_compatible.py
"""
Macro Z-Score Calculator - 100% Compatible con MacroArimax (FRED/NY Fed)

Especificación:
- Normalización por familias ($ billions vs bp)
- Signos económicos correctos (drenan vs inyectan)
- Net Liquidity calculation
- Winsorización p1-p99
- Clip |z| ≤ 3.5
- Ventanas: 252d (anual) o 126d (semi)
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Literal


# ========== CONSTANTS ==========

# Indicadores por familia
FAMILY_DOLLARS = [
    'RRPONTSYD',     # ON RRP
    'WTREGEN',       # TGA
    'WRESBAL',       # Bank Reserves
    'WALCL',         # Fed Balance
    'NL'             # Net Liquidity (derived)
]

FAMILY_BASIS_POINTS = [
    'TGCRRATE',      # Repo GC tri-party
    'SOFR_EFFR',     # SOFR - EFFR spread (derived)
    'OBFR_SOFR',     # OBFR - SOFR spread (derived)
    'BAMLH0A0HYM2',  # HY OAS
    'T10Y2Y'         # 10y-2y curve
]

# Signos económicos (True = invertir signo, mayor valor = peor)
ECONOMIC_SIGNS_INVERT = {
    # Drenan liquidez (negativo)
    'WTREGEN': True,        # TGA: +ΔTGA drena
    'RRPONTSYD': True,      # RRP: +ΔRRP drena
    'SOFR_EFFR': True,      # Spread: mayor = tensión
    'OBFR_SOFR': True,      # Spread: mayor = tensión
    'TGCRRATE': True,       # Repo: mayor rate = tensión
    'BAMLH0A0HYM2': True,   # HY OAS: mayor = estrés

    # Inyectan liquidez (positivo)
    'WRESBAL': False,       # Reserves: +ΔRes inyecta
    'WALCL': False,         # Fed balance: +Δ inyecta
    'NL': False,            # Net Liquidity: +ΔNL mejor
    'T10Y2Y': False         # Curve: informativo (no invertir por defecto)
}


# ========== UTILITIES ==========

def winsorize(series: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    """
    Winsoriza serie en percentiles p_low y p_high.

    Args:
        series: Serie a winsorizar
        p_low: Percentil inferior (default 1%)
        p_high: Percentil superior (default 99%)

    Returns:
        Serie winsorizada
    """
    series = pd.to_numeric(series, errors='coerce')
    if series.dropna().empty:
        return series

    low = series.quantile(p_low)
    high = series.quantile(p_high)
    return series.clip(lower=low, upper=high)


def calculate_delta(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calcula Δ (diferencia) de una serie.

    Args:
        series: Serie temporal
        periods: Periodos de diferencia (1 = diario, 7 = semanal)

    Returns:
        Serie con deltas
    """
    return series.diff(periods=periods)


# ========== NET LIQUIDITY ==========

def calculate_net_liquidity(
    reserves: pd.Series,
    tga: pd.Series,
    rrp: pd.Series
) -> pd.Series:
    """
    Calcula Net Liquidity según fórmula MacroArimax:

    NL_t = WRESBAL_t - WTREGEN_t - RRPONTSYD_t

    Args:
        reserves: WRESBAL (bank reserves)
        tga: WTREGEN (Treasury General Account)
        rrp: RRPONTSYD (ON RRP)

    Returns:
        Serie Net Liquidity
    """
    reserves = pd.to_numeric(reserves, errors='coerce')
    tga = pd.to_numeric(tga, errors='coerce')
    rrp = pd.to_numeric(rrp, errors='coerce')

    nl = reserves - tga - rrp
    return nl.rename('NL')


def calculate_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula spreads derivados:
    - SOFR_EFFR = SOFR - EFFR
    - OBFR_SOFR = OBFR - SOFR

    Args:
        df: DataFrame con columnas SOFR, EFFR, OBFR

    Returns:
        DataFrame con columnas spread agregadas
    """
    result = df.copy()

    if {'SOFR', 'EFFR'}.issubset(df.columns):
        result['SOFR_EFFR'] = pd.to_numeric(df['SOFR'], errors='coerce') - pd.to_numeric(df['EFFR'], errors='coerce')

    if {'OBFR', 'SOFR'}.issubset(df.columns):
        result['OBFR_SOFR'] = pd.to_numeric(df['OBFR'], errors='coerce') - pd.to_numeric(df['SOFR'], errors='coerce')

    return result


# ========== Z-SCORE CALCULATION (BY FAMILY) ==========

def calculate_zscore_by_family(
    series: pd.Series,
    family: Literal['dollars', 'bp'],
    window: int = 252,
    use_delta: bool = True,
    delta_periods: int = 1,
    winsor: bool = True,
    clip_z: float = 3.5
) -> pd.Series:
    """
    Calcula z-score normalizado por familia, siguiendo especificación MacroArimax.

    Steps:
    1. Calcular Δ (si use_delta=True)
    2. Winsorizar p1-p99 (si winsor=True)
    3. Rolling z-score (ventana window)
    4. Clip |z| ≤ clip_z

    Args:
        series: Serie temporal del indicador
        family: 'dollars' o 'bp' (determina normalización)
        window: Ventana rolling (252d anual, 126d semi)
        use_delta: Si calcular Δ antes de z-score
        delta_periods: Periodos de delta (1 = diario, 7 = semanal)
        winsor: Si aplicar winsorización
        clip_z: Límite de clip para |z|

    Returns:
        Serie con z-scores normalizados
    """
    series = pd.to_numeric(series, errors='coerce')

    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index, name=series.name)

    # Step 1: Δ si necesario
    if use_delta:
        series_transformed = calculate_delta(series, periods=delta_periods)
    else:
        series_transformed = series.copy()

    # Step 2: Winsorizar
    if winsor:
        series_transformed = winsorize(series_transformed, p_low=0.01, p_high=0.99)

    # Step 3: Rolling z-score
    roll = series_transformed.rolling(window=window, min_periods=max(window//2, 60))
    mean = roll.mean()
    std = roll.std()

    zscore = (series_transformed - mean) / std.replace(0, np.nan)

    # Step 4: Clip
    zscore = zscore.clip(lower=-clip_z, upper=clip_z)

    return zscore


def calculate_composite_zscore_macroarimax(
    indicators_df: pd.DataFrame,
    window: int = 252,
    weights: Optional[Dict[str, float]] = None,
    clip_z: float = 3.5
) -> pd.DataFrame:
    """
    Calcula composite z-score siguiendo especificación MacroArimax completa.

    Process:
    1. Deriva Net Liquidity si tiene componentes
    2. Deriva spreads si tiene tasas
    3. Calcula z-scores por familia ($ separado de bp)
    4. Aplica signos económicos (drenan vs inyectan)
    5. Combina en composite weighted

    Args:
        indicators_df: DataFrame con indicadores FRED
        window: Ventana rolling (252d o 126d)
        weights: Pesos por indicador (default: equal dentro de familia)
        clip_z: Límite de clip

    Returns:
        DataFrame con z-scores individuales + composite_z
    """
    df = indicators_df.copy()

    # Step 1: Calcula Net Liquidity si posible
    if {'WRESBAL', 'WTREGEN', 'RRPONTSYD'}.issubset(df.columns):
        df['NL'] = calculate_net_liquidity(
            df['WRESBAL'],
            df['WTREGEN'],
            df['RRPONTSYD']
        )

    # Step 2: Calcula spreads derivados
    df = calculate_spreads(df)

    # Step 3: Identifica columnas disponibles por familia
    available_dollars = [col for col in FAMILY_DOLLARS if col in df.columns]
    available_bp = [col for col in FAMILY_BASIS_POINTS if col in df.columns]

    if not available_dollars and not available_bp:
        return pd.DataFrame(index=df.index)

    # Default weights: equal dentro de familia
    if weights is None:
        weights = {}
        n_dollars = len(available_dollars)
        n_bp = len(available_bp)
        total = n_dollars + n_bp

        for col in available_dollars:
            weights[col] = 0.5 / n_dollars if n_dollars > 0 else 0
        for col in available_bp:
            weights[col] = 0.5 / n_bp if n_bp > 0 else 0

    # Step 4: Calcula z-scores por familia
    result = pd.DataFrame(index=df.index)

    for col in available_dollars:
        zscore = calculate_zscore_by_family(
            series=df[col],
            family='dollars',
            window=window,
            use_delta=True,  # Δ para flujos
            delta_periods=1,
            winsor=True,
            clip_z=clip_z
        )

        # Aplica signo económico
        if ECONOMIC_SIGNS_INVERT.get(col, False):
            zscore = -zscore

        result[f"{col}_z"] = zscore

    for col in available_bp:
        # bp típicamente usan niveles (no Δ), excepto si es rate change
        use_delta_bp = col in ['TGCRRATE']  # rates pueden usar Δ

        zscore = calculate_zscore_by_family(
            series=df[col],
            family='bp',
            window=window,
            use_delta=use_delta_bp,
            delta_periods=1,
            winsor=True,
            clip_z=clip_z
        )

        # Aplica signo económico
        if ECONOMIC_SIGNS_INVERT.get(col, False):
            zscore = -zscore

        result[f"{col}_z"] = zscore

    # Step 5: Composite weighted
    weighted_zscores = []
    for col in (available_dollars + available_bp):
        zcol = f"{col}_z"
        if zcol in result.columns:
            weight = weights.get(col, 1.0 / (len(available_dollars) + len(available_bp)))
            weighted_zscores.append(result[zcol] * weight)

    if weighted_zscores:
        result['composite_z'] = pd.concat(weighted_zscores, axis=1).sum(axis=1)
        # Clip composite también
        result['composite_z'] = result['composite_z'].clip(lower=-clip_z, upper=clip_z)
    else:
        result['composite_z'] = 0.0

    return result


# ========== STREAMLIT INTEGRATION ==========

def load_fred_csv_macroarimax(
    filepath_or_buffer,
    date_column: str = 'Date'
) -> pd.DataFrame:
    """
    Carga CSV con indicadores FRED desde MacroArimax.

    Expected columns:
    - Date
    - RRPONTSYD, WTREGEN, WRESBAL (para Net Liquidity)
    - SOFR, EFFR, OBFR (para spreads)
    - BAMLH0A0HYM2, T10Y2Y, etc.

    Args:
        filepath_or_buffer: Path o buffer del CSV
        date_column: Nombre columna fecha

    Returns:
        DataFrame con index temporal
    """
    df = pd.read_csv(filepath_or_buffer, parse_dates=[date_column])
    df = df.set_index(date_column).sort_index()

    # Forward fill para NA (típico en FRED)
    df = df.ffill()

    return df


def calculate_macro_zscore_from_fred_csv(
    filepath_or_buffer,
    window: int = 252,
    weights: Optional[Dict[str, float]] = None,
    clip_z: float = 3.5
) -> tuple[pd.DataFrame, float]:
    """
    Pipeline completo: carga CSV FRED → calcula z-scores MacroArimax → devuelve último.

    Args:
        filepath_or_buffer: Path al CSV con datos FRED
        window: Ventana rolling (252d anual, 126d semi)
        weights: Pesos por indicador
        clip_z: Clip limit

    Returns:
        (DataFrame con z-scores, composite_z último valor)
    """
    # Load FRED data
    fred_df = load_fred_csv_macroarimax(filepath_or_buffer)

    if fred_df.empty:
        return pd.DataFrame(), 0.0

    # Calculate z-scores
    result_df = calculate_composite_zscore_macroarimax(
        indicators_df=fred_df,
        window=window,
        weights=weights,
        clip_z=clip_z
    )

    # Extract last composite_z
    if 'composite_z' in result_df.columns:
        composite_z_last = float(result_df['composite_z'].dropna().iloc[-1]) if not result_df['composite_z'].dropna().empty else 0.0
    else:
        composite_z_last = 0.0

    return result_df, composite_z_last


# ========== PRESETS ==========

def get_macroarimax_default_weights() -> Dict[str, float]:
    """
    Pesos típicos usados en MacroArimax (basados en especificación).

    Familias balanceadas: 50% $ + 50% bp
    """
    return {
        # Family $ (50% total)
        'RRPONTSYD': 0.15,   # RRP
        'WTREGEN': 0.10,     # TGA
        'WRESBAL': 0.15,     # Reserves
        'NL': 0.10,          # Net Liquidity

        # Family bp (50% total)
        'SOFR_EFFR': 0.10,   # Spread SOFR-EFFR
        'OBFR_SOFR': 0.05,   # Spread OBFR-SOFR
        'TGCRRATE': 0.05,    # Repo GC
        'BAMLH0A0HYM2': 0.15, # HY OAS
        'T10Y2Y': 0.15       # 10y-2y curve
    }


# ========== QUARTER-END MODE (OPTIONAL) ==========

def is_quarter_end(date: pd.Timestamp, buffer_days: int = 3) -> bool:
    """
    Detecta si una fecha es fin de trimestre (o cerca).

    Args:
        date: Fecha a verificar
        buffer_days: Días de buffer antes/después de fin de trimestre

    Returns:
        True si es quarter-end period
    """
    from datetime import timedelta

    # Último día del mes
    is_month_end = date == pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(0)

    # Es trimestre (Mar, Jun, Sep, Dec)
    is_quarter_month = date.month in [3, 6, 9, 12]

    if is_month_end and is_quarter_month:
        return True

    # Check buffer
    quarter_ends = pd.date_range(
        start=f"{date.year}-01-01",
        end=f"{date.year}-12-31",
        freq='Q'
    )

    for qe in quarter_ends:
        if abs((date - qe).days) <= buffer_days:
            return True

    return False


def adjust_thresholds_quarter_end(
    zscore: float,
    indicator_name: str,
    is_qe: bool,
    qe_multiplier: float = 1.5
) -> float:
    """
    Ajusta umbrales en quarter-end mode (spreads/repo más altos son normales).

    Args:
        zscore: Z-score original
        indicator_name: Nombre del indicador
        is_qe: Si estamos en quarter-end period
        qe_multiplier: Multiplicador para umbrales (default 1.5×)

    Returns:
        Z-score ajustado
    """
    if not is_qe:
        return zscore

    # Indicadores que se tensionan en QE (ajustar umbrales)
    qe_sensitive = ['TGCRRATE', 'SOFR_EFFR', 'OBFR_SOFR', 'RRPONTSYD']

    if any(ind in indicator_name for ind in qe_sensitive):
        # Reduce magnitud del z-score (tensión es "normal" en QE)
        return zscore / qe_multiplier

    return zscore
