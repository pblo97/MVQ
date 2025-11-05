# portfolio_manager/macro_simple.py
"""
Macro Z-Score Calculator - Simplified
Calcula z-score compuesto desde indicadores macro subidos por el usuario.

Compatible con outputs de MacroArimax o macro_monitor_bundle.csv
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict


def calculate_zscore_rolling(series: pd.Series, window: int = 36) -> pd.Series:
    """
    Calcula z-score rolling de una serie.

    Args:
        series: Serie temporal con valores del indicador
        window: Ventana rolling (meses/semanas)

    Returns:
        Serie con z-scores rolling
    """
    series = pd.to_numeric(series, errors='coerce')
    roll = series.rolling(window=window, min_periods=max(window//2, 12))
    mean = roll.mean()
    std = roll.std()
    zscore = (series - mean) / std.replace(0, np.nan)
    return zscore


def calculate_composite_zscore(
    indicators_df: pd.DataFrame,
    indicator_columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    invert_signs: Optional[Dict[str, bool]] = None,
    window: int = 36
) -> pd.DataFrame:
    """
    Calcula z-score compuesto desde múltiples indicadores.

    Args:
        indicators_df: DataFrame con columnas de indicadores
        indicator_columns: Lista de columnas a incluir
        weights: Dict con pesos por indicador (default: equal weight)
        invert_signs: Dict indicando si invertir signo (True = mayor valor = peor)
        window: Ventana rolling para z-scores

    Returns:
        DataFrame con columnas: [indicadores z-scores individuales, composite_z]
    """
    if weights is None:
        weights = {col: 1.0 / len(indicator_columns) for col in indicator_columns}

    if invert_signs is None:
        invert_signs = {}

    result = pd.DataFrame(index=indicators_df.index)

    # Calcula z-score de cada indicador
    for col in indicator_columns:
        if col not in indicators_df.columns:
            continue

        zscore = calculate_zscore_rolling(indicators_df[col], window=window)

        # Invierte signo si necesario
        if invert_signs.get(col, False):
            zscore = -zscore

        result[f"{col}_z"] = zscore

    # Composite z-score (weighted average)
    weighted_zscores = []
    for col in indicator_columns:
        zcol = f"{col}_z"
        if zcol in result.columns:
            weight = weights.get(col, 1.0 / len(indicator_columns))
            weighted_zscores.append(result[zcol] * weight)

    if weighted_zscores:
        result['composite_z'] = pd.concat(weighted_zscores, axis=1).sum(axis=1)
    else:
        result['composite_z'] = 0.0

    return result


def load_macro_from_csv(
    filepath_or_buffer,
    date_column: str = 'Date',
    auto_detect_indicators: bool = True,
    indicator_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Carga indicadores macro desde CSV.

    Args:
        filepath_or_buffer: Path al CSV o buffer subido
        date_column: Nombre de columna con fechas
        auto_detect_indicators: Si True, detecta columnas numéricas automáticamente
        indicator_columns: Lista explícita de columnas a usar (si no auto_detect)

    Returns:
        DataFrame con index temporal y columnas de indicadores
    """
    df = pd.read_csv(filepath_or_buffer, parse_dates=[date_column])
    df = df.set_index(date_column).sort_index()

    if auto_detect_indicators:
        # Detecta columnas numéricas (excluye composites existentes)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['composite_z', 'macro_z', 'COMPOSITE_Z', 'COMPOSITE_PCA']
        indicator_columns = [c for c in numeric_cols if c not in exclude]

    if indicator_columns:
        df = df[indicator_columns]

    return df


def calculate_macro_zscore_from_csv(
    filepath_or_buffer,
    window: int = 36,
    weights: Optional[Dict[str, float]] = None,
    invert_signs: Optional[Dict[str, bool]] = None
) -> tuple[pd.DataFrame, float]:
    """
    Pipeline completo: carga CSV → calcula z-scores → devuelve último valor.

    Args:
        filepath_or_buffer: Path al CSV o buffer
        window: Ventana rolling
        weights: Pesos por indicador
        invert_signs: Inversión de signo por indicador

    Returns:
        (DataFrame con z-scores, macro_z último valor)
    """
    # Load indicators
    indicators_df = load_macro_from_csv(filepath_or_buffer)

    if indicators_df.empty:
        return pd.DataFrame(), 0.0

    # Calculate z-scores
    result_df = calculate_composite_zscore(
        indicators_df=indicators_df,
        indicator_columns=indicators_df.columns.tolist(),
        weights=weights,
        invert_signs=invert_signs,
        window=window
    )

    # Extract last composite_z
    macro_z_last = float(result_df['composite_z'].dropna().iloc[-1]) if not result_df['composite_z'].dropna().empty else 0.0

    return result_df, macro_z_last


# ========== PRESETS COMUNES ==========

def get_fred_preset_weights() -> Dict[str, float]:
    """
    Pesos típicos para indicadores FRED (compatibles con ma_streamlit.py).

    Indicadores:
    - Yield curve (T10Y3M, T10Y2Y)
    - Credit spreads (BAMLH0A0HYM2, BAA_AAA)
    - Liquidity (Reverse_Repo_Volume, WTREGEN, sofr_spread)
    - USD strength (USD_BROAD)
    """
    return {
        'T10Y3M': 0.15,
        'T10Y2Y': 0.15,
        'BAMLH0A0HYM2': 0.15,
        'BAA_AAA': 0.15,
        'Reverse_Repo_Volume': 0.10,
        'WTREGEN': 0.10,
        'sofr_spread': 0.05,
        'USD_BROAD': 0.15
    }


def get_fred_preset_invert_signs() -> Dict[str, bool]:
    """
    Indicadores que deben invertirse (mayor valor = peor condición).

    - USD_BROAD: True (dollar fuerte = peor para risk assets)
    - Credit spreads: False (mayor spread = peor, pero ya negativo)
    - Yield curve: False (positivo = expansión)
    """
    return {
        'USD_BROAD': True,
        'BAMLH0A0HYM2': False,
        'BAA_AAA': False
    }


# ========== STREAMLIT HELPER ==========

def streamlit_macro_uploader(
    st,
    default_window: int = 36,
    show_chart: bool = True
) -> tuple[Optional[pd.DataFrame], float]:
    """
    Helper para Streamlit: uploader de CSV macro + cálculo automático.

    Args:
        st: Streamlit module
        default_window: Ventana rolling default
        show_chart: Si mostrar chart del composite_z

    Returns:
        (DataFrame con z-scores, macro_z_last)
    """
    st.markdown("### 📊 Macro Indicators (FRED)")
    st.caption("Upload CSV with macro indicators (e.g., from MacroArimax)")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV should have Date column + indicator columns (T10Y3M, BAA_AAA, etc.)"
    )

    if uploaded_file is None:
        st.info("Upload a CSV with macro indicators to calculate z-score")
        return None, 0.0

    try:
        # Calculate
        with st.spinner("Calculating macro z-score..."):
            result_df, macro_z_last = calculate_macro_zscore_from_csv(
                uploaded_file,
                window=default_window,
                weights=get_fred_preset_weights(),
                invert_signs=get_fred_preset_invert_signs()
            )

        # Display
        st.success(f"✓ Macro z-score calculated: **{macro_z_last:.2f}**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Composite Z-Score", f"{macro_z_last:.2f}")
        with col2:
            if macro_z_last > 0.5:
                regime = "ON (Risk-On)"
            elif macro_z_last < -0.5:
                regime = "OFF (Risk-Off)"
            else:
                regime = "NEUTRAL"
            st.metric("Regime", regime)

        # Chart
        if show_chart:
            try:
                import plotly.express as px
                fig = px.line(
                    result_df['composite_z'].rename_axis('Date').reset_index(),
                    x='Date',
                    y='composite_z',
                    title="Composite Z-Score Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Fallback to dataframe
                st.line_chart(result_df['composite_z'])

        # Show indicators table
        with st.expander("View indicator z-scores"):
            st.dataframe(result_df.tail(12), use_container_width=True)

        return result_df, macro_z_last

    except Exception as e:
        st.error(f"Error calculating macro z-score: {e}")
        st.exception(e)
        return None, 0.0
