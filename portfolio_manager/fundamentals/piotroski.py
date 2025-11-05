# portfolio_manager/fundamentals/piotroski.py
"""
Piotroski F-Score Calculation
Based on Piotroski (2000) "Value Investing: The Use of Historical Financial Statement Information"

Calcula 9 señales binarias de calidad fundamental:
- Profitability (4): ROA, CFO, ΔROA, Accruals
- Leverage/Liquidity (3): ΔLEVER, ΔLIQUID, EQ_OFFER
- Operating Efficiency (2): ΔMARGIN, ΔTURNOVER

Score: 0-9 (9 = highest quality, 0 = lowest)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import time

# Reuse FMP utilities from existing module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qvm_trend.fquality.fmp_quality import fetch_quarterly, SLEEP


# ========== PIOTROSKI F-SCORE CALCULATION ==========

def calculate_piotroski_signals(
    inc: pd.DataFrame,
    bal: pd.DataFrame,
    cfs: pd.DataFrame,
    rat: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula las 9 señales de Piotroski para cada trimestre.

    Returns:
        DataFrame con index=date y columnas:
        - F_ROA: ROA > 0
        - F_CFO: CFO > 0
        - F_ΔROA: ROA mejorando
        - F_ACCRUAL: Accrual < 0 (CFO > NI)
        - F_ΔLEVER: Leverage reduciéndose
        - F_ΔLIQUID: Liquidez mejorando
        - F_EQ_OFFER: No equity issuance
        - F_ΔMARGIN: Margen mejorando
        - F_ΔTURN: Asset turnover mejorando
        - F_SCORE: Suma de las 9 señales (0-9)
    """
    # Align all data to income statement dates (most complete)
    if inc.empty:
        return pd.DataFrame()

    dates = inc.index

    # ========== PROFITABILITY ==========

    # 1. F_ROA: ROA > 0
    # ROA = Net Income / Total Assets
    net_income = pd.to_numeric(inc.get("netIncome", pd.Series(index=dates)), errors='coerce').reindex(dates)
    total_assets = pd.to_numeric(bal.get("totalAssets", pd.Series(index=dates)), errors='coerce').reindex(dates)

    # Use TTM (trailing 4Q sum for income, average for assets)
    net_income_ttm = net_income.rolling(4, min_periods=4).sum()
    total_assets_avg = total_assets.rolling(4, min_periods=4).mean()

    roa = net_income_ttm / total_assets_avg.replace(0, np.nan)
    F_ROA = (roa > 0).astype(int)

    # 2. F_CFO: CFO > 0
    cfo_col = "netCashProvidedByOperatingActivities"
    if cfo_col not in cfs.columns:
        cfo_col = next(
            (c for c in cfs.columns if isinstance(c, str) and ("operat" in c.lower() and "cash" in c.lower())),
            None
        )

    if cfo_col:
        cfo = pd.to_numeric(cfs[cfo_col], errors='coerce').reindex(dates)
        cfo_ttm = cfo.rolling(4, min_periods=4).sum()
        F_CFO = (cfo_ttm > 0).astype(int)
    else:
        F_CFO = pd.Series(0, index=dates)

    # 3. F_ΔROA: ΔROA > 0
    F_DELTA_ROA = (roa.diff(4) > 0).astype(int)  # Compare YoY (4 quarters ago)

    # 4. F_ACCRUAL: Accrual < 0 (Quality earnings: CFO > Net Income)
    if cfo_col:
        accrual = net_income_ttm - cfo_ttm
        F_ACCRUAL = (accrual < 0).astype(int)
    else:
        F_ACCRUAL = pd.Series(0, index=dates)

    # ========== LEVERAGE/LIQUIDITY ==========

    # 5. F_ΔLEVER: ΔLEVER < 0 (Leverage decreasing)
    # Leverage = Total Debt / Total Assets
    total_debt = pd.to_numeric(bal.get("totalDebt", pd.Series(index=dates)), errors='coerce').reindex(dates)
    leverage = total_debt / total_assets.replace(0, np.nan)
    F_DELTA_LEVER = (leverage.diff(4) < 0).astype(int)

    # 6. F_ΔLIQUID: ΔLIQUID > 0 (Liquidity improving)
    # Current Ratio = Current Assets / Current Liabilities
    current_assets = pd.to_numeric(bal.get("totalCurrentAssets", pd.Series(index=dates)), errors='coerce').reindex(dates)
    current_liab = pd.to_numeric(bal.get("totalCurrentLiabilities", pd.Series(index=dates)), errors='coerce').reindex(dates)
    current_ratio = current_assets / current_liab.replace(0, np.nan)
    F_DELTA_LIQUID = (current_ratio.diff(4) > 0).astype(int)

    # 7. F_EQ_OFFER: EQ_OFFER = 0 (No equity issuance)
    # Check if shares outstanding increased
    shares_col = next(
        (c for c in bal.columns if isinstance(c, str) and "shares" in c.lower() and "outstanding" in c.lower()),
        None
    )
    if shares_col:
        shares = pd.to_numeric(bal[shares_col], errors='coerce').reindex(dates)
        shares_pct_change = shares.pct_change(4)  # YoY change
        F_EQ_OFFER = (shares_pct_change <= 0).astype(int)  # 1 if no issuance (shares same or less)
    else:
        F_EQ_OFFER = pd.Series(0, index=dates)

    # ========== OPERATING EFFICIENCY ==========

    # 8. F_ΔMARGIN: ΔMARGIN > 0 (Gross margin improving)
    revenue = pd.to_numeric(inc.get("revenue", pd.Series(index=dates)), errors='coerce').reindex(dates)
    gross_profit = pd.to_numeric(inc.get("grossProfit", pd.Series(index=dates)), errors='coerce').reindex(dates)

    revenue_ttm = revenue.rolling(4, min_periods=4).sum()
    gross_profit_ttm = gross_profit.rolling(4, min_periods=4).sum()

    gross_margin = gross_profit_ttm / revenue_ttm.replace(0, np.nan)
    F_DELTA_MARGIN = (gross_margin.diff(4) > 0).astype(int)

    # 9. F_ΔTURN: ΔTURNOVER > 0 (Asset turnover improving)
    # Asset Turnover = Revenue / Total Assets
    asset_turnover = revenue_ttm / total_assets_avg.replace(0, np.nan)
    F_DELTA_TURN = (asset_turnover.diff(4) > 0).astype(int)

    # ========== AGGREGATE F-SCORE ==========

    signals_df = pd.DataFrame({
        'F_ROA': F_ROA,
        'F_CFO': F_CFO,
        'F_DELTA_ROA': F_DELTA_ROA,
        'F_ACCRUAL': F_ACCRUAL,
        'F_DELTA_LEVER': F_DELTA_LEVER,
        'F_DELTA_LIQUID': F_DELTA_LIQUID,
        'F_EQ_OFFER': F_EQ_OFFER,
        'F_DELTA_MARGIN': F_DELTA_MARGIN,
        'F_DELTA_TURN': F_DELTA_TURN
    }, index=dates)

    # Replace NaN with 0 for signals (conservative: if data missing, signal = 0)
    signals_df = signals_df.fillna(0).astype(int)

    # F-Score = sum of all signals (0-9)
    signals_df['F_SCORE'] = signals_df[[
        'F_ROA', 'F_CFO', 'F_DELTA_ROA', 'F_ACCRUAL',
        'F_DELTA_LEVER', 'F_DELTA_LIQUID', 'F_EQ_OFFER',
        'F_DELTA_MARGIN', 'F_DELTA_TURN'
    ]].sum(axis=1)

    return signals_df


def calculate_piotroski_history(
    symbols: List[str],
    api_key: str
) -> pd.DataFrame:
    """
    Calcula histórico trimestral de Piotroski F-Score para lista de símbolos.

    Args:
        symbols: Lista de tickers
        api_key: FMP API key

    Returns:
        DataFrame con columnas: ['symbol', 'date', 'F_SCORE', ...signals...]
    """
    symbols = [s.upper() for s in symbols if s and isinstance(s, str)]
    rows = []

    for sym in symbols:
        try:
            data = fetch_quarterly(sym, api_key)
            inc, bal, cfs, rat = data["income"], data["balance"], data["cash"], data["ratios"]

            if inc.empty or bal.empty:
                continue

            signals_df = calculate_piotroski_signals(inc, bal, cfs, rat)

            if not signals_df.empty:
                # Only keep rows with valid F_SCORE (≥ 5 signals must be calculable)
                valid_mask = signals_df['F_SCORE'].notna()
                for date, row in signals_df[valid_mask].iterrows():
                    rows.append({
                        'symbol': sym,
                        'date': date,
                        'F_SCORE': int(row['F_SCORE']),
                        **{k: int(row[k]) for k in row.index if k.startswith('F_') and k != 'F_SCORE'}
                    })

            time.sleep(SLEEP)

        except Exception as e:
            print(f"Error calculating Piotroski for {sym}: {e}")
            continue

    return pd.DataFrame(rows)


def detect_fundamental_degradation(
    piotroski_hist: pd.DataFrame,
    symbol: str,
    degradation_threshold: int = 2
) -> Dict[str, any]:
    """
    Detecta degradación fundamental comparando F-Score último vs anterior.

    Args:
        piotroski_hist: DataFrame con columnas ['symbol', 'date', 'F_SCORE']
        symbol: Ticker a analizar
        degradation_threshold: Caída en F-Score que indica degradación (default: 2 puntos)

    Returns:
        dict con keys:
        - f_score_last: F-Score actual
        - f_score_prev: F-Score anterior (1Q ago)
        - f_score_delta: Cambio (last - prev)
        - degradation_flag: 'Degrading' / 'Improving' / 'Flat' / 'N/A'
    """
    if piotroski_hist is None or piotroski_hist.empty:
        return {
            'f_score_last': np.nan,
            'f_score_prev': np.nan,
            'f_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    # Filter by symbol
    df = piotroski_hist[piotroski_hist['symbol'].str.upper() == symbol.upper()].copy()

    if df.empty:
        return {
            'f_score_last': np.nan,
            'f_score_prev': np.nan,
            'f_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')

    if len(df) < 2:
        f_last = float(df['F_SCORE'].iloc[-1]) if len(df) == 1 else np.nan
        return {
            'f_score_last': f_last,
            'f_score_prev': np.nan,
            'f_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    # Get last 2 quarters
    f_last = float(df['F_SCORE'].iloc[-1])
    f_prev = float(df['F_SCORE'].iloc[-2])
    delta = f_last - f_prev

    # Classify
    if delta <= -degradation_threshold:
        flag = 'Degrading'
    elif delta >= degradation_threshold:
        flag = 'Improving'
    else:
        flag = 'Flat'

    return {
        'f_score_last': f_last,
        'f_score_prev': f_prev,
        'f_score_delta': delta,
        'degradation_flag': flag
    }


def interpret_fscore(score: float) -> str:
    """
    Interpreta el F-Score según literatura Piotroski.

    Args:
        score: F-Score (0-9)

    Returns:
        Interpretación textual
    """
    if np.isnan(score):
        return "N/A"

    score = int(score)

    if score >= 8:
        return "Excellent"
    elif score >= 7:
        return "Strong"
    elif score >= 5:
        return "Above Average"
    elif score >= 3:
        return "Below Average"
    else:
        return "Weak"
