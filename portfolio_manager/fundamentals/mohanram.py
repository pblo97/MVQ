# portfolio_manager/fundamentals/mohanram.py
"""
Mohanram G-Score for Growth Stocks

Based on Mohanram (2005):
"Separating Winners from Losers among Low Book-to-Market Stocks using Financial Statement Analysis"
Review of Accounting Studies, 10(2-3), 133-170.

G-Score is designed for GROWTH stocks (low B/M), complementing Piotroski F-Score (for VALUE stocks).

8 Signals (each 0 or 1):
1. G_ROA: ROA > industry median
2. G_CFO: CFO > industry median
3. G_ROA_CFO: CFO > ROA (earnings quality)
4. G_ROA_VAR: Variability of ROA < industry median (stability)
5. G_SALES_VAR: Variability of sales < industry median (stability)
6. G_RD: R&D intensity > industry median (innovation)
7. G_CAPEX: Capex intensity > industry median (investment)
8. G_AD: Advertising intensity > industry median (brand building)

Score: 0-8 (8 = highest quality growth, 0 = lowest)

Key difference from Piotroski:
- Piotroski: Improvement signals (Δ > 0) for distressed value stocks
- Mohanram: Level signals (> median) for healthy growth stocks
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
import sys
import os

# Reuse FMP utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from qvm_trend.fquality.fmp_quality import fetch_quarterly, SLEEP


def calculate_mohanram_signals(
    inc: pd.DataFrame,
    bal: pd.DataFrame,
    cfs: pd.DataFrame,
    rat: pd.DataFrame,
    industry_medians: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate Mohanram G-Score signals for each quarter.

    Args:
        inc: Income statement (quarterly)
        bal: Balance sheet (quarterly)
        cfs: Cash flow statement (quarterly)
        rat: Ratios (quarterly)
        industry_medians: Dict of industry median values (if None, uses cross-sectional)

    Returns:
        DataFrame with columns:
        - G_ROA, G_CFO, G_ROA_CFO, G_ROA_VAR, G_SALES_VAR, G_RD, G_CAPEX, G_AD
        - G_SCORE (0-8)
    """
    if inc.empty:
        return pd.DataFrame()

    dates = inc.index

    # ========== PROFITABILITY ==========

    # ROA (TTM)
    net_income = pd.to_numeric(inc.get('netIncome', pd.Series(index=dates)), errors='coerce').reindex(dates)
    total_assets = pd.to_numeric(bal.get('totalAssets', pd.Series(index=dates)), errors='coerce').reindex(dates)

    net_income_ttm = net_income.rolling(4, min_periods=4).sum()
    total_assets_avg = total_assets.rolling(4, min_periods=4).mean()

    roa = net_income_ttm / total_assets_avg.replace(0, np.nan)

    # CFO / Assets
    cfo_col = 'netCashProvidedByOperatingActivities'
    if cfo_col not in cfs.columns:
        cfo_col = next(
            (c for c in cfs.columns if isinstance(c, str) and 'operat' in c.lower() and 'cash' in c.lower()),
            None
        )

    if cfo_col:
        cfo = pd.to_numeric(cfs[cfo_col], errors='coerce').reindex(dates)
        cfo_ttm = cfo.rolling(4, min_periods=4).sum()
        cfo_to_assets = cfo_ttm / total_assets_avg.replace(0, np.nan)
    else:
        cfo_to_assets = pd.Series(np.nan, index=dates)

    # 1. G_ROA: ROA > industry median
    roa_median = industry_medians.get('ROA', 0.05) if industry_medians else roa.median()
    G_ROA = (roa > roa_median).astype(int)

    # 2. G_CFO: CFO > industry median
    cfo_median = industry_medians.get('CFO', 0.05) if industry_medians else cfo_to_assets.median()
    G_CFO = (cfo_to_assets > cfo_median).astype(int)

    # 3. G_ROA_CFO: CFO > ROA (earnings quality)
    G_ROA_CFO = (cfo_to_assets > roa).astype(int)

    # ========== VARIABILITY (STABILITY) ==========

    # 4. G_ROA_VAR: Variability of ROA < industry median
    # Use rolling std of ROA over 8 quarters
    roa_var = roa.rolling(8, min_periods=4).std()
    roa_var_median = industry_medians.get('ROA_VAR', 0.03) if industry_medians else roa_var.median()
    G_ROA_VAR = (roa_var < roa_var_median).astype(int)

    # 5. G_SALES_VAR: Variability of sales growth < industry median
    revenue = pd.to_numeric(inc.get('revenue', pd.Series(index=dates)), errors='coerce').reindex(dates)
    revenue_ttm = revenue.rolling(4, min_periods=4).sum()
    sales_growth = revenue_ttm.pct_change(4)  # YoY growth
    sales_var = sales_growth.rolling(8, min_periods=4).std()
    sales_var_median = industry_medians.get('SALES_VAR', 0.15) if industry_medians else sales_var.median()
    G_SALES_VAR = (sales_var < sales_var_median).astype(int)

    # ========== INVESTMENT & INNOVATION ==========

    # 6. G_RD: R&D intensity > industry median
    # R&D / Sales (TTM)
    rd_expense = pd.to_numeric(inc.get('researchAndDevelopmentExpenses', pd.Series(index=dates)), errors='coerce').reindex(dates)
    if rd_expense.isna().all():
        # Try alternative name
        rd_expense = pd.to_numeric(inc.get('researchAndDevelopment', pd.Series(index=dates)), errors='coerce').reindex(dates)

    rd_ttm = rd_expense.rolling(4, min_periods=4).sum()
    rd_intensity = rd_ttm / revenue_ttm.replace(0, np.nan)

    rd_median = industry_medians.get('RD', 0.05) if industry_medians else rd_intensity.median()
    G_RD = (rd_intensity > rd_median).astype(int)

    # 7. G_CAPEX: Capex intensity > industry median
    # Capex / Assets (TTM)
    capex = pd.to_numeric(cfs.get('capitalExpenditure', pd.Series(index=dates)), errors='coerce').reindex(dates)
    if capex.isna().all():
        capex = pd.to_numeric(cfs.get('capex', pd.Series(index=dates)), errors='coerce').reindex(dates)

    capex_ttm = capex.abs().rolling(4, min_periods=4).sum()  # Capex is usually negative
    capex_intensity = capex_ttm / total_assets_avg.replace(0, np.nan)

    capex_median = industry_medians.get('CAPEX', 0.05) if industry_medians else capex_intensity.median()
    G_CAPEX = (capex_intensity > capex_median).astype(int)

    # 8. G_AD: Advertising intensity > industry median
    # Advertising / Sales (TTM)
    # FMP often doesn't have advertising as separate line item
    # Use SG&A as proxy (includes advertising)
    sga = pd.to_numeric(inc.get('sellingGeneralAndAdministrativeExpenses', pd.Series(index=dates)), errors='coerce').reindex(dates)
    if sga.isna().all():
        sga = pd.to_numeric(inc.get('generalAndAdministrativeExpenses', pd.Series(index=dates)), errors='coerce').reindex(dates)

    sga_ttm = sga.rolling(4, min_periods=4).sum()
    ad_intensity = sga_ttm / revenue_ttm.replace(0, np.nan)

    ad_median = industry_medians.get('AD', 0.20) if industry_medians else ad_intensity.median()
    G_AD = (ad_intensity > ad_median).astype(int)

    # ========== AGGREGATE G-SCORE ==========

    signals_df = pd.DataFrame({
        'G_ROA': G_ROA,
        'G_CFO': G_CFO,
        'G_ROA_CFO': G_ROA_CFO,
        'G_ROA_VAR': G_ROA_VAR,
        'G_SALES_VAR': G_SALES_VAR,
        'G_RD': G_RD,
        'G_CAPEX': G_CAPEX,
        'G_AD': G_AD
    }, index=dates)

    # Replace NaN with 0 (conservative: missing data = 0)
    signals_df = signals_df.fillna(0).astype(int)

    # G-Score = sum of all signals (0-8)
    signals_df['G_SCORE'] = signals_df[[
        'G_ROA', 'G_CFO', 'G_ROA_CFO', 'G_ROA_VAR', 'G_SALES_VAR',
        'G_RD', 'G_CAPEX', 'G_AD'
    ]].sum(axis=1)

    return signals_df


def calculate_mohanram_history(
    symbols: List[str],
    api_key: str
) -> pd.DataFrame:
    """
    Calculate quarterly G-Score history for list of symbols.

    Args:
        symbols: List of tickers
        api_key: FMP API key

    Returns:
        DataFrame with columns: ['symbol', 'date', 'G_SCORE', ...signals...]
    """
    symbols = [s.upper() for s in symbols if s and isinstance(s, str)]
    rows = []

    for sym in symbols:
        try:
            data = fetch_quarterly(sym, api_key)
            inc, bal, cfs, rat = data['income'], data['balance'], data['cash'], data['ratios']

            if inc.empty or bal.empty:
                continue

            signals_df = calculate_mohanram_signals(inc, bal, cfs, rat, industry_medians=None)

            if not signals_df.empty:
                valid_mask = signals_df['G_SCORE'].notna()
                for date, row in signals_df[valid_mask].iterrows():
                    rows.append({
                        'symbol': sym,
                        'date': date,
                        'G_SCORE': int(row['G_SCORE']),
                        **{k: int(row[k]) for k in row.index if k.startswith('G_') and k != 'G_SCORE'}
                    })

            time.sleep(SLEEP)

        except Exception as e:
            print(f"Error calculating Mohanram for {sym}: {e}")
            continue

    return pd.DataFrame(rows)


def detect_growth_degradation(
    mohanram_hist: pd.DataFrame,
    symbol: str,
    degradation_threshold: int = 2
) -> Dict[str, any]:
    """
    Detect growth quality degradation using G-Score.

    Args:
        mohanram_hist: DataFrame with ['symbol', 'date', 'G_SCORE']
        symbol: Ticker to analyze
        degradation_threshold: Drop in G-Score indicating degradation

    Returns:
        dict with keys:
        - g_score_last, g_score_prev, g_score_delta, degradation_flag
    """
    if mohanram_hist is None or mohanram_hist.empty:
        return {
            'g_score_last': np.nan,
            'g_score_prev': np.nan,
            'g_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    df = mohanram_hist[mohanram_hist['symbol'].str.upper() == symbol.upper()].copy()

    if df.empty:
        return {
            'g_score_last': np.nan,
            'g_score_prev': np.nan,
            'g_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')

    if len(df) < 2:
        g_last = float(df['G_SCORE'].iloc[-1]) if len(df) == 1 else np.nan
        return {
            'g_score_last': g_last,
            'g_score_prev': np.nan,
            'g_score_delta': np.nan,
            'degradation_flag': 'N/A'
        }

    g_last = float(df['G_SCORE'].iloc[-1])
    g_prev = float(df['G_SCORE'].iloc[-2])
    delta = g_last - g_prev

    if delta <= -degradation_threshold:
        flag = 'Degrading'
    elif delta >= degradation_threshold:
        flag = 'Improving'
    else:
        flag = 'Flat'

    return {
        'g_score_last': g_last,
        'g_score_prev': g_prev,
        'g_score_delta': delta,
        'degradation_flag': flag
    }


def interpret_gscore(score: float) -> str:
    """
    Interpret G-Score for growth stocks.

    Args:
        score: G-Score (0-8)

    Returns:
        Interpretation
    """
    if np.isnan(score):
        return "N/A"

    score = int(score)

    if score >= 7:
        return "Excellent"
    elif score >= 6:
        return "Strong"
    elif score >= 4:
        return "Above Average"
    elif score >= 2:
        return "Below Average"
    else:
        return "Weak"


def classify_value_vs_growth(
    symbols: List[str],
    api_key: str,
    bm_threshold: float = 0.5
) -> Dict[str, str]:
    """
    Classify stocks as Value or Growth based on B/M ratio.

    Value: Use Piotroski F-Score
    Growth: Use Mohanram G-Score

    Args:
        symbols: List of tickers
        api_key: FMP API key
        bm_threshold: Threshold for B/M (default 0.5, lower = growth)

    Returns:
        Dict mapping symbol to 'VALUE' or 'GROWTH'
    """
    from qvm_trend.fquality.fmp_quality import FMP_BASE, _get

    classification = {}

    for sym in symbols:
        try:
            # Get market cap and book value
            profile = _get(f"{FMP_BASE}/profile/{sym}", {"apikey": api_key})
            if not profile or not isinstance(profile, list):
                classification[sym] = 'UNKNOWN'
                continue

            market_cap = profile[0].get('mktCap', 0)
            if market_cap == 0:
                classification[sym] = 'UNKNOWN'
                continue

            # Get balance sheet for book value
            bal_annual = _get(f"{FMP_BASE}/balance-sheet-statement/{sym}", {"apikey": api_key, "period": "annual", "limit": 1})
            if not bal_annual or not isinstance(bal_annual, list):
                classification[sym] = 'UNKNOWN'
                continue

            equity = bal_annual[0].get('totalStockholdersEquity', 0)
            if equity <= 0:
                classification[sym] = 'GROWTH'  # Negative equity → growth
                continue

            # B/M ratio
            bm_ratio = equity / market_cap

            if bm_ratio >= bm_threshold:
                classification[sym] = 'VALUE'
            else:
                classification[sym] = 'GROWTH'

            time.sleep(SLEEP)

        except Exception as e:
            print(f"Error classifying {sym}: {e}")
            classification[sym] = 'UNKNOWN'

    return classification
