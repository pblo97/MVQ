# portfolio_manager/monitor/exits_enhanced.py
"""
Enhanced Exit Monitoring with Piotroski F-Score Integration

Extends qvm_trend/pm/exits.py with robust fundamental degradation detection
using Piotroski F-Score (academic best practice).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
import os

# Import original exit system
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from qvm_trend.pm.exits import _ma, _mom_12_1, _next_q_end

# Import Piotroski module
from portfolio_manager.fundamentals.piotroski import detect_fundamental_degradation


def build_exit_table_enhanced(
    *,
    panel: Dict[str, pd.DataFrame],
    bench_close: Optional[pd.Series] = None,
    ma_window: int = 200,
    mom_lookback: int = 252,
    review_freq: str = "Q",
    # Fundamental degradation options
    piotroski_hist: Optional[pd.DataFrame] = None,  # History of Piotroski F-Scores
    vfq_hist: Optional[pd.DataFrame] = None,         # Legacy VFQ history (fallback)
    use_piotroski: bool = True,                      # Use Piotroski (default) vs VFQ
    degradation_threshold: int = 2,                  # F-Score drop indicating degradation
    vfq_col: str = "VFQ",
    vfq_delta_thr: float = 0.10
) -> pd.DataFrame:
    """
    Enhanced exit table with Piotroski F-Score fundamental degradation detection.

    Columnas:
        symbol, price_last, MA200, ma_flag, Mom12-1, mom_flag,
        f_score_last, f_score_prev, f_score_delta, fundamental_flag,
        reason, action, next_review

    Reglas:
        - ma_flag: close < MA200
        - mom_flag: momentum 12-1 < 0
        - fundamental_flag: 'Degrading' si ΔF-Score ≤ -2; 'Improving' si ≥ +2; else 'Flat'/'N/A'
        - Acción:
            EXIT si (ma_flag AND mom_flag) OR (ma_flag AND fundamental=='Degrading')
            TRIM si ma_flag OR mom_flag OR fundamental=='Degrading'
            HOLD en caso contrario

    Args:
        panel: Dict de DataFrames con precios por símbolo
        bench_close: Serie de precios benchmark (reservado)
        ma_window: Ventana para MA (default 200)
        mom_lookback: Lookback momentum (default 252)
        review_freq: Frecuencia de revisión ('Q' = quarterly)
        piotroski_hist: DataFrame con ['symbol', 'date', 'F_SCORE']
        vfq_hist: DataFrame con ['symbol', 'date', 'VFQ'] (fallback si no hay Piotroski)
        use_piotroski: Si True, usa Piotroski; si False, usa VFQ
        degradation_threshold: Caída en F-Score que indica degradación (default 2)
        vfq_col: Nombre columna VFQ (si se usa)
        vfq_delta_thr: Threshold VFQ (si se usa)

    Returns:
        DataFrame con exit signals
    """
    rows = []
    if not panel:
        return pd.DataFrame()

    for sym, df in panel.items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        px = pd.to_numeric(df["close"], errors="coerce").dropna()
        if px.empty:
            continue

        # ========== TECHNICAL SIGNALS ==========

        # MA flag
        ma_val = _ma(px, ma_window)
        ma_flag = bool(np.isfinite(ma_val) and px.iloc[-1] < ma_val)

        # Momentum flag
        mom_val = _mom_12_1(px, lb_12m=mom_lookback, lb_1m=21)
        mom_flag = bool(np.isfinite(mom_val) and mom_val < 0)

        # ========== FUNDAMENTAL DEGRADATION ==========

        if use_piotroski and piotroski_hist is not None and not piotroski_hist.empty:
            # Use Piotroski F-Score
            fund_info = detect_fundamental_degradation(
                piotroski_hist=piotroski_hist,
                symbol=sym,
                degradation_threshold=degradation_threshold
            )

            f_last = fund_info['f_score_last']
            f_prev = fund_info['f_score_prev']
            f_delta = fund_info['f_score_delta']
            fund_flag = fund_info['degradation_flag']

        else:
            # Fallback to VFQ (legacy system)
            if vfq_hist is not None and not vfq_hist.empty:
                from qvm_trend.pm.exits import _vfq_trend
                vfq_info = _vfq_trend(vfq_hist, sym, score_col=vfq_col, delta_thr=vfq_delta_thr)

                f_last = vfq_info.get('vfq_last', np.nan)
                f_delta = vfq_info.get('vfq_chg_1q', np.nan)
                f_prev = float(f_last - f_delta) if np.isfinite(f_last) and np.isfinite(f_delta) else np.nan
                fund_flag = vfq_info.get('vfq_trend', 'N/A')
            else:
                # No fundamental data
                f_last = np.nan
                f_prev = np.nan
                f_delta = np.nan
                fund_flag = 'N/A'

        # ========== EXIT LOGIC ==========

        reasons = []
        if ma_flag:
            reasons.append(f"Close<{ma_window}MA")
        if mom_flag:
            reasons.append("Momentum 12-1 < 0")
        if fund_flag == "Degrading":
            signal_type = "Piotroski" if (use_piotroski and piotroski_hist is not None) else "VFQ"
            reasons.append(f"Fundamentals ↓ ({signal_type})")

        # Action rules
        action = "HOLD"
        if (ma_flag and mom_flag) or (ma_flag and fund_flag == "Degrading"):
            action = "EXIT"
        elif ma_flag or mom_flag or (fund_flag == "Degrading"):
            action = "TRIM"

        # Next review date
        today = px.index[-1]
        next_review = _next_q_end(today) if review_freq.upper().startswith("Q") else today

        # ========== BUILD ROW ==========

        rows.append({
            'symbol': sym,
            'price_last': float(px.iloc[-1]),
            'MA200': float(ma_val) if np.isfinite(ma_val) else np.nan,
            'ma_flag': ma_flag,
            'Mom12-1': float(mom_val) if np.isfinite(mom_val) else np.nan,
            'mom_flag': mom_flag,
            'f_score_last': float(f_last) if np.isfinite(f_last) else np.nan,
            'f_score_prev': float(f_prev) if np.isfinite(f_prev) else np.nan,
            'f_score_delta': float(f_delta) if np.isfinite(f_delta) else np.nan,
            'fundamental_flag': fund_flag,
            'next_review': next_review.date(),
            'reason': "; ".join(reasons) if reasons else "—",
            'action': action,
        })

    # Build DataFrame
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl

    # Sort by priority: EXIT > TRIM > HOLD
    order = {"EXIT": 0, "TRIM": 1, "HOLD": 2}
    tbl["priority"] = tbl["action"].map(order).fillna(3)
    tbl = (
        tbl.sort_values(["priority", "symbol"])
           .drop(columns=["priority"])
           .reset_index(drop=True)
    )

    return tbl
