from __future__ import annotations
import numpy as np
import pandas as pd

def _q_end(d: pd.Timestamp) -> pd.Timestamp:
    """Fin de trimestre del timestamp dado."""
    q = (d.month - 1)//3 + 1
    last_month = q*3
    last_day = pd.Period(f"{d.year}-{last_month:02d}").days_in_month
    return pd.Timestamp(d.year, last_month, last_day)

def _next_q_end(d: pd.Timestamp) -> pd.Timestamp:
    qe = _q_end(d)
    return qe if d <= qe else _q_end(d + pd.offsets.QuarterEnd())

def _mom_12_1(px: pd.Series, lb_12m: int = 252, lb_1m: int = 21) -> float:
    px = pd.to_numeric(px, errors="coerce").dropna()
    if len(px) < max(lb_12m, lb_1m)+1: 
        return np.nan
    r12 = px.iloc[-1]/px.iloc[-1-lb_12m] - 1.0
    r1  = px.iloc[-1]/px.iloc[-1-lb_1m]  - 1.0
    return float(r12 - r1)

def _ma(px: pd.Series, win: int = 200) -> float:
    px = pd.to_numeric(px, errors="coerce").dropna()
    if len(px) < win: 
        return np.nan
    return float(px.rolling(win).mean().iloc[-1])

def _vfq_trend(vfq_hist: pd.DataFrame, symbol: str, score_col: str = "VFQ", delta_thr: float = 0.10) -> dict:
    """
    vfq_hist: DataFrame con columnas ['symbol','date', score_col] (date parseable)
    Devuelve: {'vfq_last':..., 'vfq_chg_1q':..., 'vfq_trend':'Improving/Degrading/Flat'}
    """
    if vfq_hist is None or vfq_hist.empty or score_col not in vfq_hist.columns:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")
    df = vfq_hist[vfq_hist["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if df.empty:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")
    # index temporal trimestral
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
        df = df.assign(date=idx).dropna(subset=["date"]).set_index("date").sort_index()
    else:
        df = df.sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")
    q = df[[score_col]].resample("Q").last().dropna()
    if q.shape[0] < 2:
        return dict(vfq_last=float(q.iloc[-1][score_col]) if not q.empty else np.nan,
                    vfq_chg_1q=np.nan, vfq_trend="N/A")
    last = float(q.iloc[-1][score_col])
    prev = float(q.iloc[-2][score_col])
    d1 = last - prev
    if d1 > delta_thr:
        trend = "Improving"
    elif d1 < -delta_thr:
        trend = "Degrading"
    else:
        trend = "Flat"
    return dict(vfq_last=last, vfq_chg_1q=d1, vfq_trend=trend)

def build_exit_table(
    *,
    panel: dict[str, pd.DataFrame],
    bench_close: pd.Series | None = None,
    ma_window: int = 200,
    mom_lookback: int = 252,
    review_freq: str = "Q",
    vfq_hist: pd.DataFrame | None = None,   # opcional histórico de calidad
    vfq_col: str = "VFQ",
    vfq_delta_thr: float = 0.10,            # umbral mejora/degradación 1 trimestre
) -> pd.DataFrame:
    """
    Devuelve una tabla por símbolo con señales de salida y recomendación:
    columnas:
      ['symbol','close','MA200','MA_breach','Mom12-1','Mom_neg',
       'vfq_last','vfq_chg_1q','vfq_trend','review_next','action','reasons']
    Reglas:
      - MA_breach: close < MA200
      - Mom_neg: (12-1) < 0
      - Calidad: 'Degrading' si ΔVFQ_1Q < -vfq_delta_thr
      - Acción:
           EXIT si (MA_breach y Mom_neg) o (MA_breach y Degrading)
           TRIM si (MA_breach) o (Mom_neg) o (Degrading)
           HOLD en caso contrario
    """
    rows = []
    for sym, df in (panel or {}).items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        px = pd.to_numeric(df["close"], errors="coerce").dropna()
        if px.empty:
            continue

        ma = _ma(px, ma_window)
        ma_breach = bool(px.iloc[-1] < ma) if np.isfinite(ma) else False
        mom = _mom_12_1(px, lb_12m=mom_lookback, lb_1m=21)
        mom_neg = bool(np.isfinite(mom) and mom < 0)

        qinfo = _vfq_trend(vfq_hist, sym, score_col=vfq_col, delta_thr=vfq_delta_thr)

        reasons = []
        if ma_breach: reasons.append(f"Close<{ma_window}MA")
        if mom_neg:  reasons.append("Momentum 12-1 < 0")
        if qinfo["vfq_trend"] == "Degrading": reasons.append("Calidad ↓ (1Q)")

        # Acción
        action = "HOLD"
        if (ma_breach and mom_neg) or (ma_breach and qinfo["vfq_trend"] == "Degrading"):
            action = "EXIT"
        elif ma_breach or mom_neg or (qinfo["vfq_trend"] == "Degrading"):
            action = "TRIM"

        today = px.index[-1]
        review_next = _next_q_end(today) if review_freq.upper().startswith("Q") else today

        rows.append(dict(
            symbol=sym,
            close=float(px.iloc[-1]),
            MA200=float(ma) if np.isfinite(ma) else np.nan,
            MA_breach=ma_breach,
            **{"Mom12-1": float(mom) if np.isfinite(mom) else np.nan},
            Mom_neg=mom_neg,
            vfq_last=float(qinfo["vfq_last"]) if np.isfinite(qinfo["vfq_last"]) else np.nan,
            vfq_chg_1q=float(qinfo["vfq_chg_1q"]) if np.isfinite(qinfo["vfq_chg_1q"]) else np.nan,
            vfq_trend=qinfo["vfq_trend"],
            review_next=review_next.date(),
            action=action,
            reasons="; ".join(reasons) if reasons else "—",
        ))

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl

    # Prioriza EXIT > TRIM > HOLD
    order = {"EXIT": 0, "TRIM": 1, "HOLD": 2}
    tbl["priority"] = tbl["action"].map(order).fillna(3)
    tbl = tbl.sort_values(["priority", "symbol"]).drop(columns=["priority"]).reset_index(drop=True)
    return tbl