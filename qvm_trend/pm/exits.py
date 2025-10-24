# qvm_trend/pm/exits.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

# =========================
# Helpers de calendario
# =========================
def _q_end(d: pd.Timestamp) -> pd.Timestamp:
    """Fin de trimestre del timestamp dado."""
    d = pd.Timestamp(d)
    q = (d.month - 1)//3 + 1
    last_month = q * 3
    last_day = pd.Period(f"{d.year}-{last_month:02d}").days_in_month
    return pd.Timestamp(d.year, last_month, last_day)

def _next_q_end(d: pd.Timestamp) -> pd.Timestamp:
    """Próximo fin de trimestre (incluye el actual si aún no terminó)."""
    d = pd.Timestamp(d)
    qe = _q_end(d)
    return qe if d <= qe else _q_end(d + pd.offsets.QuarterEnd())

# =========================
# Señales técnicas
# =========================
def _mom_12_1(px: pd.Series, lb_12m: int = 252, lb_1m: int = 21) -> float:
    """Momentum 12-1 clásico (anual menos último mes)."""
    px = pd.to_numeric(px, errors="coerce").dropna()
    if len(px) < max(lb_12m, lb_1m) + 1:
        return np.nan
    try:
        r12 = px.iloc[-1] / px.iloc[-1 - lb_12m] - 1.0
        r1  = px.iloc[-1] / px.iloc[-1 - lb_1m]  - 1.0
        return float(r12 - r1)
    except Exception:
        return np.nan

def _ma(px: pd.Series, win: int = 200) -> float:
    """Media móvil simple."""
    px = pd.to_numeric(px, errors="coerce").dropna()
    if len(px) < win:
        return np.nan
    return float(px.rolling(win).mean().iloc[-1])

# =========================
# Calidad (VFQ/Quality) 1Q
# =========================
def _match_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """Devuelve el nombre de columna que coincide (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _vfq_trend(
    vfq_hist: Optional[pd.DataFrame],
    symbol: str,
    score_col: str = "VFQ",
    delta_thr: float = 0.10
) -> dict:
    """
    vfq_hist: DataFrame con columnas ['symbol','date', score_col] (date parseable)
    Devuelve:
      {'vfq_last': float, 'vfq_chg_1q': float, 'vfq_trend': 'Improving/Degrading/Flat/N/A'}
    """
    if vfq_hist is None or vfq_hist.empty:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    df = vfq_hist.copy()
    # Columnas tolerantes (symbol, date, score)
    sym_col = _match_col(df, ["symbol", "ticker"])
    if sym_col is None:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    sc_col = _match_col(df, [score_col, "qualityscore", "vq", "score", "vfq"])
    if sc_col is None:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    date_col = _match_col(df, ["date", "asof", "timestamp"])
    if date_col is None:
        # Permite que el índice ya sea datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Filtra símbolo
    mask = df[sym_col].astype(str).str.upper() == str(symbol).upper()
    df = df.loc[mask, [sc_col] + ([date_col] if date_col else [])].copy()
    if df.empty:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    # Index temporal
    if date_col:
        df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    df = df.sort_index()
    # Series trimestral (último de cada trimestre)
    q = df[[sc_col]].resample("Q").last().dropna()
    if q.empty:
        return dict(vfq_last=np.nan, vfq_chg_1q=np.nan, vfq_trend="N/A")

    vfq_last = float(pd.to_numeric(q[sc_col], errors="coerce").dropna().iloc[-1])
    if q.shape[0] < 2:
        return dict(vfq_last=vfq_last, vfq_chg_1q=np.nan, vfq_trend="N/A")

    vfq_prev = float(pd.to_numeric(q[sc_col], errors="coerce").dropna().iloc[-2])
    d1 = vfq_last - vfq_prev

    if d1 > delta_thr:
        trend = "Improving"
    elif d1 < -delta_thr:
        trend = "Degrading"
    else:
        trend = "Flat"

    return dict(vfq_last=vfq_last, vfq_chg_1q=float(d1), vfq_trend=trend)

# =========================
# Tabla principal
# =========================
def build_exit_table(
    *,
    panel: Dict[str, pd.DataFrame],
    bench_close: Optional[pd.Series] = None,  # reservado para futuras reglas relativas
    ma_window: int = 200,
    mom_lookback: int = 252,
    review_freq: str = "Q",
    vfq_hist: Optional[pd.DataFrame] = None,  # histórico de calidad (opcional)
    vfq_col: str = "VFQ",
    vfq_delta_thr: float = 0.10,              # umbral de deterioro/mejora 1Q
) -> pd.DataFrame:
    """
    Devuelve tabla por símbolo con señales y recomendación.
    Columnas esperadas:
      symbol, price_last, MA200, ma_flag, Mom12-1, mom_flag,
      vfq_last, vfq_prev, vfq_delta, quality_flag, reason, action, next_review

    Reglas:
      - ma_flag: close < MA200
      - mom_flag: (12-1) < 0
      - quality_flag: 'Degrading' si ΔVFQ_1Q < -vfq_delta_thr; 'Improving' si > +umbral; 'Flat' si en rango; 'N/A' si no hay histórico
      - Acción:
          EXIT si (ma_flag y mom_flag) o (ma_flag y quality=='Degrading')
          TRIM si (ma_flag) o (mom_flag) o (quality=='Degrading')
          HOLD en caso contrario
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

        # Señales técnicas
        ma_val = _ma(px, ma_window)
        ma_flag = bool(np.isfinite(ma_val) and px.iloc[-1] < ma_val)

        mom_val = _mom_12_1(px, lb_12m=mom_lookback, lb_1m=21)
        mom_flag = bool(np.isfinite(mom_val) and mom_val < 0)

        # Calidad (VFQ)
        qinfo = _vfq_trend(vfq_hist, sym, score_col=vfq_col, delta_thr=vfq_delta_thr)
        v_last = qinfo.get("vfq_last", np.nan)
        v_chg  = qinfo.get("vfq_chg_1q", np.nan)
        v_tr   = qinfo.get("vfq_trend", "N/A")

        # Para tabla: vfq_prev = last - delta (si delta válido)
        if np.isfinite(v_last) and np.isfinite(v_chg):
            v_prev = float(v_last - v_chg)
        else:
            v_prev = np.nan

        # Razones & Acción
        reasons = []
        if ma_flag:
            reasons.append(f"Close<{ma_window}MA")
        if mom_flag:
            reasons.append("Momentum 12-1 < 0")
        if v_tr == "Degrading":
            reasons.append("Calidad ↓ (1Q)")

        action = "HOLD"
        if (ma_flag and mom_flag) or (ma_flag and v_tr == "Degrading"):
            action = "EXIT"
        elif ma_flag or mom_flag or (v_tr == "Degrading"):
            action = "TRIM"

        today = px.index[-1]
        next_review = _next_q_end(today) if review_freq.upper().startswith("Q") else today

        rows.append(dict(
            symbol=sym,
            price_last=float(px.iloc[-1]),
            MA200=float(ma_val) if np.isfinite(ma_val) else np.nan,
            ma_flag=ma_flag,
            **{"Mom12-1": float(mom_val) if np.isfinite(mom_val) else np.nan},
            mom_flag=mom_flag,
            vfq_last=float(v_last) if np.isfinite(v_last) else np.nan,
            vfq_prev=float(v_prev) if np.isfinite(v_prev) else np.nan,
            vfq_delta=float(v_chg) if np.isfinite(v_chg) else np.nan,
            quality_flag=v_tr,
            next_review=next_review.date(),
            reason="; ".join(reasons) if reasons else "—",
            action=action,
        ))

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl

    # Orden de prioridad: EXIT > TRIM > HOLD
    order = {"EXIT": 0, "TRIM": 1, "HOLD": 2}
    tbl["priority"] = tbl["action"].map(order).fillna(3)
    tbl = (
        tbl.sort_values(["priority", "symbol"])
           .drop(columns=["priority"])
           .reset_index(drop=True)
    )
    return tbl
