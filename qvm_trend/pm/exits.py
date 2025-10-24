# qvm_trend/pm/exits.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


def _ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def _mom_12_1(close: pd.Series) -> pd.Series:
    """
    Momento 12-1 clásico: retorno 12m excluyendo el último mes.
    Señal 'negativa' cuando < 0.
    """
    mret = close.pct_change()
    # retorno acumulado últimos 12m
    r12 = (1.0 + mret).rolling(252, min_periods=252).apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    # excluir el último mes ~21d
    r1  = (1.0 + mret).rolling(21,  min_periods=21 ).apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    return (r12 - r1).rename("mom_12_1")


def _quarter_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return pd.DatetimeIndex([])
    # fin de trimestre naturales
    qe = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="Q")
    # ajusta al índice disponible (tomar el último día disponible <= fin de trimestre)
    out = []
    for d in qe:
        s = idx[idx <= d]
        if len(s) > 0:
            out.append(s[-1])
    return pd.DatetimeIndex(out).unique()


def build_exit_table(
    panel: Dict[str, pd.DataFrame],
    bench_close: Optional[pd.Series] = None,
    *,
    ma_window: int = 200,
    mom_lookback: int = 252,
    review_freq: str = "Q",   # por ahora soportado: 'Q'
) -> pd.DataFrame:
    """
    Devuelve tabla de salida por símbolo con:
      symbol, last_date, close, ma200, below_ma200, mom_12_1, mom_neg, exit_flag, reason
    Reglas:
      - Revisión por trimestre (último día hábil del trimestre).
      - Señal de salida si (close < MA200) o (mom_12_1 < 0) en la revisión.
    """
    rows = []
    for sym, df in (panel or {}).items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        px = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(px) < max(ma_window, mom_lookback) + 5:
            # poca historia — no emitimos señal
            rows.append({
                "symbol": sym, "last_date": px.index.max() if len(px) else pd.NaT,
                "close": float(px.iloc[-1]) if len(px) else np.nan,
                "ma200": np.nan, "below_ma200": False,
                "mom_12_1": np.nan, "mom_neg": False,
                "exit_flag": False, "reason": "historia insuficiente"
            })
            continue

        ma = _ma(px, ma_window)
        mom = _mom_12_1(px)

        # fecha de revisión (último fin de trimestre disponible)
        qends = _quarter_ends(px.index)
        if len(qends) == 0:
            review_dt = px.index.max()
        else:
            review_dt = qends[-1]

        # tomar valores a la fecha de revisión
        c_rev   = float(px.reindex(px.index).ffill().loc[review_dt])
        ma_rev  = float(ma.reindex(px.index).ffill().loc[review_dt]) if not ma.dropna().empty else np.nan
        mom_rev = float(mom.reindex(px.index).ffill().loc[review_dt]) if not mom.dropna().empty else np.nan

        below = bool(c_rev < ma_rev) if np.isfinite(ma_rev) else False
        mneg  = bool(mom_rev < 0.0)   if np.isfinite(mom_rev) else False

        reason = []
        if below: reason.append("close<MA200")
        if mneg:  reason.append("Mom12-1<0")

        rows.append({
            "symbol": sym,
            "last_date": review_dt,
            "close": c_rev,
            "ma200": ma_rev,
            "below_ma200": below,
            "mom_12_1": mom_rev,
            "mom_neg": mneg,
            "exit_flag": bool(below or mneg),
            "reason": " & ".join(reason) if reason else "mantener",
        })

    out = pd.DataFrame(rows).sort_values(["exit_flag","symbol"], ascending=[False, True])
    return out.reset_index(drop=True)
