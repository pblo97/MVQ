from typing import Dict, Tuple
from .factors import BreakoutFeatures
from __future__ import annotations
import pandas as pd
import numpy as np
from .factors_growth_aware import compute_qvm_scores, apply_megacap_rules

DEFAULT_TH = {
    "rvol_min": 1.5,
    "closepos_min": 0.60,
    "p52_min": 0.95,
    "ud_vol_min": 1.2,
    "rs_slope_min": 0.0,
    "atr_pct_min": 0.6,
    "float_vel_min": 0.01
}

WEIGHTS = {
    # ponderaciones (puedes ajustarlas desde la UI si quieres)
    "RVOL": 2.0,
    "ClosePos": 2.0,
    "P52": 1.5,
    "TSMOM20": 1.0,
    "TSMOM63": 1.0,
    "MA20_slope": 1.0,
    "OBV_slope20": 1.0,
    "ADL_slope20": 1.0,
    "UDVolRatio20": 1.0,
    "RS_MA20_slope": 1.0,
    "ATR_pct": 1.0,
    "GapHold": 1.0,
    "FloatVelocity": 1.0,
}


def breakout_score(feat: BreakoutFeatures, th: Dict, weights: Dict = WEIGHTS) -> Tuple[float, Dict[str, bool]]:
    f = feat
    tests = {
        "RVOL": f.rvol20 >= th["rvol_min"],
        "ClosePos": f.closepos >= th["closepos_min"],
        "P52": f.p52 >= th["p52_min"],
        "TSMOM20": f.tsmom20 > 0,
        "TSMOM63": f.tsmom63 > 0,
        "MA20_slope": (f.ma20_slope if f.ma20_slope is not None else -1) > 0,
        "OBV_slope20": (f.obv_slope20 if f.obv_slope20 is not None else -1) > 0,
        "ADL_slope20": (f.adl_slope20 if f.adl_slope20 is not None else -1) > 0,
        "UDVolRatio20": f.updown_vol_ratio20 >= th["ud_vol_min"],
        "RS_MA20_slope": (f.rs_ma20_slope if f.rs_ma20_slope is not None else -1) > th["rs_slope_min"],
        "ATR_pct": f.atr_pct_rank >= th["atr_pct_min"],
        "GapHold": bool(f.gap_hold)
    }
    if f.float_velocity is not None:
        tests["FloatVelocity"] = f.float_velocity >= th["float_vel_min"]

    # score ponderado
    w_sum = 0.0
    s_sum = 0.0
    for k, ok in tests.items():
        w = float(weights.get(k, 1.0))
        w_sum += w
        s_sum += (w if ok else 0.0)
    score = s_sum / w_sum if w_sum > 0 else 0.0
    return float(score), tests


def entry_signal(score: float, tests: Dict[str, bool], min_score=0.6) -> bool:
    core_ok = tests.get("RVOL", False) and tests.get("ClosePos", False) and tests.get("P52", False)
    return (score >= min_score) and core_ok


def _z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

__all__ = [
    "build_momentum_proxy",
    "blend_breakout_qvm",
]

EPS = 1e-12

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + EPS)

def _rank01(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.divide(a, b, out=np.zeros_like(pd.Series(a, copy=False), dtype=float),
                     where=np.isfinite(b) & (b != 0))

def build_momentum_proxy(
    prices: pd.DataFrame,
    *,
    price_col: str = "close",
    id_col: str = "ticker",
    date_col: str = "date",
    w_short: float = 0.20,
    w_med: float = 0.30,
    w_long: float = 0.50,
    short_win: int = 63,
    med_win: int = 126,
    long_win: int = 252,
    vol_win: int = 63,
) -> pd.Series:
    """
    Calcula un score de momentum por activo (multi-horizonte con penalización por volatilidad).
    Devuelve una Series indexada por <id_col> llamada 'momentum_score' (z-score).
    Acepta:
      - DF "largo": columnas [id_col, date_col, price_col]
      - DF indexado MultiIndex [id_col, date_col] + una columna de precios
    """
    df = prices.copy()

    # Normaliza a formato largo si viene como MultiIndex
    if id_col not in df.columns and isinstance(df.index, pd.MultiIndex):
        df = df.reset_index().rename(columns={df.columns[0]: id_col, df.columns[1]: date_col})
        if price_col not in df.columns:
            value_cols = [c for c in df.columns if c not in (id_col, date_col)]
            if len(value_cols) != 1:
                raise ValueError("No se puede inferir price_col; especifícalo.")
            price_col = value_cols[0]

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([id_col, date_col])

    def per_id(g: pd.DataFrame) -> float:
        px = pd.to_numeric(g[price_col], errors="coerce").astype(float)
        if px.isna().all() or len(px) < min(short_win, med_win, long_win) + 1:
            return np.nan
        rets = np.log(px).diff()

        def cumret(n):
            r = rets.rolling(n).sum().iloc[-1]
            return float(r) if np.isfinite(r) else np.nan

        r_short = cumret(short_win)
        r_med   = cumret(med_win)
        r_long  = cumret(long_win)

        vol = float(rets.rolling(vol_win).std(ddof=0).iloc[-1]) if len(rets) >= vol_win else np.nan

        raw = (w_short * r_short) + (w_med * r_med) + (w_long * r_long)
        if np.isfinite(vol) and vol > 0:
            raw = raw / (vol + EPS)  # tipo Sharpe
        return raw

    scores = df.groupby(id_col, sort=False, group_keys=False).apply(per_id).astype(float)
    scores.name = "momentum_score"
    # Entrega z-score para facilitar el blend posterior
    return _zscore(scores)

def blend_breakout_qvm(
    df: pd.DataFrame,
    *,
    col_qvm: str = "qvm_score",
    col_breakout: str = "breakout_score",
    w_qvm: float = 0.60,
    w_breakout: float = 0.40,
    to_percentile: bool = True,
) -> pd.Series:
    """
    Mezcla un score QVM con un score de breakout.
    Estandariza ambos a z-score y hace un blend ponderado.
    Retorna percentil [0,1] si to_percentile=True (útil para ranking global).
    """
    if col_qvm not in df.columns or col_breakout not in df.columns:
        raise KeyError(f"Faltan columnas '{col_qvm}' y/o '{col_breakout}' en df.")

    z_qvm = _zscore(df[col_qvm])
    z_bo  = _zscore(df[col_breakout])

    blended = (w_qvm * z_qvm) + (w_breakout * z_bo)
    blended.name = "blended_qvm_breakout"
    return _rank01(blended) if to_percentile else blended