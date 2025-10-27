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

def blend_breakout_qvm(df_base: pd.DataFrame,
                       breakout_col: str = "BreakoutScore",
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap",
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       w_breakout: float = 0.30) -> pd.DataFrame:
    """
    Mezcla QVM (growth-aware) con tu BreakoutScore.
    Devuelve df con:
      value_adj, quality_adj, *_neut, qvm_score,
      mega_exception_ok, quality_too_low,
      final_alpha = (1 - w_breakout)*z(qvm_score) + w_breakout*z(BreakoutScore)
    """
    req_cols = {sector_col, mcap_col, momentum_col, breakout_col}
    missing = [c for c in req_cols if c not in df_base.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas para QVM/blend: {missing}")

    qvm = compute_qvm_scores(
        df_base,
        w_quality=w_quality, w_value=w_value, w_momentum=w_momentum,
        momentum_col=momentum_col, sector_col=sector_col, mcap_col=mcap_col
    )
    qvm = apply_megacap_rules(qvm, momentum_col=momentum_col)
    qvm["final_alpha"] = (1 - w_breakout) * _z(qvm["qvm_score"]) + w_breakout * _z(qvm[breakout_col])
    return qvm

def build_momentum_proxy(df_sig: pd.DataFrame) -> pd.Series:
    """
    Proxy simple de momentum si no traes 12-1:
     40% ClosePos + 40% P52 + 20% slope RS (si existe)
    """
    def _get(c): return pd.to_numeric(df_sig.get(c), errors="coerce")
    closepos = _get("ClosePos")
    p52 = _get("P52")
    rs_slope = _get("rs_ma20_slope") if "rs_ma20_slope" in df_sig.columns else pd.Series(index=df_sig.index, data=np.nan)
    comp = 0.40*closepos.fillna(closepos.median()) + 0.40*p52.fillna(p52.median()) + 0.20*rs_slope.fillna(0.0)
    return (comp - comp.mean()) / (comp.std(ddof=0) + 1e-12)
