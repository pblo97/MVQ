from typing import Dict, Tuple
from .factors import BreakoutFeatures

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



