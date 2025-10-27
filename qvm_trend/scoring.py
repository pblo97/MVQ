from __future__ import annotations
from typing import Dict, Tuple
from .factors import BreakoutFeatures

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

try:
    from .factors_growth_aware import (
        compute_qvm_scores,
        apply_megacap_rules,
    )
    _HAS_GA = True
except Exception:
    _HAS_GA = False

# ----------------------------- Utils ----------------------------- #
def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def _rank_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True)

def _neutralize_by_sector_cap(df: pd.DataFrame, score_col: str,
                              sector_col: str = "sector",
                              mcap_col: str = "market_cap",
                              buckets=(("Mega", 150e9, np.inf),
                                       ("Large", 10e9, 150e9),
                                       ("Mid", 2e9, 10e9),
                                       ("Small", 0, 2e9))) -> pd.Series:
    out = df.copy()
    out[mcap_col] = pd.to_numeric(out.get(mcap_col), errors="coerce")
    # buckets
    edges = [b[1] for b in buckets] + [buckets[-1][2]]
    labels = [b[0] for b in buckets]
    out["_cap_bucket"] = pd.cut(out[mcap_col].astype(float), bins=edges, labels=labels, include_lowest=True, right=False)
    # z por grupo
    z_sector = out.groupby(sector_col, group_keys=False)[score_col].apply(_zscore)
    z_cap    = out.groupby("_cap_bucket", group_keys=False)[score_col].apply(_zscore)
    return 0.5*z_sector + 0.5*z_cap

# ----------------------- Momentum proxy -------------------------- #
def build_momentum_proxy(df_sig: pd.DataFrame) -> pd.Series:
    """
    Proxy de momentum (0..~) usando señales técnicas si no traes serie propia.
    Usa columnas si existen: ClosePos (↑), P52 (↑), rs_ma20_slope (↑), hits (↑)
    Devuelve una Serie indexada por *symbol*.
    """
    if df_sig is None or not isinstance(df_sig, pd.DataFrame) or df_sig.empty:
        return pd.Series(dtype=float)

    x = df_sig.copy()
    if "symbol" not in x.columns:
        # Sin símbolo, devolvemos z de ClosePos global como fallback
        cols = [c for c in ["ClosePos", "P52", "rs_ma20_slope", "hits"] if c in x.columns]
        s = sum(_zscore(x[c]) for c in cols) / max(1, len(cols))
        return pd.Series(s, index=x.index, dtype=float)

    parts = []
    if "ClosePos" in x.columns:       parts.append(_zscore(x["ClosePos"]))
    if "P52" in x.columns:            parts.append(_zscore(x["P52"]))
    if "rs_ma20_slope" in x.columns:  parts.append(_zscore(x["rs_ma20_slope"]))
    if "hits" in x.columns:           parts.append(_zscore(x["hits"]))
    if not parts:
        mom = pd.Series(0.0, index=x.index)
    else:
        mom = sum(parts) / len(parts)

    # promedio por símbolo (por si hay duplicados)
    mom_sym = pd.DataFrame({"symbol": x["symbol"], "mom": mom}).groupby("symbol", as_index=True)["mom"].mean()
    return mom_sym

# -------------------- QVM + Breakout blender -------------------- #
def blend_breakout_qvm(base: pd.DataFrame,
                       breakout_col: str = "BreakoutScore",
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap",
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       w_breakout: float = 0.30) -> pd.DataFrame:
    """
    Crea un score QVM (growth-aware si hay datos) y lo mezcla con Breakout.
    - Si existen columnas 'value_adj_neut' y 'quality_adj_neut', las usa.
    - Si no, intenta usar 'ValueScore' y 'QualityScore' (neutralizando por sector+cap).
    - Si hay fundamentales suficientes y factors_growth_aware disponible, calcula value/quality growth-aware.
    Devuelve copia de base con columnas:
      ['value_adj_neut','quality_adj_neut','qvm_score','final_alpha',
       'mega_exception_ok','quality_too_low'] si aplica.
    """
    if base is None or not isinstance(base, pd.DataFrame) or base.empty:
        return pd.DataFrame()

    df = base.copy()

    # Asegurar columnas clave
    if sector_col not in df.columns and "sector_vfq" in df.columns:
        df[sector_col] = df["sector_vfq"]
    if mcap_col not in df.columns:
        for alt in ("marketCap_unified", "marketCap"):
            if alt in df.columns:
                df[mcap_col] = pd.to_numeric(df[alt], errors="coerce")
                break

    # Momentum z
    if momentum_col not in df.columns:
        # intentar construir desde señales si trajeron columnas conocidas
        mom_proxy = build_momentum_proxy(df if "symbol" in df.columns else pd.DataFrame())
        if "symbol" in df.columns and not mom_proxy.empty:
            df = df.merge(mom_proxy.rename("momentum_score"), left_on="symbol", right_index=True, how="left")
        else:
            df[momentum_col] = 0.0
    m_z = _zscore(pd.to_numeric(df[momentum_col], errors="coerce").fillna(0.0))

    # Value/Quality growth-aware si es posible
    have_adj = {"value_adj_neut","quality_adj_neut"}.issubset(df.columns)
    if not have_adj and _HAS_GA:
        try:
            # compute_qvm_scores calcula value/quality adj + neut (no usa breakout)
            tmp = compute_qvm_scores(
                df.rename(columns={mcap_col: "market_cap", sector_col: "sector"}),
                w_quality=0.40, w_value=0.25, w_momentum=0.35,
                momentum_col=momentum_col, sector_col="sector", mcap_col="market_cap"
            )
            for c in ("value_adj_neut","quality_adj_neut","value_adj","quality_adj","qvm_score"):
                if c in tmp.columns:
                    df[c] = tmp[c]
            have_adj = {"value_adj_neut","quality_adj_neut"}.issubset(df.columns)
        except Exception:
            have_adj = False

    # Fallback a VFQ si sigue faltando
    if not have_adj:
        if "ValueScore" in df.columns:
            df["value_adj_neut"] = _neutralize_by_sector_cap(
                df.rename(columns={sector_col: "sector", mcap_col: "market_cap"}),
                score_col="ValueScore", sector_col="sector", mcap_col="market_cap"
            )
        else:
            df["value_adj_neut"] = 0.0

        if "QualityScore" in df.columns:
            df["quality_adj_neut"] = _neutralize_by_sector_cap(
                df.rename(columns={sector_col: "sector", mcap_col: "market_cap"}),
                score_col="QualityScore", sector_col="sector", mcap_col="market_cap"
            )
        else:
            df["quality_adj_neut"] = 0.0

    # QVM score
    qvm_score = (
        w_quality  * _zscore(df["quality_adj_neut"]) +
        w_value    * _zscore(df["value_adj_neut"]) +
        w_momentum * m_z
    )
    df["qvm_score"] = qvm_score

    # Breakout z y mezcla final
    b_z = _zscore(pd.to_numeric(df.get(breakout_col, 0.0), errors="coerce").fillna(0.0))
    df["final_alpha"] = (1 - w_breakout) * qvm_score + w_breakout * b_z

    # Guardrails mega-cap si están disponibles
    if _HAS_GA:
        try:
            flags = apply_megacap_rules(
                df.rename(columns={sector_col: "sector", mcap_col: "market_cap"}),
                momentum_col=momentum_col,
                quality_col="quality_adj_neut",
                value_col="value_adj_neut",
            )[["mega_exception_ok","quality_too_low","q_pct_sector","v_pct_sector","m_pct_global"]]
            df = df.merge(flags, left_index=True, right_index=True, how="left")
        except Exception:
            df["mega_exception_ok"] = False
            df["quality_too_low"] = False
    else:
        df["mega_exception_ok"] = False
        df["quality_too_low"] = False

    return df