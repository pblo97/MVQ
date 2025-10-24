# qvm_trend/fundamentals/fmp_quality.py
from __future__ import annotations
import math, time
import numpy as np
import pandas as pd
import requests
from typing import List, Dict

FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ---------- helpers ----------
def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.clip(s.quantile(0.02), s.quantile(0.98))  # winsor 2%
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)

def _safe(series, key_list, default=np.nan):
    for k in key_list:
        if k in series:
            return series.get(k)
    return default

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _get_json(url, params, retry=2, sleep=0.5):
    for t in range(retry+1):
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(sleep * (t+1))
    return None

# ---------- fetchers (TTM / growth / EPS series) ----------
def fetch_ttm(symbols: List[str], api_key: str) -> Dict[str, dict]:
    """
    Devuelve dict[symbol] -> dict con campos relevantes (mezcla de ratios-ttm y key-metrics-ttm).
    """
    out = {s: {} for s in symbols}
    for chunk in _chunk(symbols, 20):
        params = {"apikey": api_key, "symbol": ",".join(chunk)}
        ratios = _get_json(f"{FMP_BASE}/ratios-ttm", params)
        km     = _get_json(f"{FMP_BASE}/key-metrics-ttm", params)
        if isinstance(ratios, list):
            for row in ratios:
                s = row.get("symbol")
                if s in out:
                    out[s].update(row)
        if isinstance(km, list):
            for row in km:
                s = row.get("symbol")
                if s in out:
                    out[s].update(row)
        time.sleep(0.12)  # respeta rate limit
    return out

def fetch_growth(symbols: List[str], api_key: str) -> Dict[str, dict]:
    """
    financial-growth (annual) para assetGrowth, epsgrowth, etc.
    """
    out = {}
    for s in symbols:
        params = {"apikey": api_key, "symbol": s, "period": "annual", "limit": 8}
        js = _get_json(f"{FMP_BASE}/financial-growth/{s}", params)
        if isinstance(js, list) and js:
            # último anual
            out[s] = js[0]
        time.sleep(0.12)
    return out

def fetch_eps_series(symbols: List[str], api_key: str) -> Dict[str, pd.Series]:
    """
    Serie de EPS (diluted) trimestral para medir estabilidad (varianza) y CAGR aprox.
    """
    out = {}
    for s in symbols:
        params = {"apikey": api_key, "period": "quarter", "limit": 40}
        js = _get_json(f"{FMP_BASE}/income-statement/{s}", params)
        if isinstance(js, list) and js:
            df = pd.DataFrame(js)
            if "date" in df.columns and "epsdiluted" in df.columns:
                ser = pd.to_numeric(df.set_index("date")["epsdiluted"], errors="coerce").sort_index()
                out[s] = ser
        time.sleep(0.12)
    return out

# ---------- quality score ----------
FEATURE_WEIGHTS = {
    # Profitability
    "roic":           +1.0,   # roicTTM / returnOnInvestedCapitalTTM
    "gross_margin":   +0.5,   # grossProfitMarginTTM
    "op_margin":      +0.6,   # operatingProfitMarginTTM or operatingMarginTTM

    # Leverage & quality
    "debt_to_equity": -0.4,   # debtToEquityTTM
    "interest_cov":   +0.6,   # interestCoverageTTM

    # Accruals (menor es mejor)
    "accruals":       -0.8,   # accrualsTTM / accrualRatioTTM (fallback)

    # Investment/returns to shareholders
    "asset_growth":   -0.7,   # assetGrowth
    "buyback":        +0.5,   # buybackRatioTTM o sharesbuybackratioTTM

    # Earnings quality
    "eps_cagr":       +0.8,   # CAGR aprox 5y (o desde series disponibles)
    "eps_var":        -0.6,   # varianza/vol de EPS (menor, mejor)
}

def _eps_features(eps_q: pd.Series) -> Dict[str, float]:
    eps_q = pd.to_numeric(eps_q, errors="coerce").dropna()
    if len(eps_q) < 8:
        return {"eps_cagr": np.nan, "eps_var": np.nan}
    # suaviza negativos pequeños
    eps_q = eps_q.replace(0, np.nan).dropna()
    if len(eps_q) < 8:
        return {"eps_cagr": np.nan, "eps_var": np.nan}
    # usa últimos ~20 trimestres si hay
    eps_q = eps_q.tail(20)
    # varianza normalizada (CV)
    eps_var = float((eps_q.pct_change().replace([np.inf,-np.inf], np.nan)).std())
    # CAGR aprox desde trimestral → anualiza “a ojo” (4 trimestres)
    start, end = float(eps_q.iloc[0]), float(eps_q.iloc[-1])
    n_years = max(1.0, len(eps_q)/4.0)
    if start <= 0 or end <= 0:
        eps_cagr = np.nan
    else:
        eps_cagr = (end/start)**(1.0/n_years) - 1.0
    return {"eps_cagr": eps_cagr, "eps_var": eps_var}

def compute_quality_from_fmp(symbols: List[str], api_key: str) -> pd.DataFrame:
    symbols = [s.upper() for s in symbols]
    ttm   = fetch_ttm(symbols, api_key)
    grow  = fetch_growth(symbols, api_key)
    eps_s = fetch_eps_series(symbols, api_key)

    rows = []
    for s in symbols:
        d = ttm.get(s, {})
        g = grow.get(s, {})
        eps = eps_s.get(s, pd.Series(dtype=float))

        # map fields with fallbacks
        roic  = _safe(d, ["roicTTM","returnOnInvestedCapitalTTM"])
        gm    = _safe(d, ["grossProfitMarginTTM"])
        opm   = _safe(d, ["operatingProfitMarginTTM","operatingMarginTTM"])
        dte   = _safe(d, ["debtToEquityTTM","debttoequityTTM"])
        icov  = _safe(d, ["interestCoverageTTM","interestCoverage"])
        accr  = _safe(d, ["accrualsTTM","accrualRatioTTM","accruals"], default=np.nan)
        asetg = _safe(g, ["assetGrowth","assetgrowth"], default=np.nan)
        buyb  = _safe(d, ["buybackRatioTTM","sharesbuybackratioTTM","shareBuybackRatioTTM"], default=np.nan)

        ef = _eps_features(eps)
        rows.append({
            "symbol": s,
            "roic": roic, "gross_margin": gm, "op_margin": opm,
            "debt_to_equity": dte, "interest_cov": icov,
            "accruals": accr, "asset_growth": asetg, "buyback": buyb,
            "eps_cagr": ef["eps_cagr"], "eps_var": ef["eps_var"],
        })

    df = pd.DataFrame(rows).set_index("symbol")

    # z-scores (signos ya están en FEATURE_WEIGHTS)
    zcols = {}
    for k, w in FEATURE_WEIGHTS.items():
        if k in df.columns:
            zcols[k] = _z(df[k])
        else:
            zcols[k] = pd.Series(np.nan, index=df.index)
    Z = pd.DataFrame(zcols)

    # score robusto: media ponderada de features disponibles para cada símbolo
    weights = pd.Series(FEATURE_WEIGHTS)
    # solo contar pesos de features no NaN por fila
    def row_score(zrow):
        mask = zrow.notna()
        if not mask.any(): return np.nan
        w = weights[mask]
        return float((zrow[mask] * w).sum() / abs(w).sum())

    quality = Z.apply(row_score, axis=1).rename("QualityScore")
    return quality.reset_index()
