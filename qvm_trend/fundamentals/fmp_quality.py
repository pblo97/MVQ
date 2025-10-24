# qvm_trend/fundamentals/fmp_quality.py
from __future__ import annotations

import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"


# --------------------------- utilidades básicas --------------------------- #
def _z(s: pd.Series) -> pd.Series:
    """Z-score con winsor 2% y guardas para series cortas."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().size < 5:  # muy pocos datos -> todo NaN para no inventar
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(0.02), s.quantile(0.98)
    s = s.clip(lo, hi)
    std = s.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def _safe(d: dict, keys: List[str], default=np.nan):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _sleep_between(i: int, base: float = 0.12):
    # pequeña espera progresiva para respetar rate limits free
    time.sleep(base + 0.02 * (i % 5))


class _HTTP:
    """Cliente simple con reintentos."""
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "mvq-quality/1.0"})

    def get_json(self, url: str, params: dict, retry: int = 2, backoff: float = 0.5) -> Optional[dict]:
        last_err = None
        for t in range(retry + 1):
            try:
                r = self.s.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
                last_err = f"HTTP {r.status_code}: {r.text[:120]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(backoff * (t + 1))
        # Devuelve None en error; el llamador decide
        return None


# --------------------------- fetchers de FMP --------------------------- #
def fetch_ttm(symbols: List[str], api_key: str) -> Dict[str, dict]:
    """
    Devuelve dict[symbol] -> mezcla de 'ratios-ttm' y 'key-metrics-ttm' (último dato).
    Importante: estos endpoints NO soportan batch estable, se consulta por símbolo.
    """
    http = _HTTP()
    out: Dict[str, dict] = {s: {} for s in symbols}
    for i, s in enumerate(symbols):
        # ratios-ttm/{symbol}
        rj = http.get_json(f"{FMP_BASE}/ratios-ttm/{s}", {"apikey": api_key}) or []
        if isinstance(rj, list) and rj:
            out[s].update(rj[0])  # último TTM
        _sleep_between(i)

        # key-metrics-ttm/{symbol}
        kj = http.get_json(f"{FMP_BASE}/key-metrics-ttm/{s}", {"apikey": api_key}) or []
        if isinstance(kj, list) and kj:
            out[s].update(kj[0])
        _sleep_between(i)
    return out


def fetch_growth(symbols: List[str], api_key: str) -> Dict[str, dict]:
    """financial-growth anual (último registro) por símbolo."""
    http = _HTTP()
    out: Dict[str, dict] = {}
    for i, s in enumerate(symbols):
        js = http.get_json(f"{FMP_BASE}/financial-growth/{s}",
                           {"apikey": api_key, "period": "annual", "limit": 8}) or []
        if isinstance(js, list) and js:
            out[s] = js[0]
        _sleep_between(i)
    return out


def fetch_eps_series(symbols: List[str], api_key: str) -> Dict[str, pd.Series]:
    """Serie trimestral de EPS diluido para estabilidad/crecimiento."""
    http = _HTTP()
    out: Dict[str, pd.Series] = {}
    for i, s in enumerate(symbols):
        js = http.get_json(f"{FMP_BASE}/income-statement/{s}",
                           {"apikey": api_key, "period": "quarter", "limit": 40}) or []
        if isinstance(js, list) and js:
            df = pd.DataFrame(js)
            col = "epsdiluted" if "epsdiluted" in df.columns else ("eps" if "eps" in df.columns else None)
            if col and "date" in df.columns:
                ser = pd.to_numeric(df.set_index("date")[col], errors="coerce").sort_index()
                out[s] = ser
        _sleep_between(i)
    return out


# --------------------------- features y scoring --------------------------- #
FEATURE_WEIGHTS = {
    # Rentabilidad
    "roic":           +1.0,  # roicTTM / returnOnInvestedCapitalTTM
    "gross_margin":   +0.5,  # grossProfitMarginTTM
    "op_margin":      +0.6,  # operatingProfitMarginTTM / operatingMarginTTM

    # Apalancamiento / calidad
    "debt_to_equity": -0.4,  # debtToEquityTTM
    "interest_cov":   +0.6,  # interestCoverageTTM

    # Devengos (menor, mejor)
    "accruals":       -0.8,  # accrualsTTM / accrualRatioTTM

    # Inversión y retorno al accionista
    "asset_growth":   -0.7,  # financial-growth.assetGrowth
    "buyback":        +0.5,  # buybackRatioTTM / sharesBuybackRatioTTM

    # Calidad de ganancias
    "eps_cagr":       +0.8,  # CAGR ~5y desde serie trimestral
    "eps_var":        -0.6,  # volatilidad de cambios trimestrales (menor, mejor)
}


def _eps_features(eps_q: pd.Series) -> Dict[str, float]:
    eps_q = pd.to_numeric(eps_q, errors="coerce").dropna()
    if len(eps_q) < 8:
        return {"eps_cagr": np.nan, "eps_var": np.nan}

    eps_q = eps_q.replace(0, np.nan).dropna()
    if len(eps_q) < 8:
        return {"eps_cagr": np.nan, "eps_var": np.nan}

    eps_q = eps_q.tail(20)  # ~5 años
    pct = eps_q.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    eps_var = float(pct.std()) if pct.size else np.nan

    start, end = float(eps_q.iloc[0]), float(eps_q.iloc[-1])
    n_years = max(1.0, len(eps_q) / 4.0)
    eps_cagr = ((end / start) ** (1.0 / n_years) - 1.0) if (start > 0 and end > 0) else np.nan
    return {"eps_cagr": eps_cagr, "eps_var": eps_var}


def compute_quality_from_fmp(
    symbols: List[str],
    api_key: str,
    include_components: bool = False,
) -> pd.DataFrame:
    """
    Calcula QualityScore para 'symbols' usando FMP.
    Devuelve DataFrame con: symbol, QualityScore (+ opcionalmente features y z-scores).
    """
    symbols = [s.upper() for s in symbols if s and isinstance(s, str)]
    if not symbols:
        return pd.DataFrame(columns=["symbol", "QualityScore"])

    ttm = fetch_ttm(symbols, api_key)
    grw = fetch_growth(symbols, api_key)
    eps = fetch_eps_series(symbols, api_key)

    rows = []
    for s in symbols:
        d = ttm.get(s, {}) or {}
        g = grw.get(s, {}) or {}
        e = eps.get(s, pd.Series(dtype=float))

        roic = _safe(d, ["roicTTM", "returnOnInvestedCapitalTTM"])
        gm = _safe(d, ["grossProfitMarginTTM"])
        opm = _safe(d, ["operatingProfitMarginTTM", "operatingMarginTTM"])
        dte = _safe(d, ["debtToEquityTTM", "debttoequityTTM"])
        icov = _safe(d, ["interestCoverageTTM", "interestCoverage"])
        accr = _safe(d, ["accrualsTTM", "accrualRatioTTM", "accruals"])
        asetg = _safe(g, ["assetGrowth", "assetgrowth"])
        buyb = _safe(d, ["buybackRatioTTM", "sharesbuybackratioTTM", "shareBuybackRatioTTM"])

        ef = _eps_features(e)
        rows.append({
            "symbol": s,
            "roic": roic,
            "gross_margin": gm,
            "op_margin": opm,
            "debt_to_equity": dte,
            "interest_cov": icov,
            "accruals": accr,
            "asset_growth": asetg,
            "buyback": buyb,
            "eps_cagr": ef["eps_cagr"],
            "eps_var": ef["eps_var"],
        })

    feats = pd.DataFrame(rows).set_index("symbol")

    # z-scores por columna (si no hay datos suficientes -> NaN)
    Z = pd.DataFrame({k: _z(feats[k]) if k in feats.columns else pd.Series(np.nan, index=feats.index)
                      for k in FEATURE_WEIGHTS.keys()})

    weights = pd.Series(FEATURE_WEIGHTS)

    def _row_score(zrow: pd.Series) -> float:
        mask = zrow.notna()
        if not mask.any():
            return np.nan
        w = weights[mask]
        # media ponderada por el valor absoluto de los pesos (para no sesgar por signo)
        return float((zrow[mask] * w).sum() / np.abs(w).sum())

    quality = Z.apply(_row_score, axis=1).rename("QualityScore")

    if include_components:
        out = pd.concat([feats, Z.add_prefix("z_"), quality], axis=1).reset_index()
    else:
        out = quality.reset_index()

    return out
