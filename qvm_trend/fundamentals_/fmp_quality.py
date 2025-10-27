# qvm_trend/fundamentals/fmp_quality.py
from __future__ import annotations
import time, math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"
# Profundidad razonable para Starter (≈ 5 años trimestral)
Q_LIMIT = 20
SLEEP = 0.12  # pequeño backoff para no gatillar rate limit

# ==========================
# Utils
# ==========================
def _winsor_z(s: pd.Series, p: float = 0.02) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    s = s.clip(lo, hi)
    return (s - s.mean()) / (s.std(ddof=1) + 1e-12)

def _get(url: str, params: dict, retries: int = 2) -> list | dict | None:
    for t in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            # Backoff simple ante 429 u otros
            time.sleep(SLEEP * (t + 1) * 2)
        except Exception:
            time.sleep(SLEEP * (t + 1))
    return None

def _quarter_df(js: list, index_col: str = "date") -> pd.DataFrame:
    if not isinstance(js, list) or not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
        df = df.dropna(subset=[index_col]).set_index(index_col).sort_index()
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
    return df

def _last4_med(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().tail(4)
    return float(s.median()) if len(s) else np.nan

def _last4_sum(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().tail(4)
    return float(s.sum()) if len(s) else np.nan

# ==========================
# Fetchers (quarterly history)
# ==========================
def fetch_quarterly(symbol: str, api_key: str) -> dict[str, pd.DataFrame]:
    """
    Descarga historia trimestral para un símbolo:
      income-statement, balance-sheet-statement, cash-flow-statement,
      ratios y key-metrics.
    """
    params_q = {"apikey": api_key, "period": "quarter", "limit": Q_LIMIT}

    inc = _quarter_df(_get(f"{FMP_BASE}/income-statement/{symbol}", params_q))
    bal = _quarter_df(_get(f"{FMP_BASE}/balance-sheet-statement/{symbol}", params_q))
    cfs = _quarter_df(_get(f"{FMP_BASE}/cash-flow-statement/{symbol}", params_q))
    rat = _quarter_df(_get(f"{FMP_BASE}/ratios/{symbol}", params_q))
    met = _quarter_df(_get(f"{FMP_BASE}/key-metrics/{symbol}", params_q))

    time.sleep(SLEEP)  # backoff suave para no pegar al rate limit

    return {"income": inc, "balance": bal, "cash": cfs, "ratios": rat, "metrics": met}

# ==========================
# Features
# ==========================
def _eps_features(inc: pd.DataFrame) -> Tuple[float, float]:
    """
    Devuelve (eps_cagr, eps_var) usando epsdiluted trimestral.
    eps_var = std de variación porcentual trimestral.
    eps_cagr ~ CAGR anualizado aprox (usando últimos 16-20 trimestres).
    """
    if inc.empty or "epsdiluted" not in inc.columns:
        return (np.nan, np.nan)
    eps_q = pd.to_numeric(inc["epsdiluted"], errors="coerce").dropna().tail(20)
    if len(eps_q) < 8:
        return (np.nan, np.nan)
    # variabilidad de EPS QoQ
    eps_var = float(eps_q.pct_change().replace([np.inf, -np.inf], np.nan).std())
    # CAGR aprox (4 trimestres ~ 1 año)
    start, end = float(eps_q.iloc[0]), float(eps_q.iloc[-1])
    n_years = max(1.0, len(eps_q) / 4.0)
    if start <= 0 or end <= 0:
        eps_cagr = np.nan
    else:
        eps_cagr = (end / start) ** (1.0 / n_years) - 1.0
    return (eps_cagr, eps_var)

def _buyback_ratio_from_shares(metrics: pd.DataFrame) -> float:
    """
    Aproxima buyback ratio como % reducción de shares outstanding en el último año (4Q).
    Busca 'sharesOutstanding'; si no existe, devuelve NaN.
    """
    col_candidates = [c for c in metrics.columns if c.lower().startswith("shares") and "outstanding" in c.lower()]
    if metrics.empty or not col_candidates:
        return np.nan
    s = pd.to_numeric(metrics[col_candidates[0]], errors="coerce").dropna().tail(5)
    if len(s) < 5:  # necesitamos comparar t y t-4 (aprox 1 año)
        return np.nan
    last, prev4 = float(s.iloc[-1]), float(s.iloc[-5])
    if prev4 <= 0:
        return np.nan
    chg = (last - prev4) / prev4  # negativo = reducción
    return float(-chg)  # reducción de acciones => buyback positivo

def _accruals_proxy(inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame) -> float:
    # 1) elegir CFO
    if "netCashProvidedByOperatingActivities" in cfs.columns:
        cf_col = "netCashProvidedByOperatingActivities"
    else:
        cf_col = next(
            (c for c in cfs.columns if isinstance(c, str) and ("operat" in c.lower() and "net" in c.lower())),
            None
        )
    if cf_col is None:
        return np.nan

    # 2) working capital
    if {"totalCurrentAssets", "totalCurrentLiabilities"}.issubset(bal.columns):
        WC = (pd.to_numeric(bal["totalCurrentAssets"], errors="coerce") -
              pd.to_numeric(bal["totalCurrentLiabilities"], errors="coerce"))
    else:
        return np.nan

    dWC_4Q   = _last4_sum(WC.diff())
    CFO_4Q   = _last4_sum(pd.to_numeric(cfs[cf_col], errors="coerce"))
    assets_4Q = _last4_med(pd.to_numeric(bal.get("totalAssets", pd.Series()), errors="coerce"))

    if not np.isfinite(dWC_4Q) or not np.isfinite(CFO_4Q) or not np.isfinite(assets_4Q) or assets_4Q == 0:
        return np.nan

    return float((dWC_4Q - CFO_4Q) / abs(assets_4Q))


# Pesos (puede afinarlos a gusto)
FEATURE_WEIGHTS = {
    # Profitability
    "roic":         +1.0,  # mediana 4Q de roic (ratios/metrics)
    "gross_margin": +0.5,  # grossProfitMargin
    "op_margin":    +0.6,  # operatingMargin / operatingProfitMargin

    # Leverage & quality
    "debt_to_equity": -0.4,
    "interest_cov":   +0.6,

    # Accruals (menor es mejor)
    "accruals":     -0.8,

    # Investment/returns to shareholders
    "asset_growth": -0.7,  # de financial-growth o derivado (si no, NaN)
    "buyback":      +0.5,  # reducción de shares aprox

    # Earnings quality
    "eps_cagr":     +0.8,
    "eps_var":      -0.6,
}

# ==========================
# Main
# ==========================
def compute_quality_from_fmp(symbols: List[str], api_key: str) -> pd.DataFrame:
    """
    Descarga historia trimestral (Starter plan friendly) y construye QualityScore.
    Devuelve DataFrame con columnas: symbol, QualityScore y features individuales.
    """
    symbols = [s.upper() for s in symbols if s and isinstance(s, str)]
    rows = []

    for sym in symbols:
        data = fetch_quarterly(sym, api_key)
        inc, bal, cfs, rat, met = data["income"], data["balance"], data["cash"], data["ratios"], data["metrics"]

        # --- Profitability (mediana de últimos 4Q) ---
        roic = _last4_med(met.get("roic", pd.Series())) if "roic" in met.columns else \
               _last4_med(rat.get("returnOnInvestedCapital", pd.Series()))
        gross_margin = _last4_med(rat.get("grossProfitMargin", pd.Series()))
        op_margin = _last4_med(rat.get("operatingProfitMargin", pd.Series()) if "operatingProfitMargin" in rat.columns
                               else rat.get("operatingMargin", pd.Series()))

        # --- Leverage & coverage (últimos 4Q median) ---
        dte = _last4_med(rat.get("debtEquityRatio", pd.Series()) if "debtEquityRatio" in rat.columns
                         else rat.get("debtToEquity", pd.Series()))
        icov = _last4_med(rat.get("interestCoverage", pd.Series()))

        # --- Accruals ---
        accr = _last4_med(rat.get("accruals", pd.Series()))  # si el endpoint lo trae
        if not np.isfinite(accr):
            accr = _accruals_proxy(inc, bal, cfs)

        # --- Investment / buybacks ---
        # Asset growth anual (de financial-growth ANUAL si querés; Starter suele permitirlo).
        fg = _get(f"{FMP_BASE}/financial-growth/{sym}", {"apikey": api_key, "period": "annual", "limit": 6})
        time.sleep(SLEEP)
        asset_growth = np.nan
        if isinstance(fg, list) and fg:
            df_g = _quarter_df(fg, index_col="date")  # fechas anuales igual sirven
            col_candidates = [c for c in df_g.columns if "assetgrowth" in c.lower()]
            if col_candidates:
                asset_growth = float(pd.to_numeric(df_g[col_candidates[0]], errors="coerce").dropna().iloc[-1]) \
                               if not df_g[col_candidates[0]].dropna().empty else np.nan

        buyback = _buyback_ratio_from_shares(met)

        # --- Earnings quality ---
        eps_cagr, eps_var = _eps_features(inc)

        rows.append({
            "symbol": sym,
            "roic": roic, "gross_margin": gross_margin, "op_margin": op_margin,
            "debt_to_equity": dte, "interest_cov": icov,
            "accruals": accr, "asset_growth": asset_growth, "buyback": buyback,
            "eps_cagr": eps_cagr, "eps_var": eps_var,
        })

    df = pd.DataFrame(rows).set_index("symbol")

    # Z-scores robustos y score final
    Z = pd.DataFrame({k: _winsor_z(df[k]) if k in df.columns else pd.Series(np.nan, index=df.index)
                      for k in FEATURE_WEIGHTS.keys()})
    weights = pd.Series(FEATURE_WEIGHTS)

    def row_score(zrow: pd.Series) -> float:
        m = zrow.notna()
        if not m.any():
            return np.nan
        w = weights[m]
        return float((zrow[m] * w).sum() / abs(w).sum())

    df["QualityScore"] = Z.apply(row_score, axis=1)
    return df.reset_index()[["symbol", "QualityScore"] + list(FEATURE_WEIGHTS.keys())]

# Añade al final de qvm_trend/fundamentals/fmp_quality.py

def _roll_med(s: pd.Series, w: int = 4) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rolling(w, min_periods=w).median()

def _roll_sum(s: pd.Series, w: int = 4) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rolling(w, min_periods=w).sum()

def quality_history_from_fmp(symbols: List[str], api_key: str) -> pd.DataFrame:
    """
    Devuelve histórico trimestral de QualityScore:
    columnas: ['symbol','date','VFQ']
    Usa rolling(4Q) para aproximar TTM por trimestre.
    """
    out_rows = []
    for sym in [s.upper() for s in symbols]:
        data = fetch_quarterly(sym, api_key)
        # ✅ FIX: incluye "cash"
        inc, bal, cfs, rat, met = data["income"], data["balance"], data["cash"], data["ratios"], data["metrics"]


        # Alineamos un índice trimestral base (el de income por lo general es el más completo)
        if inc.empty:
            continue
        dates = inc.index

        # Profitability (rolling median 4Q)
        roic = None
        if "roic" in met.columns:
            roic = _roll_med(met["roic"].reindex(dates))
        elif "returnOnInvestedCapital" in rat.columns:
            roic = _roll_med(rat["returnOnInvestedCapital"].reindex(dates))

        gm = _roll_med(rat.get("grossProfitMargin", pd.Series(index=dates))).reindex(dates)
        opm_src = rat["operatingProfitMargin"] if "operatingProfitMargin" in rat.columns else rat.get("operatingMargin", pd.Series(index=dates))
        opm = _roll_med(opm_src.reindex(dates)) if opm_src is not None else pd.Series(index=dates, dtype=float)

        # Leverage / coverage (rolling median 4Q)
        dte_src = rat["debtEquityRatio"] if "debtEquityRatio" in rat.columns else rat.get("debtToEquity", pd.Series(index=dates))
        dte  = _roll_med(dte_src.reindex(dates)) if dte_src is not None else pd.Series(index=dates, dtype=float)
        icov = _roll_med(rat.get("interestCoverage", pd.Series(index=dates)).reindex(dates))

        # Accruals (si ratios trae, usamos eso; si no, proxy con ΔWC − CFO sobre activos, rolling 4Q)
        accr = None
        if "accruals" in rat.columns:
            accr = _roll_med(rat["accruals"].reindex(dates))
        else:
            # Proxy
            # Proxy accruals si no viene en ratios
            WC = None
            if {"totalCurrentAssets", "totalCurrentLiabilities"}.issubset(bal.columns):
                WC = (pd.to_numeric(bal["totalCurrentAssets"], errors="coerce") -
                    pd.to_numeric(bal["totalCurrentLiabilities"], errors="coerce")).reindex(dates)

            # columna CFO
            if "netCashProvidedByOperatingActivities" in cfs.columns:
                cf_col = "netCashProvidedByOperatingActivities"
            else:
                cf_col = next(
                    (c for c in cfs.columns if isinstance(c, str) and ("operat" in c.lower() and "net" in c.lower())),
                    None
                )

            if WC is not None and cf_col:
                dWC    = WC.diff()
                CFO    = pd.to_numeric(cfs[cf_col], errors="coerce").reindex(dates)
                assets = pd.to_numeric(bal.get("totalAssets", pd.Series(index=dates)), errors="coerce").reindex(dates)

                num = _roll_sum(dWC) - _roll_sum(CFO)
                den = _roll_med(assets).abs() + 1e-12
                accr = (num / den)
            else:
                accr = pd.Series(index=dates, dtype=float)


        # Buyback (reducción % de shares en 4Q)
        sh_col = next((c for c in met.columns if c.lower().startswith("shares") and "outstanding" in c.lower()), None)
        if sh_col:
            sh = pd.to_numeric(met[sh_col], errors="coerce").reindex(dates)
            buyback = -(sh / sh.shift(4) - 1.0)
        else:
            buyback = pd.Series(index=dates, dtype=float)

        # Asset growth anual (último dato anual mapeado al trimestre más cercano)
        fg = _get(f"{FMP_BASE}/financial-growth/{sym}", {"apikey": api_key, "period": "annual", "limit": 8})
        asset_growth = pd.Series(index=dates, dtype=float)
        if isinstance(fg, list) and fg:
            df_g = _quarter_df(fg, index_col="date")  # fechas anuales
            col = next((c for c in df_g.columns if "assetgrowth" in c.lower()), None)
            if col:
                asset_growth = pd.to_numeric(df_g[col], errors="coerce").reindex(dates, method="ffill")

        # EPS features por trimestre (usando historia hasta t)
        eps = pd.to_numeric(inc.get("epsdiluted", pd.Series(index=dates)), errors="coerce")
        eps_var = eps.pct_change().rolling(16, min_periods=8).std()
        # CAGR aprox anualizado con ventana de hasta 20Q
        def _eps_cagr_point(s: pd.Series) -> float:
            s = s.dropna()
            if len(s) < 8 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
                return np.nan
            yrs = max(1.0, len(s)/4.0)
            return (float(s.iloc[-1]/s.iloc[0]) ** (1.0/yrs)) - 1.0
        eps_cagr = eps.rolling(20, min_periods=8).apply(_eps_cagr_point, raw=False)

        # Construcción del score por trimestre
        feat_df = pd.DataFrame({
            "roic": roic, "gross_margin": gm, "op_margin": opm,
            "debt_to_equity": dte, "interest_cov": icov,
            "accruals": accr, "asset_growth": asset_growth, "buyback": buyback,
            "eps_cagr": eps_cagr, "eps_var": eps_var
        }, index=dates)

        # Z winsorizado por columna
        Z = feat_df.apply(_winsor_z)

        weights = pd.Series(FEATURE_WEIGHTS)
        def _row_score(r):
            m = r.notna()
            if not m.any(): return np.nan
            w = weights[m]
            return float((r[m]*w).sum() / abs(w).sum())

        VFQ = Z.apply(_row_score, axis=1).dropna()
        if not VFQ.empty:
            out_rows += [{"symbol": sym, "date": d, "VFQ": float(v)} for d, v in VFQ.items()]

        time.sleep(SLEEP)

    return pd.DataFrame(out_rows, columns=["symbol","date","VFQ"])
