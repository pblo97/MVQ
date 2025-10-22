# qvm_trend/fundamentals.py
from __future__ import annotations
from typing import List, Dict, Any

import math
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import time 

# HTTP común (robusto, con rate limit/backoff) provisto en data_io.py
from .data_io import _http_get

# Cache opcional (si no está, definimos no-ops)
try:
    from .cache_io import save_df, load_df
except Exception:
    def save_df(df: pd.DataFrame, key: str):  # no-op
        return
    def load_df(key: str) -> Optional[pd.DataFrame]:  # no-op
        return None

# ======================================================================================
# Helpers genéricos
# ======================================================================================

def _first_obj(x):
    """Devuelve el primer objeto si es lista; si es dict lo devuelve; si no, {}."""
    if isinstance(x, list):
        return x[0] if x else {}
    return x if isinstance(x, dict) else {}

def _safe_float(x):
    try:
        if x in ("", None):
            return None
        return float(x)
    except Exception:
        return None

def _yr_series(items, key):
    """Convierte list[dict] anual/quarter en lista de (fecha, valor) con coerción numérica."""
    out = []
    for it in (items or []):
        d = it.get("date")
        v = _safe_float(it.get(key))
        if d and v is not None:
            out.append((pd.to_datetime(d), v))
    out.sort(key=lambda z: z[0])
    return out

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

# ======================================================================================
# FUNDAMENTALES (set mínimo) → para VFQ
# ======================================================================================

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

def _fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> Dict[str, Any]:
    """
    Descarga el set mínimo y normaliza nombres:
      evToEbitda, fcf_ttm, cfo_ttm, ebit_ttm, grossProfitTTM, totalAssetsTTM,
      roic, roa, netMargin, marketCap (si hay/ hint)
    Usa TTM y cae en annual si falta.
    """
    s = symbol.strip().upper()
    out: Dict[str, Any] = {"symbol": s}

    # --- KEY METRICS TTM (ev/ebitda, grossProfitTTM, totalAssetsTTM) ---
    try:
        j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{s}")
        kmttm = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
    except Exception:
        kmttm = {}

    # --- RATIOS TTM (roic/roa/netMargin) ---
    try:
        j = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{s}")
        rttm = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
    except Exception:
        rttm = {}

    # --- CASH-FLOW TTM (CFO/FCF) ---
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{s}")
        cfttm = cfttm if isinstance(cfttm, dict) else {}
    except Exception:
        cfttm = {}

    # --- INCOME TTM (EBIT aprox) ---
    try:
        incttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{s}")
        incttm = incttm if isinstance(incttm, dict) else {}
    except Exception:
        incttm = {}

    # Map TTM → normalizados
    evttm  = _num(kmttm.get("enterpriseValueOverEBITDATTM"))
    gpttm  = _num(kmttm.get("grossProfitTTM"))
    tattm  = _num(kmttm.get("totalAssetsTTM"))
    fcf_t  = _num(cfttm.get("freeCashFlowTTM"))
    cfo_t  = _num(cfttm.get("netCashProvidedByOperatingActivitiesTTM"))
    ebit_t = _num(incttm.get("ebitTTM") or incttm.get("operatingIncomeTTM"))

    roic_t = _num(rttm.get("returnOnCapitalEmployedTTM") or rttm.get("returnOnInvestedCapitalTTM"))
    roa_t  = _num(rttm.get("returnOnAssetsTTM"))
    nmar_t = _num(rttm.get("netProfitMarginTTM"))

    out["evToEbitda"]        = evttm
    out["grossProfitTTM"]    = gpttm
    out["totalAssetsTTM"]    = tattm
    out["fcf_ttm"]           = fcf_t
    out["cfo_ttm"]           = cfo_t
    out["ebit_ttm"]          = ebit_t
    out["roic"]              = roic_t
    out["roa"]               = roa_t
    out["netMargin"]         = nmar_t
    out["marketCap"]         = _num(kmttm.get("marketCap")) or (market_cap_hint if market_cap_hint else None)

    # --- Flags de fuente (útiles para debug) ---
    out["__src_ev"]   = "ttm" if evttm is not None else None
    out["__src_gp"]   = "ttm" if gpttm is not None else None
    out["__src_ta"]   = "ttm" if tattm is not None else None
    out["__src_fcf"]  = "ttm" if fcf_t is not None else None
    out["__src_cfo"]  = "ttm" if cfo_t is not None else None
    out["__src_ebit"] = "ttm" if ebit_t is not None else None
    out["__src_roic"] = "ttm" if roic_t is not None else None
    out["__src_roa"]  = "ttm" if roa_t is not None else None
    out["__src_nmar"] = "ttm" if nmar_t is not None else None

    # -------- Fallback ANUAL si falta algo crítico --------
    need_annual = any(
        x is None for x in [out["evToEbitda"], out["grossProfitTTM"], out["totalAssetsTTM"],
                            out["fcf_ttm"], out["cfo_ttm"], out["ebit_ttm"], out["roic"], out["roa"], out["netMargin"]]
    )

    if need_annual:
        # ANNUAL key-metrics (ev/ebitda, grossProfit, totalAssets, marketCap)
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{s}", params={"period":"annual","limit":4})
            km = j[0] if isinstance(j, list) and j else {}
        except Exception:
            km = {}
        if out["evToEbitda"]     is None: out["evToEbitda"]     = _num(km.get("enterpriseValueOverEBITDA"))
        if out["grossProfitTTM"] is None: out["grossProfitTTM"] = _num(km.get("grossProfit"))
        if out["totalAssetsTTM"] is None: out["totalAssetsTTM"] = _num(km.get("totalAssets"))
        if out["marketCap"]      is None: out["marketCap"]      = _num(km.get("marketCap")) or (market_cap_hint if market_cap_hint else None)

        # ANNUAL ratios (roic/roa/net margin)
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/ratios/{s}", params={"period":"annual","limit":4})
            rr = j[0] if isinstance(j, list) and j else {}
        except Exception:
            rr = {}
        if out["roic"]      is None: out["roic"]      = _num(rr.get("returnOnCapitalEmployed") or rr.get("returnOnInvestedCapital"))
        if out["roa"]       is None: out["roa"]       = _num(rr.get("returnOnAssets"))
        if out["netMargin"] is None: out["netMargin"] = _num(rr.get("netProfitMargin"))

        # ANNUAL cash-flow (CFO/FCF)
        if out["cfo_ttm"] is None or out["fcf_ttm"] is None:
            try:
                cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{s}", params={"period":"annual","limit":1})
                cf0 = cf[0] if isinstance(cf, list) and cf else {}
            except Exception:
                cf0 = {}
            if out["cfo_ttm"] is None: out["cfo_ttm"] = _num(cf0.get("netCashProvidedByOperatingActivities"))
            if out["fcf_ttm"] is None: out["fcf_ttm"] = _num(cf0.get("freeCashFlow"))

        # ANNUAL income (EBIT)
        if out["ebit_ttm"] is None:
            try:
                inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{s}", params={"period":"annual","limit":1})
                inc0 = inc[0] if isinstance(inc, list) and inc else {}
            except Exception:
                inc0 = {}
            out["ebit_ttm"] = _num(inc0.get("ebit") or inc0.get("operatingIncome"))

    return out


def _coverage_count(df: pd.DataFrame) -> int:
    if df is None or df.empty: 
        return 0
    cols = [c for c in ["evToEbitda","fcf_ttm","cfo_ttm","ebit_ttm",
                        "grossProfitTTM","totalAssetsTTM","roic","roa","netMargin"] if c in df.columns]
    return int(df[cols].notna().sum(axis=1).sum()) if cols else 0

def download_fundamentals(symbols: List[str],
                          market_caps: Dict[str, float] | None = None,
                          cache_key: str | None = None,
                          force: bool = False,
                          max_symbols_per_minute: int = 50) -> pd.DataFrame:
    """
    Descarga mínimos de batalla para VFQ con:
      - reintentos suaves y limitación de tasa
      - evita cachear snapshots sin cobertura
    """
    from .cache_io import load_df, save_df  # lazy
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None and not dfc.empty:
            return dfc

    rows, errs = [], 0
    mc_map = market_caps or {}
    throttle = max(0.0, 60.0 / max(1, max_symbols_per_minute))
    for i, s in enumerate(symbols):
        # simple rate-limit
        if i > 0 and throttle > 0:
            time.sleep(throttle)
        try:
            rec = _fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s))
            rows.append(rec)
        except Exception as e:
            errs += 1
            rows.append({"symbol": s, "__err_fund": str(e)[:180]})

    df = pd.DataFrame(rows).drop_duplicates("symbol")

    # Si literalmente no hay cobertura, intenta un segundo pase con 25 símbolos aleatorios
    if _coverage_count(df) == 0 and len(symbols) > 0:
        sample = list(pd.Series(symbols).drop_duplicates().sample(min(25, len(symbols)), random_state=42))
        rows2 = []
        for s in sample:
            try:
                rows2.append(_fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s)))
                time.sleep(throttle)
            except Exception as e:
                rows2.append({"symbol": s, "__err_fund": str(e)[:180]})
        df2 = pd.DataFrame(rows2).drop_duplicates("symbol")
        # mergea lo que haya
        df = df.set_index("symbol").combine_first(df2.set_index("symbol")).reset_index()

    # NO guardes si sigue sin cobertura (evita “cachear vacío”)
    if key and _coverage_count(df) > 0:
        try: save_df(df, key)
        except Exception: pass

    return df

def download_guardrails_batch(symbols: List[str], cache_key: str | None = None, force: bool = False) -> pd.DataFrame:
    key = f"guard_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc
    rows = []
    for s in symbols:
        try:
            rows.append(download_guardrails(s))
        except Exception as e:
            rows.append({"symbol": s, "__err_guard": str(e)[:180]})
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key: save_df(df, key)
    return df

def _winsor(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s is None or s.empty: 
        return s
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def _bucket_by_quantiles(s: pd.Series, q: int = 3) -> pd.Series:
    r = s.rank(method="first", na_option="keep")
    try:
        return pd.qcut(r, q, labels=False, duplicates="drop")
    except Exception:
        if r.max() and r.max() > 0:
            pct = r / r.max()
        else:
            pct = r
        return pd.Series(np.select(
            [pct <= 0.33, pct <= 0.66, pct > 0.66],
            [0,1,2],
            default=np.nan
        ), index=s.index)

def build_vfq_scores(df_universe: pd.DataFrame, df_fund: pd.DataFrame,
                     size_buckets: int = 3) -> pd.DataFrame:
    """
    Fusiona universo + fundamentales mínimos y calcula VFQ de forma tolerante a NaNs.
    Devuelve un DF con:
      ['symbol','sector','marketCap_unified','coverage_count','ValueScore','QualityScore','VFQ','VFQ_pct_sector', ...]
    """
    # --- merge base (SIN usar 'or' sobre DataFrames)
    if isinstance(df_universe, pd.DataFrame):
        dfu = df_universe.copy()
    else:
        dfu = pd.DataFrame()

    if isinstance(df_fund, pd.DataFrame):
        dff = df_fund.copy()
    else:
        dff = pd.DataFrame()

    if dfu.empty or "symbol" not in dfu.columns:
        return pd.DataFrame(columns=["symbol","VFQ","coverage_count"])

    if "symbol" not in dff.columns:
        dff = pd.DataFrame(columns=["symbol"])  # vacío pero mergeable

    df = dfu.merge(dff, on="symbol", how="left").copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # --- columnas de identificación
    for col in ["sector","industry"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df["sector"] = df["sector"].astype(str).replace({None: "Unknown"}).fillna("Unknown")

    # --- market cap unificado
       # --- market cap unificado (robusto a _x/_y y variantes)
    def to_num(colname: str) -> pd.Series:
        return pd.to_numeric(df[colname], errors="coerce") if colname in df.columns else pd.Series(np.nan, index=df.index)

    # 1) Candidatos de market cap (acepta marketCap, marketCap_x, marketCap_y, marketCap_profile, marketCap_ev, etc.)
    mcap = pd.Series(np.nan, index=df.index)
    mcap_candidates = (
        ["marketCap", "marketCap_profile", "marketCap_ev"] +
        [c for c in df.columns if c.lower().startswith("marketcap")]
    )
    for c in mcap_candidates:
        if c in df.columns:
            mcap = mcap.fillna(to_num(c))

    # 2) Fallback: price * sharesOutstanding (también robusto a _x/_y)
    price_series = pd.Series(np.nan, index=df.index)
    for c in [c for c in df.columns if c.lower().startswith("price")]:
        price_series = price_series.fillna(to_num(c))

    shares_series = pd.Series(np.nan, index=df.index)
    shares_candidates = (
        ["sharesOutstanding", "shares_out_ttm"] +
        [c for c in df.columns if c.lower().startswith("sharesoutstanding")]
    )
    for c in shares_candidates:
        if c in df.columns:
            shares_series = shares_series.fillna(to_num(c))

    mcap = mcap.fillna(price_series * shares_series)
    df["marketCap_unified"] = pd.to_numeric(mcap, errors="coerce")

    # --- bucket por tamaño (usa helper global _bucket_by_quantiles)
    df["size_bucket"] = _bucket_by_quantiles(df["marketCap_unified"], q=size_buckets)
    grp_key = df["sector"].astype(str) + "|" + df["size_bucket"].astype(str)

    # --------- derivadas para Value/Quality ----------
    ev  = to_num("evToEbitda")
    fcf = to_num("fcf_ttm")
    gp  = to_num("grossProfitTTM")
    ta  = to_num("totalAssetsTTM")

    df["inv_ev_ebitda"] = (1.0 / ev).replace([np.inf, -np.inf], np.nan)
    df["fcf_yield"] = (fcf / df["marketCap_unified"]).replace([np.inf, -np.inf], np.nan)
    df["gross_profitability"] = (gp / ta).replace([np.inf, -np.inf], np.nan)

    # --- columnas VFQ disponibles
    val_cols = [c for c in ["fcf_yield","inv_ev_ebitda"] if c in df.columns]
    q_cols   = [c for c in ["gross_profitability","roic","roa","netMargin"] if c in df.columns]

    # winsor suave
    for c in val_cols + q_cols:
        df[c] = _winsor(df[c], 0.01)

    fields = val_cols + q_cols
    if len(fields) == 0:
        df["coverage_count"] = 0
        df["ValueScore"] = np.nan
        df["QualityScore"] = np.nan
        df["VFQ"] = np.nan
        df["VFQ_pct_sector"] = 1.0
        return df

    df["coverage_count"] = df[fields].notna().sum(axis=1)

    def _rank_group(col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.groupby(grp_key).rank(method="average", ascending=False, na_option="bottom")

    df["ValueScore"]   = pd.concat([_rank_group(c) for c in val_cols], axis=1).mean(axis=1) if val_cols else np.nan
    df["QualityScore"] = pd.concat([_rank_group(c) for c in q_cols],  axis=1).mean(axis=1) if q_cols else np.nan
    df["VFQ"]          = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1, skipna=True)

    # VFQ percentil intra-sector (robusto)
    try:
        sec = df["sector"].astype(str).replace({None: "Unknown"}).fillna("Unknown")
        df["VFQ_pct_sector"] = df.groupby(sec)["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = df["VFQ_pct_sector"].fillna(1.0)

    return df

# ======================================================================================
# GUARDRAILS (calidad) — batch + aplicación de umbrales
# ======================================================================================

def download_guardrails(symbol: str) -> dict:
    """
    Calcula métricas para guardrails (con fallbacks robustos):
      - ebit_ttm, cfo_ttm, fcf_ttm (profit floor)
      - net_issuance (Δ acciones)
      - asset_growth (y/y)
      - accruals_ta = (NI - CFO)/assets promedio
      - netdebt_ebitda
    """
    sym = (symbol or "").strip().upper()
    out = {"symbol": sym}

    # --- KEY-METRICS TTM ---
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        kt0 = _first_obj(kttm)
        out["shares_out_ttm"] = _safe_float(kt0.get("sharesOutstanding"))
        out["net_debt_ttm"]   = _safe_float(kt0.get("netDebtTTM"))
        out["ebitda_ttm"]     = _safe_float(kt0.get("ebitdaTTM"))
    except Exception:
        pass

    # --- CFO/FCF TTM ---
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{sym}")
        cf0 = _first_obj(cfttm)
        out["cfo_ttm"] = _safe_float(cf0.get("netCashProvidedByOperatingActivitiesTTM"))
        out["fcf_ttm"] = _safe_float(cf0.get("freeCashFlowTTM"))
    except Exception:
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}",
                           params={"period": "annual", "limit": 1})
            cf0 = _first_obj(cf)
            out["cfo_ttm"] = _safe_float(cf0.get("netCashProvidedByOperatingActivities"))
            out["fcf_ttm"] = _safe_float(cf0.get("freeCashFlow"))
        except Exception:
            pass

    # --- EBIT TTM ---
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        it0 = _first_obj(inc_ttm)
        out["ebit_ttm"] = _safe_float(it0.get("ebitTTM") or it0.get("operatingIncomeTTM"))
    except Exception:
        try:
            inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                            params={"period": "annual", "limit": 1})
            i0 = _first_obj(inc)
            out["ebit_ttm"] = _safe_float(i0.get("ebit") or i0.get("operatingIncome"))
        except Exception:
            pass

    # --- Series anuales para growth/accruals/issuance ---
    try:
        bal = _http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}",
                        params={"period": "annual", "limit": 5})
    except Exception:
        bal = []
    try:
        inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                        params={"period": "annual", "limit": 5})
    except Exception:
        inc = []
    try:
        cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}",
                       params={"period": "annual", "limit": 5})
    except Exception:
        cf = []
    try:
        km = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{sym}",
                       params={"period": "annual", "limit": 6})
    except Exception:
        km = []

    # Asset growth
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]; _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0) / a0 if (a0 not in (None, 0)) else None

    # Accruals/TA
    ni = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf, "netCashProvidedByOperatingActivities")
    ta = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1]; _, cfo1 = cfo[-1]
        _, ta1 = ta[-1]; _, ta0 = ta[-2]
        avg_assets = None
        if ta1 is not None and ta0 is not None:
            avg_assets = (ta1 + ta0) / 2.0
        accruals = None if (ni1 is None or cfo1 is None) else (ni1 - cfo1)
        out["accruals_ta"] = (accruals / avg_assets) if (accruals is not None and avg_assets not in (None, 0)) else None

    # Net issuance (preferir key-metrics; si no, balance)
    shares_km = _yr_series(km, "sharesOutstanding")
    shares_bs = _yr_series(bal, "commonStockSharesOutstanding")
    seq = shares_km if len(shares_km) >= 2 else shares_bs
    if len(seq) >= 2:
        _, s0 = seq[-2]; _, s1 = seq[-1]
        out["net_issuance"] = (s1 - s0) / s0 if (s0 not in (None, 0)) else None

    # NetDebt/EBITDA: anual directo o reconstrucción
    nd_eb = None
    if isinstance(km, list) and km:
        for item in reversed(km):
            nd = _safe_float(item.get("netDebt"))
            eb = _safe_float(item.get("ebitda"))
            if nd is not None and eb not in (None, 0):
                nd_eb = nd / eb
                break
    if nd_eb is None:
        try:
            b0 = _first_obj(_http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}",
                                      params={"period": "annual", "limit": 1}))
            i0 = _first_obj(_http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                                      params={"period": "annual", "limit": 1}))
            total_debt = _safe_float(b0.get("totalDebt")) or _safe_float(b0.get("shortTermDebt"))
            cash_eq = _safe_float(b0.get("cashAndCashEquivalents")) or 0.0
            eb = _safe_float(i0.get("ebitda"))
            if total_debt is not None and eb not in (None, 0):
                nd_eb = (total_debt - (cash_eq or 0.0)) / eb
        except Exception:
            pass

    out["netdebt_ebitda"] = nd_eb
    return out


def download_guardrails_batch(symbols: List[str],
                              cache_key: str | None = None,
                              force: bool = False) -> pd.DataFrame:
    key = f"guard_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    rows = []
    for s in symbols:
        try:
            rows.append(download_guardrails(s))
        except Exception:
            rows.append({"symbol": s})
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key:
        save_df(df, key)
    return df


# helper: devuelve serie numérica; si no existe la columna, crea NaN
def _num_or_nan(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index)
    return pd.to_numeric(d[col], errors="coerce")

def apply_quality_guardrails(df: pd.DataFrame,
                             require_profit_floor: bool = True,
                             profit_floor_min_hits: int = 2,   # de {EBIT>0, CFO>0, FCF>0}
                             max_net_issuance: float = 0.03,
                             max_asset_growth: float = 0.20,
                             max_accruals_ta: float = 0.10,
                             max_netdebt_ebitda: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica umbrales de guardrails y devuelve:
      - df_filtrado (cumplen todos)
      - df_diag (con flags/diagnóstico)
    Robusto a columnas faltantes.
    """
    d = df.copy()

    # Asegurar columnas como series numéricas (o NaN)
    ebit = _num_or_nan(d, "ebit_ttm")
    cfo  = _num_or_nan(d, "cfo_ttm")
    fcf  = _num_or_nan(d, "fcf_ttm")
    neti = _num_or_nan(d, "net_issuance")
    ag   = _num_or_nan(d, "asset_growth")
    acc  = _num_or_nan(d, "accruals_ta")
    ndeb = _num_or_nan(d, "netdebt_ebitda")

    # Profit floor (series booleanas)
    ebit_ok = (ebit > 0)
    cfo_ok  = (cfo  > 0)
    fcf_ok  = (fcf  > 0)
    d["profit_hits"] = ebit_ok.astype(int) + cfo_ok.astype(int) + fcf_ok.astype(int)
    if require_profit_floor:
        profit_pass = (d["profit_hits"] >= int(profit_floor_min_hits))
    else:
        profit_pass = pd.Series(True, index=d.index)

    # Otros guardrails (NaN-safe)
    issuance_pass = (neti.fillna(0) <= float(max_net_issuance))
    asset_pass    = (ag.abs()      <= float(max_asset_growth))
    accruals_pass = (acc.abs()     <= float(max_accruals_ta))
    # Permitimos NaN en netdebt/EBITDA como "no bloquear"
    lev_pass      = (ndeb.fillna(0) <= float(max_netdebt_ebitda)) | ndeb.isna()

    mask = profit_pass & issuance_pass & asset_pass & accruals_pass & lev_pass

    # flags de diagnóstico (todas series del mismo tamaño)
    d["guard_profit"]   = profit_pass
    d["guard_issuance"] = issuance_pass
    d["guard_assets"]   = asset_pass
    d["guard_accruals"] = accruals_pass
    d["guard_leverage"] = lev_pass
    d["guard_all"]      = mask

    return d[mask].copy(), d


def build_vfq_scores_dynamic(
    df: pd.DataFrame,
    value_metrics: list[str],
    quality_metrics: list[str],
    w_value: float = 0.5,
    w_quality: float = 0.5,
    method_intra: str = "mean",    # "mean" | "median" | "weighted_mean" (equipesos)
    winsor_p: float = 0.01,
    size_buckets: int = 3,
    group_mode: str = "sector",    # "sector" | "sector|size"
) -> pd.DataFrame:
    df = df.copy()

    # Helpers
    def _numcol(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    def _winsor(s: pd.Series, p: float):
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all() or p <= 0:
            return s
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)

    # ——— derivadas mínimas si faltan (ya las tienes, pero por si acaso)
    if "inv_ev_ebitda" in value_metrics and "inv_ev_ebitda" not in df.columns:
        ev = _numcol("evToEbitda")
        df["inv_ev_ebitda"] = (1.0 / ev).replace([np.inf, -np.inf], np.nan)

    if "fcf_yield" in value_metrics and "fcf_yield" not in df.columns:
        df["fcf_yield"] = (_numcol("fcf_ttm") / _numcol("marketCap_unified")).replace([np.inf, -np.inf], np.nan)

    if "gross_profitability" in quality_metrics and "gross_profitability" not in df.columns:
        df["gross_profitability"] = (_numcol("grossProfitTTM") / _numcol("totalAssetsTTM")).replace([np.inf, -np.inf], np.nan)

    # Conjuntos reales disponibles
    V = [c for c in value_metrics if c in df.columns]
    Q = [c for c in quality_metrics if c in df.columns]

    # Winsorizar todo lo que se usa
    for c in set(V + Q):
        df[c] = _winsor(df[c], winsor_p)

    # Coverage (solo sobre columnas efectivamente disponibles)
    use_cols = V + Q
    df["coverage_count"] = df[use_cols].notna().sum(axis=1) if use_cols else 0

    # Agrupamiento: sector o sector|size
    # Tamaño por cuantiles (si se pide)
    if size_buckets > 1:
        mcap = _numcol("marketCap_unified")
        r = mcap.rank(method="first", na_option="keep")
        try:
            size_bucket = pd.qcut(r, size_buckets, labels=False, duplicates="drop")
        except Exception:
            size_bucket = pd.Series(np.nan, index=df.index)
    else:
        size_bucket = pd.Series(0, index=df.index)

    df["sector"] = df.get("sector", "Unknown").fillna("Unknown").astype(str)
    grp_key = df["sector"] if group_mode == "sector" else df["sector"].astype(str) + "|" + size_bucket.astype(str)

    # Ranking por grupo (↑ mejor => descending)
    def _rank_group(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.groupby(grp_key).rank(method="average", ascending=False, na_option="bottom")

    def _block_score(cols):
        if not cols:
            return pd.Series(np.nan, index=df.index)
        ranks = pd.concat([_rank_group(c) for c in cols], axis=1)
        if method_intra == "median":
            return ranks.median(axis=1)
        # weighted_mean: pesos iguales → mean
        return ranks.mean(axis=1)

    df["ValueScore"]   = _block_score(V)
    df["QualityScore"] = _block_score(Q)

    # VFQ con pesos de bloques
    w_sum = (w_value or 0) + (w_quality or 0)
    if w_sum == 0:
        w_value = w_quality = 0.5
        w_sum = 1.0
    df["VFQ"] = (df["ValueScore"] * w_value + df["QualityScore"] * w_quality) / w_sum

    # Percentil intra-sector (robusto)
    try:
        df["VFQ_pct_sector"] = df.groupby("sector")["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = df["VFQ_pct_sector"].fillna(1.0)

    return df
