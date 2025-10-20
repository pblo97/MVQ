# mvq/qvm_trend/fundamentals.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from .data_io import _http_get
from .cache_io import save_df, load_df

__all__ = [
    "download_fundamentals", "build_vfq_scores",
    "download_guardrails_batch", "apply_quality_guardrails"
]

# ============================== UTILIDADES ==============================

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def coalesce_first(df: pd.DataFrame, candidates: list[str], new_col: str, to_numeric: bool=False) -> pd.DataFrame:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df[new_col] = pd.Series([None]*len(df), index=df.index)
        return df
    buf = pd.concat([df[c] for c in cols], axis=1)
    out = buf.bfill(axis=1).iloc[:, 0]
    if to_numeric:
        out = pd.to_numeric(out, errors="coerce")
    df[new_col] = out
    return df

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def _yr_series(items, key):
    out = []
    for it in (items or []):
        d = it.get("date"); v = _safe_num(it.get(key))
        if d is not None and v is not None:
            out.append((pd.to_datetime(d), v))
    out.sort(key=lambda z: z[0])
    return out

def _first_item(obj):
    """Devuelve primer item para endpoints que pueden ser dict o list."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj:
        return obj[0]
    return None

def _sum_last_quarters(items, key, n=4):
    """Suma los últimos n trimestres (quarterly)."""
    if not isinstance(items, list) or not items:
        return None
    try:
        df = pd.DataFrame(items)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        vals = pd.to_numeric(df[key], errors="coerce").dropna().tail(n)
        if len(vals) == 0:
            return None
        return float(vals.sum())
    except Exception:
        return None

# ==================== SET MÍNIMO (DESCARGA + CACHE) ====================

def _fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> dict:
    """
    Mínimos robustos para VFQ con fallbacks:
      - ROA/ROIC/Margen: ratios-ttm -> ratios (annual)
      - EV/EBITDA, FCF, GrossProfit, TotalAssets, Shares, MarketCap: key-metrics-ttm -> key-metrics (annual)
      - fcf_yield, gross_profitability, inv_ev_ebitda
    """
    sym = (symbol or "").strip().upper()
    out: Dict[str, object] = {"symbol": sym}

    # Ratios: TTM → annual
    roa = roic = net_m = None
    try:
        r = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}")
        if isinstance(r, list) and r:
            x = r[0]
            roa   = x.get("returnOnAssetsTTM")
            roic  = x.get("returnOnCapitalEmployedTTM") or x.get("returnOnInvestedCapitalTTM")
            net_m = x.get("netProfitMarginTTM")
    except Exception:
        pass
    if roa is None or roic is None or net_m is None:
        try:
            r1 = _http_get(f"https://financialmodelingprep.com/api/v3/ratios/{sym}",
                           params={"period": "annual", "limit": 1})
            if isinstance(r1, list) and r1:
                y = r1[0]
                if roa   is None: roa   = y.get("returnOnAssets")
                if roic  is None: roic  = y.get("returnOnCapitalEmployed") or y.get("returnOnInvestedCapital")
                if net_m is None: net_m = y.get("netProfitMargin")
        except Exception:
            pass
    out["roa"] = roa
    out["roic"] = roic
    out["netMargin"] = net_m

    # Key-metrics: TTM → annual
    evToEbitda = fcf_ttm = gp_ttm = ta_ttm = shares = mcap = None
    try:
        k = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        if isinstance(k, list) and k:
            y = k[0]
            evToEbitda = y.get("enterpriseValueOverEBITDATTM")
            fcf_ttm    = y.get("freeCashFlowTTM")
            gp_ttm     = y.get("grossProfitTTM")
            ta_ttm     = y.get("totalAssetsTTM")
            shares     = y.get("sharesOutstanding")
            mcap       = y.get("marketCap") or mcap
    except Exception:
        pass
    if any(v is None for v in [evToEbitda, fcf_ttm, gp_ttm, ta_ttm, mcap]):
        try:
            km1 = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{sym}",
                            params={"period": "annual", "limit": 1})
            if isinstance(km1, list) and km1:
                z = km1[0]
                evToEbitda = evToEbitda or z.get("enterpriseValueOverEBITDA")
                fcf_ttm    = fcf_ttm    or z.get("freeCashFlow")
                gp_ttm     = gp_ttm     or z.get("grossProfit")
                ta_ttm     = ta_ttm     or z.get("totalAssets")
                shares     = shares     or z.get("sharesOutstanding")
                mcap       = mcap       or z.get("marketCap")
        except Exception:
            pass

    out["evToEbitda"]        = evToEbitda
    out["fcf_ttm"]           = fcf_ttm
    out["grossProfitTTM"]    = gp_ttm
    out["totalAssetsTTM"]    = ta_ttm
    out["sharesOutstanding"] = shares

    # Market cap fallback
    if not mcap:
        try:
            mc = _http_get(f"https://financialmodelingprep.com/api/v3/market-capitalization/{sym}", params={"limit": 1})
            if isinstance(mc, list) and mc:
                mcap = mc[0].get("marketCap")
        except Exception:
            pass
    out["marketCap"] = mcap or market_cap_hint

    # Derivados
    fcf = out.get("fcf_ttm"); gp = out.get("grossProfitTTM"); ta = out.get("totalAssetsTTM"); ev = out.get("evToEbitda")
    mcap = out.get("marketCap")

    out["fcf_yield"] = (fcf / mcap) if (fcf not in (None, 0) and mcap not in (None, 0)) else None
    out["gross_profitability"] = (gp / ta) if (gp not in (None, 0) and ta not in (None, 0)) else None
    out["inv_ev_ebitda"] = (1.0 / ev) if (ev not in (None, 0)) else None

    fields = ["fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"]
    out["coverage_count"] = int(sum(out.get(f) is not None for f in fields))
    return out

def download_fundamentals(symbols: List[str],
                          market_caps: Dict[str, float] | None = None,
                          cache_key: str | None = None,
                          force: bool=False) -> pd.DataFrame:
    """
    Descarga set mínimo para todos los símbolos y cachea (parquet).
    """
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    rows: List[dict] = []
    mc_map = market_caps or {}
    for s in symbols:
        try:
            rows.append(_fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s)))
        except Exception:
            rows.append({"symbol": s})

    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key:
        save_df(df, key)
    return df

# ======================= VFQ (RANKS + NEUTRALIZACIÓN) =======================

def build_vfq_scores(df_universe: pd.DataFrame,
                     df_fund: pd.DataFrame,
                     size_buckets: int = 3) -> pd.DataFrame:
    """
    Une universo (sector/marketCap) + fundamentales TTM/anual y calcula:
      ValueScore = mean(rank(fcf_yield, 1/EV/EBITDA))
      QualityScore = mean(rank(gross_profitability, roic, roa, netMargin))
      VFQ = mean(ValueScore, QualityScore)
      VFQ_pct_sector = percentil intra-sector
    Robusto a columnas faltantes.
    """
    dfu = df_universe.copy()
    dff = df_fund.copy()
    df = dfu.merge(dff, on="symbol", how="left")

    # Unificar sector / market cap
    df = coalesce_first(df, ["sector","sector_x","sector_y"], "sector_unified", to_numeric=False)
    df["sector_unified"] = df["sector_unified"].fillna("Unknown").astype(str)
    df = coalesce_first(df, ["marketCap","marketCap_x","marketCap_y","marketCap_profile"], "marketCap_unified", to_numeric=True)

    # Buckets de tamaño
    r = df["marketCap_unified"].rank(method="first", na_option="keep")
    try:
        df["size_bucket"] = pd.qcut(r, size_buckets, labels=False, duplicates="drop")
    except Exception:
        df["size_bucket"] = 1

    # Métricas esperadas
    val_candidates = ["fcf_yield", "inv_ev_ebitda"]
    qlt_candidates  = ["gross_profitability", "roic", "roa", "netMargin"]
    all_fields = val_candidates + qlt_candidates

    # Winsor solo sobre columnas existentes
    for c in all_fields:
        if c in df.columns:
            df[c] = winsorize(pd.to_numeric(df[c], errors="coerce"), 0.01)

    # Grupo: sector × tamaño
    grp = df["sector_unified"].astype(str) + "|" + df["size_bucket"].astype(str)

    def _rank_if_exists(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        mat = pd.concat([df[c] for c in present], axis=1)
        ranks = pd.concat(
            [mat[c].groupby(grp).rank(method="average", ascending=False, na_option="bottom")
             for c in present], axis=1
        )
        return ranks.mean(axis=1)

    # Scores
    df["ValueScore"]   = _rank_if_exists(val_candidates)
    df["QualityScore"] = _rank_if_exists(qlt_candidates)
    df["VFQ"] = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1)

    # Percentil intra-sector
    try:
        df["VFQ_pct_sector"] = df.groupby("sector_unified")["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = np.nan

    # Cobertura VFQ
    existing = [f for f in all_fields if f in df.columns]
    if existing:
        df["coverage_count"] = df[existing].notna().sum(axis=1)
    else:
        df["coverage_count"] = 0

    keep = ["symbol","sector_unified","marketCap_unified","coverage_count",
            "fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin",
            "ValueScore","QualityScore","VFQ","VFQ_pct_sector"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# =========================== GUARDRAILS (CALIDAD) ===========================

def download_guardrails(symbol: str) -> dict:
    """
    TTM + fallbacks:
      - Key-metrics-ttm (net_debt_ttm, ebitda_ttm) → annual si falta
      - CFO/FCF: ttm → quarterly (sum 4) → annual
      - EBIT: ttm → quarterly (sum 4) → annual
      - Net issuance, asset growth, accruals/TA, netdebt/EBITDA (annual)
    """
    sym = (symbol or "").strip().upper()
    out: Dict[str, object] = {"symbol": sym}

    # --- TTM básicos (key-metrics-ttm → annual)
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        k = _first_item(kttm)
        if k:
            out["shares_out_ttm"] = _safe_num(k.get("sharesOutstanding"))
            out["net_debt_ttm"]   = _safe_num(k.get("netDebtTTM")) or None
            out["ebitda_ttm"]     = _safe_num(k.get("ebitdaTTM")) or None
    except Exception:
        pass

    # --- CFO/FCF TTM → quarterly sum → annual
    out["cfo_ttm"] = None
    out["fcf_ttm"] = None
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{sym}")
        c = _first_item(cfttm)
        if c:
            out["cfo_ttm"] = _safe_num(c.get("netCashProvidedByOperatingActivitiesTTM"))
            out["fcf_ttm"] = _safe_num(c.get("freeCashFlowTTM"))
    except Exception:
        pass
    if out.get("cfo_ttm") is None or out.get("fcf_ttm") is None:
        try:
            cf_q = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}", params={"period":"quarter","limit":6})
        except Exception:
            cf_q = []
        if out.get("cfo_ttm") is None:
            s = _sum_last_quarters(cf_q, "netCashProvidedByOperatingActivities", 4)
            if s is not None: out["cfo_ttm"] = s
        if out.get("fcf_ttm") is None:
            s = _sum_last_quarters(cf_q, "freeCashFlow", 4)
            if s is not None: out["fcf_ttm"] = s
    if out.get("cfo_ttm") is None or out.get("fcf_ttm") is None:
        try:
            cf_a = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}", params={"period":"annual","limit":1})
            if isinstance(cf_a, list) and cf_a:
                out["cfo_ttm"] = out["cfo_ttm"] or _safe_num(cf_a[0].get("netCashProvidedByOperatingActivities"))
                out["fcf_ttm"] = out["fcf_ttm"] or _safe_num(cf_a[0].get("freeCashFlow"))
        except Exception:
            pass

    # --- EBIT TTM → quarterly sum → annual
    out["ebit_ttm"] = None
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        it = _first_item(inc_ttm)
        if it:
            out["ebit_ttm"] = _safe_num(it.get("ebitTTM") or it.get("operatingIncomeTTM"))
    except Exception:
        pass
    if out.get("ebit_ttm") is None:
        try:
            inc_q = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period":"quarter","limit":6})
        except Exception:
            inc_q = []
        s = _sum_last_quarters(inc_q, "ebit", 4)
        if s is None:
            s = _sum_last_quarters(inc_q, "operatingIncome", 4)
        if s is not None:
            out["ebit_ttm"] = s
    if out.get("ebit_ttm") is None:
        try:
            inc_a = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period":"annual","limit":1})
            if isinstance(inc_a, list) and inc_a:
                out["ebit_ttm"] = _safe_num(inc_a[0].get("ebit") or inc_a[0].get("operatingIncome"))
        except Exception:
            pass

    # --- Series anuales: growth, accruals, issuance, leverage
    try:
        bal = _http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}", params={"period":"annual","limit":5})
    except Exception:
        bal = []
    try:
        inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period":"annual","limit":5})
    except Exception:
        inc = []
    try:
        cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}", params={"period":"annual","limit":5})
    except Exception:
        cf = []
    try:
        km = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{sym}", params={"period":"annual","limit":6})
    except Exception:
        km = []

    # Asset growth (y/y)
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]; _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0)/a0 if (a0 not in (None, 0)) else None

    # Accruals/TA ~ (NI − CFO) / activos medios
    ni  = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf,  "netCashProvidedByOperatingActivities")
    ta  = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1]; _, cfo1 = cfo[-1]
        _, ta1 = ta[-1]; _, ta0 = ta[-2]
        accruals   = (ni1 - cfo1)
        avg_assets = ((ta1 or 0.0) + (ta0 or 0.0))/2.0 if (ta1 and ta0) else None
        out["accruals_ta"] = (accruals/avg_assets) if (avg_assets not in (None, 0)) else None

    # Net issuance (Δ acciones 12–24m) usando key-metrics anual
    shares = _yr_series(km, "sharesOutstanding")
    if len(shares) >= 2:
        _, s0 = shares[-2]; _, s1 = shares[-1]
        out["net_issuance"] = (s1 - s0)/s0 if (s0 not in (None, 0)) else None
    else:
        out["net_issuance"] = None

    # NetDebt/EBITDA (último anual con ambos)
    net_debt = None; ebitda = None
    if isinstance(km, list) and km:
        for item in reversed(km):
            nd = _safe_num(item.get("netDebt")); eb = _safe_num(item.get("ebitda"))
            if nd is not None and eb not in (None, 0):
                net_debt, ebitda = nd, eb
                break
    out["netdebt_ebitda"] = (net_debt/ebitda) if (net_debt is not None and ebitda not in (None, 0)) else None

    return out

def download_guardrails_batch(symbols: list[str], cache_key: str | None = None, force: bool=False) -> pd.DataFrame:
    key = f"guard_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc
    rows: List[dict] = []
    for s in symbols:
        try:
            rows.append(download_guardrails(s))
        except Exception:
            rows.append({"symbol": s})
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key:
        save_df(df, key)
    return df

def apply_quality_guardrails(df: pd.DataFrame,
                             require_profit_floor: bool = True,
                             profit_floor_min_hits: int = 2,
                             max_net_issuance: float = 0.03,
                             max_asset_growth: float = 0.20,
                             max_accruals_ta: float = 0.10,
                             max_netdebt_ebitda: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robustez:
      - Columnas faltantes -> Serie NaN
      - Conversión a numérico antes de comparar
    Devuelve (subset_que_pasa, diagnóstico_completo_con_flags).
    """
    d = df.copy()

    def col(name: str) -> pd.Series:
        if name in d.columns:
            return pd.to_numeric(d[name], errors="coerce")
        return pd.Series(np.nan, index=d.index, dtype="float64")

    ebit = col("ebit_ttm")
    cfo  = col("cfo_ttm")
    fcf  = col("fcf_ttm")

    ebit_ok = (ebit > 0)
    cfo_ok  = (cfo  > 0)
    fcf_ok  = (fcf  > 0)

    d["profit_hits"] = ebit_ok.fillna(False).astype(int) + \
                       cfo_ok.fillna(False).astype(int) + \
                       fcf_ok.fillna(False).astype(int)
    profit_pass = (d["profit_hits"] >= int(profit_floor_min_hits)) if require_profit_floor else pd.Series(True, index=d.index)

    net_issuance   = col("net_issuance")
    asset_growth   = col("asset_growth")
    accruals_ta    = col("accruals_ta")
    netdebt_ebitda = col("netdebt_ebitda")

    issuance_pass = (net_issuance.fillna(0) <= float(max_net_issuance))
    asset_pass    = (asset_growth.abs() <= float(max_asset_growth))
    accruals_pass = (accruals_ta.abs() <= float(max_accruals_ta))
    lev_pass      = (netdebt_ebitda.fillna(0) <= float(max_netdebt_ebitda)) | netdebt_ebitda.isna()

    mask = profit_pass & issuance_pass & asset_pass & accruals_pass & lev_pass

    d["guard_profit"]   = profit_pass
    d["guard_issuance"] = issuance_pass
    d["guard_assets"]   = asset_pass
    d["guard_accruals"] = accruals_pass
    d["guard_leverage"] = lev_pass
    d["guard_all"]      = mask

    return d[mask].copy(), d
