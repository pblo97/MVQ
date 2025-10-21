# qvm_trend/fundamentals.py
from __future__ import annotations

import math
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

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

def _fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> dict:
    """
    Descarga un set mínimo para construir VFQ con múltiples fallbacks:
      - evToEbitda (TTM → annual)
      - fcf_ttm (TTM → annual cash-flow)
      - grossProfit / totalAssets (TTM → annual income/balance)
      - ratios TTM (ROIC, ROA, netMargin) → annual ratios si faltan
      - sharesOutstanding, marketCap (con fallback a /market-capitalization)
    """
    sym = (symbol or "").strip().upper()
    out = {"symbol": sym}

    # --- Ratios TTM: ROIC/ROA/netMargin
    try:
        rttm = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}")
        r0 = _first_obj(rttm)
        out["roa"]       = _safe_float(r0.get("returnOnAssetsTTM"))
        out["roic"]      = _safe_float(r0.get("returnOnCapitalEmployedTTM") or r0.get("returnOnInvestedCapitalTTM"))
        out["netMargin"] = _safe_float(r0.get("netProfitMarginTTM"))
    except Exception:
        pass
    # Fallback ratios annual
    if any(out.get(k) is None for k in ["roa", "roic", "netMargin"]):
        try:
            ra = _http_get(f"https://financialmodelingprep.com/api/v3/ratios/{sym}", params={"period": "annual", "limit": 1})
            r1 = _first_obj(ra)
            out["roa"]       = out.get("roa")       or _safe_float(r1.get("returnOnAssets"))
            out["roic"]      = out.get("roic")      or _safe_float(r1.get("returnOnCapitalEmployed"))
            out["netMargin"] = out.get("netMargin") or _safe_float(r1.get("netProfitMargin"))
        except Exception:
            pass

    # --- Key-metrics TTM: EV/EBITDA, FCF, GrossProfit, TotalAssets, Shares, MarketCap
    evToEbitda = fcf_ttm = gp_ttm = ta_ttm = shares = mcap = None
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        k0 = _first_obj(kttm)
        evToEbitda = _safe_float(k0.get("enterpriseValueOverEBITDATTM"))
        fcf_ttm    = _safe_float(k0.get("freeCashFlowTTM"))
        gp_ttm     = _safe_float(k0.get("grossProfitTTM"))
        ta_ttm     = _safe_float(k0.get("totalAssetsTTM"))
        shares     = _safe_float(k0.get("sharesOutstanding"))
        mcap       = _safe_float(k0.get("marketCap"))
    except Exception:
        pass

    # Fallback key-metrics annual
    if any(v is None for v in [evToEbitda, fcf_ttm, gp_ttm, ta_ttm, mcap, shares]):
        try:
            km1 = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{sym}", params={"period": "annual", "limit": 1})
            z = _first_obj(km1)
            evToEbitda = evToEbitda or _safe_float(z.get("enterpriseValueOverEBITDA"))
            fcf_ttm    = fcf_ttm    or _safe_float(z.get("freeCashFlow"))
            gp_ttm     = gp_ttm     or _safe_float(z.get("grossProfit"))
            ta_ttm     = ta_ttm     or _safe_float(z.get("totalAssets"))
            shares     = shares     or _safe_float(z.get("sharesOutstanding"))
            mcap       = mcap       or _safe_float(z.get("marketCap"))
        except Exception:
            pass

    # Fallback final para grossProfit/totalAssets desde income/balance anual
    if gp_ttm is None or ta_ttm is None:
        try:
            inc_a = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period": "annual", "limit": 1})
            bal_a = _http_get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}", params={"period": "annual", "limit": 1})
            i0 = _first_obj(inc_a); b0 = _first_obj(bal_a)
            gp_ttm = gp_ttm or _safe_float(i0.get("grossProfit"))
            ta_ttm = ta_ttm or _safe_float(b0.get("totalAssets"))
        except Exception:
            pass

    # Market cap fallback
    if not mcap:
        try:
            mc = _http_get(f"https://financialmodelingprep.com/api/v3/market-capitalization/{sym}", params={"limit": 1})
            m0 = _first_obj(mc)
            mcap = _safe_float(m0.get("marketCap"))
        except Exception:
            pass
    if not mcap and market_cap_hint:
        mcap = float(market_cap_hint)

    # Output crudo
    out["evToEbitda"]        = evToEbitda
    out["fcf_ttm"]           = fcf_ttm
    out["grossProfitTTM"]    = gp_ttm
    out["totalAssetsTTM"]    = ta_ttm
    out["sharesOutstanding"] = shares
    out["marketCap"]         = mcap

    # Derivados VFQ
    out["inv_ev_ebitda"]        = (1.0 / evToEbitda) if (evToEbitda not in (None, 0)) else None
    out["fcf_yield"]            = (fcf_ttm / mcap)   if (fcf_ttm is not None and mcap not in (None, 0)) else None
    out["gross_profitability"]  = (gp_ttm / ta_ttm)  if (gp_ttm is not None and ta_ttm not in (None, 0)) else None
    return out


def download_fundamentals(symbols: List[str], market_caps=None, cache_key=None, force=False) -> pd.DataFrame:
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    rows = []
    mc_map = market_caps or {}
    for s in symbols:
        try:
            rows.append(_fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s)))
        except Exception as e:
            rows.append({"symbol": s, "__err_fund": str(e)[:180]})
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key: save_df(df, key)
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


def build_vfq_scores(df_base: pd.DataFrame, df_fund: pd.DataFrame) -> pd.DataFrame:
    """
    Une universo + fundamentales y construye:
      - fcf_yield / inv_ev_ebitda / gross_profitability
      - ValueScore, QualityScore, VFQ y VFQ_pct_sector
      - coverage_count
    Robusto a columnas faltantes.
    """
    if df_base is None or df_base.empty:
        return pd.DataFrame()

    dfb = df_base.copy()

    # Asegurar columnas mínimas en el universo
    if "symbol" not in dfb.columns:
        raise ValueError("df_base debe contener al menos la columna 'symbol'.")
    if "sector" not in dfb.columns:
        dfb["sector"] = "Unknown"
    if "marketCap" not in dfb.columns:
        dfb["marketCap"] = np.nan
    if "price" not in dfb.columns:
        dfb["price"] = np.nan

    # Merge con fundamentales (puede venir vacío)
    dff = df_fund.copy() if (df_fund is not None and not df_fund.empty) else pd.DataFrame(columns=["symbol"])
    df = dfb.merge(dff, on="symbol", how="left")

    # Reafirmar columnas críticas por si el merge las perdió
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "marketCap" not in df.columns:
        df["marketCap"] = np.nan

    # Series numéricas seguras
    ev   = _num_or_nan(df, "evToEbitda")
    mcap = _num_or_nan(df, "marketCap")
    fcf  = _num_or_nan(df, "fcf_ttm")
    gp   = _num_or_nan(df, "grossProfitTTM")
    ta   = _num_or_nan(df, "totalAssetsTTM")

    # Derivadas VFQ
    ev_nz = ev.replace(0, np.nan)
    df["inv_ev_ebitda"] = np.where(ev_nz.notna(), 1.0 / ev_nz, np.nan)
    df["fcf_yield"] = np.where((mcap.notna()) & (mcap != 0) & fcf.notna(), fcf / mcap, np.nan)
    df["gross_profitability"] = np.where((ta.notna()) & (ta != 0) & gp.notna(), gp / ta, np.nan)

    # Winsor suave
    for c in ["fcf_yield", "inv_ev_ebitda", "gross_profitability", "roic", "roa", "netMargin"]:
        if c in df.columns:
            df[c] = _winsorize(pd.to_numeric(df[c], errors="coerce"), 0.01)

    # Cobertura VFQ
    vfq_cols_all = ["fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"]
    vfq_cols = [c for c in vfq_cols_all if c in df.columns]
    df["coverage_count"] = df[vfq_cols].notna().sum(axis=1) if vfq_cols else 0

    # Buckets de tamaño
    r = df["marketCap"].rank(method="first", na_option="keep")
    try:
        df["size_bucket"] = pd.qcut(r, 3, labels=False, duplicates="drop")
    except Exception:
        pct = r / r.max() if (hasattr(r, "max") and r.max() and r.max() > 0) else r
        df["size_bucket"] = np.select([pct <= 0.33, pct <= 0.66, pct > 0.66], [0, 1, 2], default=np.nan)

    # Sector seguro
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    grp = df["sector"].astype(str) + "|" + df["size_bucket"].astype(str)

    # Ranking por grupo (si existen columnas)
    def _rank_by_group(s: pd.Series, ascending: bool, group: pd.Series) -> pd.Series:
        return s.groupby(group).rank(method="average", ascending=ascending, na_option="bottom")

    val_inputs = [c for c in ["fcf_yield","inv_ev_ebitda"] if c in df.columns]
    q_inputs   = [c for c in ["gross_profitability","roic","roa","netMargin"] if c in df.columns]

    df["ValueScore"] = (pd.concat([_rank_by_group(df[c], False, grp) for c in val_inputs], axis=1).mean(axis=1)
                        if val_inputs else np.nan)
    df["QualityScore"] = (pd.concat([_rank_by_group(df[c], False, grp) for c in q_inputs], axis=1).mean(axis=1)
                          if q_inputs else np.nan)

    df["VFQ"] = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1)
    df["VFQ_pct_sector"] = df.groupby("sector")["VFQ"].rank(pct=True)

    # Orden amigable
    cols = ["symbol","sector","marketCap","coverage_count",
            "fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin",
            "ValueScore","QualityScore","VFQ","VFQ_pct_sector"]
    cols = [c for c in cols if c in df.columns]
    return df[cols + [c for c in df.columns if c not in cols]].copy()

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
