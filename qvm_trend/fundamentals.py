# qvm_trend/fundamentals.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict

from .data_io import _http_get
from .cache_io import save_df, load_df


# =============================================================================
# SET MÍNIMO DE BATALLA (TTM) + CARGA/CACHE
# =============================================================================

def fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> dict:
    """
    Descarga fundamentales TTM mínimos robustos para VFQ:
      - fcf_yield, inv_ev_ebitda, gross_profitability, roic, roa, netMargin
      - marketCap (con fallback)
      - coverage_count (# de métricas válidas)
    """
    sym = (symbol or "").strip().upper()
    out: Dict[str, object] = {"symbol": sym}

    # Ratios TTM: ROA/ROIC/Net Margin
    try:
        r = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}")
        if isinstance(r, list) and r:
            x = r[0]
            out["roa"] = x.get("returnOnAssetsTTM")
            out["roic"] = x.get("returnOnCapitalEmployedTTM") or x.get("returnOnInvestedCapitalTTM")
            out["netMargin"] = x.get("netProfitMarginTTM")
    except Exception:
        pass

    # Key-metrics TTM: EV/EBITDA, FCF, GrossProfit, TotalAssets, Shares, (a veces) MarketCap
    try:
        k = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        if isinstance(k, list) and k:
            y = k[0]
            out["evToEbitda"]       = y.get("enterpriseValueOverEBITDATTM")
            out["fcf_ttm"]          = y.get("freeCashFlowTTM")
            out["grossProfitTTM"]   = y.get("grossProfitTTM")
            out["totalAssetsTTM"]   = y.get("totalAssetsTTM")
            out["sharesOutstanding"]= y.get("sharesOutstanding")
            if y.get("marketCap"):
                out["marketCap"] = y.get("marketCap")
    except Exception:
        pass

    # Fallback de MarketCap si no vino en key-metrics-ttm
    if "marketCap" not in out or out["marketCap"] in (None, 0):
        try:
            mc = _http_get(f"https://financialmodelingprep.com/api/v3/market-capitalization/{sym}",
                           params={"limit": 1})
            if isinstance(mc, list) and mc:
                out["marketCap"] = mc[0].get("marketCap")
        except Exception:
            out["marketCap"] = market_cap_hint

    # Derivados principales
    fcf = out.get("fcf_ttm")
    mcap = out.get("marketCap")
    gp   = out.get("grossProfitTTM")
    ta   = out.get("totalAssetsTTM")
    ev   = out.get("evToEbitda")

    out["fcf_yield"] = (fcf / mcap) if (fcf not in (None, 0) and mcap not in (None, 0)) else None
    out["gross_profitability"] = (gp / ta) if (gp not in (None, 0) and ta not in (None, 0)) else None
    out["inv_ev_ebitda"] = (1.0 / ev) if (ev not in (None, 0)) else None

    # Coverage
    fields = ["fcf_yield", "inv_ev_ebitda", "gross_profitability", "roic", "roa", "netMargin"]
    out["coverage_count"] = int(sum(out.get(f) is not None for f in fields))

    return out


def download_fundamentals(symbols: List[str],
                          market_caps: Dict[str, float] | None = None,
                          cache_key: str | None = None,
                          force: bool = False) -> pd.DataFrame:
    """
    Descarga 'set mínimo de batalla' para todos los símbolos y cachea en parquet.
    """
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    rows: List[dict] = []
    mc_map = market_caps or {}
    for s in symbols:
        mc_hint = mc_map.get(s)
        try:
            rows.append(fetch_min_battle_fmp(s, market_cap_hint=mc_hint))
        except Exception:
            rows.append({"symbol": s})

    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key:
        save_df(df, key)
    return df


# =============================================================================
# VFQ: WINSOR + RANKS POR SECTOR×TAMAÑO + COALESCE DE COLUMNAS
# =============================================================================

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def coalesce_first(df: pd.DataFrame, candidates: list[str], new_col: str, to_numeric: bool = False) -> pd.DataFrame:
    """
    Toma las columnas en 'candidates' (si existen), hace 'primer no-nulo' por fila
    y la guarda en 'new_col'. Si to_numeric=True, coerciona a numérico.
    """
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df[new_col] = pd.Series([None] * len(df), index=df.index)
        return df
    buf = pd.concat([df[c] for c in cols], axis=1)
    out = buf.bfill(axis=1).iloc[:, 0]
    if to_numeric:
        out = pd.to_numeric(out, errors="coerce")
    df[new_col] = out
    return df


def build_vfq_scores(df_universe: pd.DataFrame,
                     df_fund: pd.DataFrame,
                     size_buckets: int = 3) -> pd.DataFrame:
    """
    Une universo (sector/marketCap) + fundamentales y calcula:
      - ValueScore = mean(rank(fcf_yield, 1/EV/EBITDA))
      - QualityScore = mean(rank(gross_profitability, roic, roa, netMargin))
      - VFQ = mean(ValueScore, QualityScore)
      - VFQ_pct_sector = percentil intra-sector
    Es robusto a columnas duplicadas (marketCap_x/y, sector_x/y).
    """
    dfu = df_universe.copy()
    dff = df_fund.copy()
    # merge flexible: puede traer marketCap/sector de ambos lados
    df = dfu.merge(dff, on="symbol", how="left")

    # Unificar sector y marketCap (coalesce)
    df = coalesce_first(df, ["sector", "sector_x", "sector_y"], "sector_unified", to_numeric=False)
    df = coalesce_first(df, ["marketCap", "marketCap_x", "marketCap_y", "marketCap_profile"],
                        "marketCap_unified", to_numeric=True)

    # Buckets por tamaño (tolerante si muchos NaN)
    r = df["marketCap_unified"].rank(method="first", na_option="keep")
    try:
        df["size_bucket"] = pd.qcut(r, size_buckets, labels=False, duplicates="drop")
    except Exception:
        df["size_bucket"] = 1

    # Winsor de métricas (suaviza outliers)
    for c in ["fcf_yield", "inv_ev_ebitda", "gross_profitability", "roic", "roa", "netMargin"]:
        if c in df.columns:
            df[c] = winsorize(pd.to_numeric(df[c], errors="coerce"), 0.01)

    # Grupo: sector × tamaño
    grp = df["sector_unified"].astype(str) + "|" + df["size_bucket"].astype(str)

    def _rank(s: pd.Series, asc: bool = False) -> pd.Series:
        return s.groupby(grp).rank(method="average", ascending=asc, na_option="bottom")

    # Value
    v_parts = []
    if "fcf_yield" in df.columns:
        v_parts.append(_rank(df["fcf_yield"], asc=False))
    if "inv_ev_ebitda" in df.columns:
        v_parts.append(_rank(df["inv_ev_ebitda"], asc=False))
    df["ValueScore"] = pd.concat(v_parts, axis=1).mean(axis=1) if len(v_parts) else np.nan

    # Quality
    q_parts = []
    for c in ["gross_profitability", "roic", "roa", "netMargin"]:
        if c in df.columns:
            q_parts.append(_rank(df[c], asc=False))
    df["QualityScore"] = pd.concat(q_parts, axis=1).mean(axis=1) if len(q_parts) else np.nan

    df["VFQ"] = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1)
    df["VFQ_pct_sector"] = df.groupby("sector_unified")["VFQ"].rank(pct=True)

    # Si no vino coverage_count desde df_fund, calcúlalo aquí por si acaso
    if "coverage_count" not in df.columns:
        fields = ["fcf_yield", "inv_ev_ebitda", "gross_profitability", "roic", "roa", "netMargin"]
        df["coverage_count"] = df[fields].notna().sum(axis=1)

    keep = [
        "symbol",
        "sector_unified", "marketCap_unified",
        "coverage_count",
        "fcf_yield", "inv_ev_ebitda", "gross_profitability", "roic", "roa", "netMargin",
        "ValueScore", "QualityScore", "VFQ", "VFQ_pct_sector"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# =============================================================================
# GUARDRAILS (CALIDAD): DESCARGA, CACHE Y FILTRO
# =============================================================================

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None


def _yr_series(items, key):
    """Construye serie anual (list[dict]) -> list[(date, value)] con coerción numérica."""
    out = []
    for it in (items or []):
        d = it.get("date")
        v = _safe_num(it.get(key))
        if d is not None and v is not None:
            out.append((pd.to_datetime(d), v))
    out.sort(key=lambda z: z[0])
    return out


def download_guardrails(symbol: str) -> dict:
    """
    Descarga estados anuales (últ. 4-5) y TTM para calcular guardrails:
      - EBIT_TTM, CFO_TTM, FCF_TTM
      - Net Issuance (12–24m)
      - Asset Growth (y/y)
      - Accruals/TA (anual)
      - NetDebt/EBITDA (últ. año disponible)
    """
    sym = (symbol or "").strip().upper()
    out: Dict[str, object] = {"symbol": sym}

    # TTM básicos
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        if isinstance(kttm, list) and kttm:
            k = kttm[0]
            out["shares_out_ttm"] = _safe_num(k.get("sharesOutstanding"))
            out["net_debt_ttm"]   = _safe_num(k.get("netDebtTTM")) or None
            out["ebitda_ttm"]     = _safe_num(k.get("ebitdaTTM")) or None
    except Exception:
        pass

    # CFO/FCF TTM
    try:
        cfttm = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement-ttm/{sym}")
        if isinstance(cfttm, dict) and cfttm:
            out["cfo_ttm"] = _safe_num(cfttm.get("netCashProvidedByOperatingActivitiesTTM"))
            out["fcf_ttm"] = _safe_num(cfttm.get("freeCashFlowTTM"))
    except Exception:
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}",
                           params={"limit": 1})
            if isinstance(cf, list) and cf:
                out["cfo_ttm"] = _safe_num(cf[0].get("netCashProvidedByOperatingActivities"))
                out["fcf_ttm"] = _safe_num(cf[0].get("freeCashFlow"))
        except Exception:
            pass

    # EBIT TTM (o proxys)
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        if isinstance(inc_ttm, dict) and inc_ttm:
            out["ebit_ttm"] = _safe_num(inc_ttm.get("ebitTTM") or inc_ttm.get("operatingIncomeTTM"))
    except Exception:
        try:
            inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                            params={"period": "annual", "limit": 1})
            if isinstance(inc, list) and inc:
                out["ebit_ttm"] = _safe_num(inc[0].get("ebit") or inc[0].get("operatingIncome"))
        except Exception:
            pass

    # Series anuales para growth, accruals, issuance
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

    # Asset growth (y/y)
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]
        _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0) / a0 if (a0 not in (None, 0)) else None

    # Accruals/TA (aprox): (NI − CFO) / Activos medios (último año)
    ni  = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf,  "netCashProvidedByOperatingActivities")
    ta  = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1];    _, ni0 = ni[-2]
        _, cfo1 = cfo[-1];  _, cfo0 = cfo[-2]
        _, ta1 = ta[-1];    _, ta0 = ta[-2]
        accruals = (ni1 - cfo1)
        avg_assets = ((ta1 or 0.0) + (ta0 or 0.0)) / 2.0 if (ta1 and ta0) else None
        out["accruals_ta"] = (accruals / avg_assets) if (avg_assets not in (None, 0)) else None

    # Net issuance (Δ acciones 12–24m)
    shares = _yr_series(km, "sharesOutstanding")
    if len(shares) >= 2:
        _, s0 = shares[-2]; _, s1 = shares[-1]
        out["net_issuance"] = (s1 - s0) / s0 if (s0 not in (None, 0)) else None
    else:
        out["net_issuance"] = None

    # NetDebt/EBITDA (último anual con ambos)
    net_debt = None
    ebitda = None
    if isinstance(km, list) and km:
        for item in reversed(km):
            nd = _safe_num(item.get("netDebt"))
            eb = _safe_num(item.get("ebitda"))
            if nd is not None and eb not in (None, 0):
                net_debt = nd; ebitda = eb; break
    out["netdebt_ebitda"] = (net_debt / ebitda) if (net_debt is not None and ebitda not in (None, 0)) else None

    return out


def download_guardrails_batch(symbols: list[str],
                              cache_key: str | None = None,
                              force: bool = False) -> pd.DataFrame:
    """
    Descarga guardrails para todos los símbolos y cachea.
    """
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
                             profit_floor_min_hits: int = 2,  # de {EBIT>0, CFO>0, FCF>0}
                             max_net_issuance: float = 0.03,
                             max_asset_growth: float = 0.20,
                             max_accruals_ta: float = 0.10,
                             max_netdebt_ebitda: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica guardrails y devuelve (subset_que_pasa, diagnóstico_con_flags).
    """
    d = df.copy()

    # Pisos de rentabilidad (TTM): conteo de hits
    ebit_ok = (d.get("ebit_ttm") > 0)
    cfo_ok  = (d.get("cfo_ttm")  > 0)
    fcf_ok  = (d.get("fcf_ttm")  > 0)
    d["profit_hits"] = ebit_ok.astype(int) + cfo_ok.astype(int) + fcf_ok.astype(int)
    profit_pass = (d["profit_hits"] >= int(profit_floor_min_hits)) if require_profit_floor else True

    # Resto de guardrails
    issuance_pass = (d.get("net_issuance").fillna(0) <= float(max_net_issuance))
    asset_pass    = (d.get("asset_growth").abs() <= float(max_asset_growth))
    accruals_pass = (d.get("accruals_ta").abs() <= float(max_accruals_ta))
    lev_pass      = (d.get("netdebt_ebitda").fillna(0) <= float(max_netdebt_ebitda)) | d.get("netdebt_ebitda").isna()

    mask = profit_pass & issuance_pass & asset_pass & accruals_pass & lev_pass

    d["guard_profit"]   = profit_pass
    d["guard_issuance"] = issuance_pass
    d["guard_assets"]   = asset_pass
    d["guard_accruals"] = accruals_pass
    d["guard_leverage"] = lev_pass
    d["guard_all"]      = mask

    return d[mask].copy(), d


