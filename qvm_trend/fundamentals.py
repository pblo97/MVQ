# qvm_trend/fundamentals.py
import numpy as np
import pandas as pd
from typing import List, Dict
from .data_io import _http_get
from .cache_io import save_df, load_df

# --- descarga "set mínimo de batalla" por símbolo ----------------------------
def fetch_min_battle_fmp(symbol: str, market_cap_hint: float | None = None) -> dict:
    sym = symbol.strip().upper()
    out = {"symbol": sym}

    # ratios ttm: ROA, ROIC, NetMargin
    try:
        r = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}")
        if isinstance(r, list) and r:
            x = r[0]
            out["roa"] = x.get("returnOnAssetsTTM")
            out["roic"] = x.get("returnOnCapitalEmployedTTM") or x.get("returnOnInvestedCapitalTTM")
            out["netMargin"] = x.get("netProfitMarginTTM")
    except Exception:
        pass

    # key-metrics ttm: EV/EBITDA, FCF, GrossProfit, TotalAssets, Shares
    try:
        k = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        if isinstance(k, list) and k:
            y = k[0]
            out["evToEbitda"] = y.get("enterpriseValueOverEBITDATTM")
            out["fcf_ttm"] = y.get("freeCashFlowTTM")
            out["grossProfitTTM"] = y.get("grossProfitTTM")
            out["totalAssetsTTM"] = y.get("totalAssetsTTM")
            out["sharesOutstanding"] = y.get("sharesOutstanding")
            if not market_cap_hint and y.get("marketCap"):
                out["marketCap"] = y.get("marketCap")
    except Exception:
        pass

    # market cap fallback
    if "marketCap" not in out or out["marketCap"] in (None, 0):
        try:
            mc = _http_get(f"https://financialmodelingprep.com/api/v3/market-capitalization/{sym}", params={"limit":1})
            if isinstance(mc, list) and mc:
                out["marketCap"] = mc[0].get("marketCap")
        except Exception:
            out["marketCap"] = market_cap_hint

    # FCF Yield
    fcf = out.get("fcf_ttm"); mcap = out.get("marketCap")
    out["fcf_yield"] = (fcf / mcap) if (fcf and mcap) else None

    # Gross Profitability
    gp = out.get("grossProfitTTM"); ta = out.get("totalAssetsTTM")
    out["gross_profitability"] = (gp / ta) if (gp and ta) else None

    # 1/EV/EBITDA
    ev = out.get("evToEbitda")
    out["inv_ev_ebitda"] = (1.0/ev) if ev not in (None, 0) else None

    # (opcionales) Shareholder Yield / Asset Growth / Accruals se pueden agregar luego

    # coverage
    fields = ["fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"]
    out["coverage_count"] = int(sum(out.get(f) is not None for f in fields))
    return out

# --- batch + cache en parquet ------------------------------------------------
def download_fundamentals(symbols: List[str], market_caps: Dict[str, float] | None = None,
                          cache_key: str | None = None, force: bool=False) -> pd.DataFrame:
    """
    Descarga set mínimo para todos los símbolos y cachea.
    """
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    rows = []
    for s in symbols:
        mc_hint = market_caps.get(s) if market_caps else None
        rows.append(fetch_min_battle_fmp(s, market_cap_hint=mc_hint))
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    if key: save_df(df, key)
    return df

# --- VFQ scores (winsor + ranks por sector y tamaño) ------------------------
def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

# qvm_trend/fundamentals.py  (reemplazo de build_vfq_scores + helper)

def coalesce_first(df: pd.DataFrame, candidates: list[str], new_col: str, to_numeric: bool=False):
    """
    Toma las columnas en 'candidates' (si existen), hace forward-first-non-null por fila
    y la guarda en 'new_col'. Si to_numeric=True, coerciona a numérico.
    """
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        df[new_col] = pd.Series([None]*len(df), index=df.index)
        return df
    buf = pd.concat([df[c] for c in cols], axis=1)
    out = buf.bfill(axis=1).iloc[:,0]
    if to_numeric:
        out = pd.to_numeric(out, errors="coerce")
    df[new_col] = out
    return df


def build_vfq_scores(df_universe: pd.DataFrame, df_fund: pd.DataFrame, size_buckets: int = 3) -> pd.DataFrame:
    """
    Une universo (sector, marketCap) + fundamentales; calcula ValueScore, QualityScore y VFQ.
    Robusto a columnas duplicadas: sector_x/sector_y, marketCap_x/marketCap_y.
    """
    # Mezcla universo + fundas
    dfu = df_universe.copy()
    dff = df_fund.copy()
    df = dfu.merge(dff, on="symbol", how="left")

    # Sector unificado (sector, sector_x, sector_y)
    df = coalesce_first(df, ["sector", "sector_x", "sector_y"], "sector_unified", to_numeric=False)
    # MarketCap unificado (marketCap, marketCap_x, marketCap_y, marketCap_profile)
    df = coalesce_first(df, ["marketCap", "marketCap_x", "marketCap_y", "marketCap_profile"], 
                        "marketCap_unified", to_numeric=True)

    # Buckets por tamaño (tolerante si muchos NaN)
    r = df["marketCap_unified"].rank(method="first")
    try:
        df["size_bucket"] = pd.qcut(r, size_buckets, labels=False, duplicates="drop")
    except Exception:
        df["size_bucket"] = 1

    # Winsor
    for c in ["fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"]:
        if c in df.columns:
            df[c] = winsorize(pd.to_numeric(df[c], errors="coerce"), 0.01)

    # Grupo: sector × tamaño (usa el sector_unified)
    grp = df["sector_unified"].astype(str) + "|" + df["size_bucket"].astype(str)

    def _rank(s, asc=False):
        return s.groupby(grp).rank(method="average", ascending=asc, na_option="bottom")

    # Value: FCF yield, 1/EV/EBITDA
    v_parts = []
    if "fcf_yield" in df.columns:      v_parts.append(_rank(df["fcf_yield"], asc=False))
    if "inv_ev_ebitda" in df.columns:  v_parts.append(_rank(df["inv_ev_ebitda"], asc=False))
    df["ValueScore"] = pd.concat(v_parts, axis=1).mean(axis=1) if v_parts else np.nan

    # Quality: GrossProfitability, ROIC, ROA, NetMargin
    q_parts = []
    for c in ["gross_profitability","roic","roa","netMargin"]:
        if c in df.columns:
            q_parts.append(_rank(df[c], asc=False))
    df["QualityScore"] = pd.concat(q_parts, axis=1).mean(axis=1) if q_parts else np.nan

    df["VFQ"] = pd.concat([df["ValueScore"], df["QualityScore"]], axis=1).mean(axis=1)
    # percentil intra-sector (con el sector_unified)
    df["VFQ_pct_sector"] = df.groupby("sector_unified")["VFQ"].rank(pct=True)

    # Selección de columnas de salida amigable
    keep = ["symbol",
            "sector_unified","marketCap_unified",
            "coverage_count",
            "fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin",
            "ValueScore","QualityScore","VFQ","VFQ_pct_sector"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]

