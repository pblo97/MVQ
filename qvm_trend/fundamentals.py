# qvm_trend/fundamentals.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import concurrent.futures as cf
import time
import math

import numpy as np
import pandas as pd

# HTTP común (robusto, con rate limit/backoff) provisto en data_io.py
from .data_io import _http_get

# Cache opcional (si no está, definimos no-ops)
try:
    from .cache_io import save_df, load_df
except Exception:
    def save_df(df: pd.DataFrame, key: str):
        return
    def load_df(key: str) -> Optional[pd.DataFrame]:
        return None

# ======================================================================
# Helpers genéricos / numéricos robustos
# ======================================================================

CAP_Z = 3.0  # límite de seguridad para z-scores

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

def _to_float(s: pd.Series | np.ndarray | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s = pd.to_numeric(s, errors="coerce")
    return s.astype(float)

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = _to_float(s)
    if s.notna().sum() < 3 or p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (s - mu) / sd

def _safe_div(a, b) -> pd.Series:
    a = _to_float(a)
    b = _to_float(b)
    out = a.div(b)
    return out.replace([np.inf, -np.inf], np.nan)

def _rank_pct(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    return s.rank(pct=True, method="average")

def _winsor(s: pd.Series, p: float = 0.01) -> pd.Series:
    # Alias interno para módulos que usaban _winsor en vez de _winsorize
    return _winsorize(s, p)

# ======================================================================
# Intangibles / I+D
# ======================================================================

def capitalize_rd(df: pd.DataFrame, rd_col="rd_expense_ttm", amort_years: int = 3) -> pd.DataFrame:
    """
    Capitaliza I+D (80%) y genera:
      - rd_asset (activo intangible por I+D)
      - op_income_xrd (EBIT operativo ajustado + amort. I+D)
      - assets_xrd (activos + rd_asset)
    Requiere columnas:
      rd_expense_ttm, operating_income_ttm, total_assets_ttm
    """
    out = df.copy()
    needed = {rd_col, "operating_income_ttm", "total_assets_ttm"}
    if not needed.issubset(out.columns):
        return out.assign(
            rd_asset=np.nan,
            op_income_xrd=np.nan,
            assets_xrd=out.get("total_assets_ttm", np.nan),
        )

    for col in [rd_col, "operating_income_ttm", "total_assets_ttm"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    rd = out[rd_col].fillna(0.0)
    cap_ratio = 0.80
    rd_asset = cap_ratio * rd * amort_years
    amort = rd_asset / amort_years

    out["rd_asset"] = rd_asset
    out["op_income_xrd"] = out["operating_income_ttm"].fillna(0) + amort
    out["assets_xrd"] = out["total_assets_ttm"].fillna(0) + rd_asset
    return out

# ======================================================================
# Value y Quality (growth/intangible-aware)
# ======================================================================

def value_growth_aware(df: pd.DataFrame) -> pd.Series:
    """
    Value “growth-aware”:
      40% EV/EBITDA NTM (invertido)
      30% EV/Gross Profit TTM (invertido)
      30% EV/Sales NTM penalizado por Capex/Sales (invertido)
    Overrides:
      +boost si FCF_yield_5y (ajustada por SBC) está en top quintil sectorial
    Requiere: ev, ebitda_ntm, gross_profit_ttm, sales_ntm, capex_ttm, sbc_ttm
             y opcionalmente fcf_5y_median (si no, se aproxima con fcf_ttm)
    """
    out = df.copy()

    # Forzar numérico en columnas usadas
    for col in ["ev","ebitda_ntm","gross_profit_ttm","sales_ntm",
                "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    ev = out.get("ev")
    gp = out.get("gross_profit_ttm")
    ebitda_ntm = out.get("ebitda_ntm")
    sales_ntm = out.get("sales_ntm")
    capex = out.get("capex_ttm", pd.Series(index=out.index, data=np.nan))
    sbc = out.get("sbc_ttm", pd.Series(index=out.index, data=0.0)).fillna(0.0)
    fcf_ttm = out.get("fcf_ttm", pd.Series(index=out.index, data=np.nan))
    fcf_5y_median = out.get("fcf_5y_median", fcf_ttm)

    # Flags de calidad mínima (evita disparos por divisiones con ~0)
    pre_rev   = (pd.to_numeric(sales_ntm, errors="coerce") <= 0) | (pd.to_numeric(gp, errors="coerce") <= 0)
    bad_ebitda= (pd.to_numeric(ebitda_ntm, errors="coerce") <= 0)

    ev_over_ebitda = _safe_div(ev, ebitda_ntm)
    ev_over_gp     = _safe_div(ev, gp)
    ev_over_sales  = _safe_div(ev, sales_ntm)

    capex_sales = _safe_div(capex, sales_ntm).fillna(0.0).clip(lower=0.0, upper=1.0)  # tope razonable
    ev_over_sales_pen = ev_over_sales * (1 + capex_sales)

    # Invertidos + winsor + CAP de z (evita outliers absurdos)
    def _inv_w(s):
        inv = 1.0 / s.replace(0, np.nan)
        return _winsorize(inv, 0.01).fillna(0.0)

    v1 = _inv_w(ev_over_ebitda)
    v2 = _inv_w(ev_over_gp)
    v3 = _inv_w(ev_over_sales_pen)

    raw = 0.40 * _zscore(v1) + 0.30 * _zscore(v2) + 0.30 * _zscore(v3)
    raw = raw.clip(-CAP_Z, CAP_Z)

    # Penalizaciones explícitas
    penalty = pd.Series(0.0, index=raw.index)
    penalty = penalty.mask(pre_rev,   -1.5)  # sin ventas o sin GP ⇒ fuerte castigo
    penalty = penalty.mask(bad_ebitda, -0.8) # EBITDA ≤ 0 ⇒ castigo moderado

    # Boost por FCF 5y ajustado por SBC (cap suave)
    fcf_yield5 = _safe_div((fcf_5y_median - sbc), ev)
    f5_pct = _rank_pct(fcf_yield5)
    boost = (f5_pct >= 0.80).astype(float) * 0.25

    return (raw + penalty + boost).fillna(-1.0)

def quality_intangible_aware(df: pd.DataFrame) -> pd.Series:
    """
    Quality ajustado por intangibles:
      - GP/Assets_xRD
      - ROIC_xRD (NOPAT_xRD / InvestedCapital_xRD)
      - Estabilidad de márgenes (inv. de la desviación 5y)
      - Accruals (NOA) bajos
      - NetCash/EBITDA
    Requiere: gross_profit_ttm, operating_income_ttm, total_assets_ttm,
              (opcional) rd_expense_ttm para capitalización,
              ebitda_ttm/ntm, net_debt_ttm, noa_ttm, invested_capital_ttm,
              current_liabilities_ttm, tax_rate
    """
    out = capitalize_rd(df).copy()

    # Coerción numérica segura
    for col in ["gross_profit_ttm","assets_xrd","total_assets_ttm","ebitda_ttm",
                "ebitda_ntm","net_debt_ttm","noa_ttm","invested_capital_ttm",
                "current_liabilities_ttm","operating_income_ttm","op_income_xrd",
                "tax_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    gp = out.get("gross_profit_ttm")
    assets_xrd = out.get("assets_xrd", out.get("total_assets_ttm"))
    ebitda = out.get("ebitda_ttm", out.get("ebitda_ntm"))
    net_debt = out.get("net_debt_ttm")
    noa = out.get("noa_ttm")
    ic = out.get("invested_capital_ttm", out.get("total_assets_ttm", 0) - out.get("current_liabilities_ttm", 0))

    tax_rate = out.get("tax_rate", pd.Series(index=out.index, data=0.20)).fillna(0.20)
    op_xrd = out.get("op_income_xrd", out.get("operating_income_ttm", 0))
    nopat_xrd = _to_float(op_xrd) * (1 - _to_float(tax_rate))

    gp_assets = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd = _winsorize(_safe_div(nopat_xrd, ic), 0.01)

    # Estabilidad de márgenes: si hay historial de margen operativo por fila (lista/array)
    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(index=out.index, data=np.nan)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(noa.fillna(noa.median()) if noa is not None else pd.Series(index=out.index, data=0), 0.01)
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(net_debt.fillna(0), _to_float(ebitda).abs() + 1e-9), 0.01)

    score = (
        0.35 * _zscore(gp_assets) +
        0.35 * _zscore(roic_xrd) +
        0.10 * stab +
        0.10 * _zscore(netcash_ebitda) +
        0.10 * accruals_score
    ).clip(-CAP_Z, CAP_Z).fillna(-1.0)

    return score

# ======================================================================
# Neutralización por sector/capitalización y QVM
# ======================================================================

def neutralize_by_sector_cap(df: pd.DataFrame, score_col: str, sector_col: str = "sector",
                             mcap_col: str = "market_cap",
                             buckets=(("Mega", 150e9, np.inf),
                                      ("Large", 10e9, 150e9),
                                      ("Mid", 2e9, 10e9),
                                      ("Small", 0, 2e9))) -> pd.Series:
    """
    Devuelve score neutralizado por sector y bucket de market cap:
      final = 0.5*z_sector + 0.5*z_capbucket
    - Se ordenan los buckets por su borde inferior para garantizar bins crecientes.
    """
    out = df.copy()
    out[mcap_col] = pd.to_numeric(out.get(mcap_col, np.nan), errors="coerce")

    # Ordenar buckets por límite inferior y construir bins crecientes
    b_sorted = sorted(list(buckets), key=lambda b: float(b[1]))
    # edges: [low0, low1, low2, ..., high_last]
    edges = [b_sorted[0][1]] + [b[1] for b in b_sorted[1:]] + [b_sorted[-1][2]]
    # Asegurar estrictamente creciente
    edges = [float(x) for x in edges]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)

    labels = [b[0] for b in b_sorted]
    try:
        out["_cap_bucket"] = pd.cut(out[mcap_col], bins=edges, labels=labels, include_lowest=True, right=False)
    except Exception:
        out["_cap_bucket"] = pd.Series(np.nan, index=out.index, dtype="object")

    def z_by(group):
        return _zscore(group[score_col])

    # z por sector
    z_sector = out.groupby(sector_col, group_keys=False, dropna=False).apply(z_by).rename("z_sector")

    # z por bucket (si todo NaN, devuelve NaN y el promedio final lo trata)
    z_cap = out.groupby("_cap_bucket", group_keys=False, dropna=False).apply(z_by).rename("z_cap")

    final = 0.5 * z_sector + 0.5 * z_cap
    return final

def compute_qvm_scores(df: pd.DataFrame,
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap") -> pd.DataFrame:
    """
    Calcula Value y Quality “growth-aware”,
    neutraliza por sector+cap y devuelve:
      value_adj, quality_adj, value_adj_neut, quality_adj_neut, qvm_score
    """
    df = df.copy()
    df["value_adj"] = value_growth_aware(df)
    df["quality_adj"] = quality_intangible_aware(df)
    df["value_adj_neut"] = neutralize_by_sector_cap(df, "value_adj", sector_col, mcap_col)
    df["quality_adj_neut"] = neutralize_by_sector_cap(df, "quality_adj", sector_col, mcap_col)
    m = _zscore(_to_float(df.get(momentum_col, np.nan)))
    df["qvm_score"] = (
        w_quality * _zscore(df["quality_adj_neut"]) +
        w_value   * _zscore(df["value_adj_neut"])   +
        w_momentum* m
    )
    return df

def apply_megacap_rules(df: pd.DataFrame,
                        momentum_col="momentum_score",
                        quality_col="quality_adj_neut",
                        value_col="value_adj_neut") -> pd.DataFrame:
    """
    Reglas:
     - Permite peso aunque Value quede 35–45p si Momentum>=70p y Quality>=55p (sectorial).
     - Si Quality<45p o profit warning, marca 'quality_too_low'.
    """
    out = df.copy()
    out["q_pct_sector"] = out.groupby("sector")[quality_col].transform(lambda s: s.rank(pct=True))
    out["v_pct_sector"] = out.groupby("sector")[value_col].transform(lambda s: s.rank(pct=True))
    out["m_pct_global"] = out[momentum_col].rank(pct=True)
    out["mega_exception_ok"] = (
        (out["m_pct_global"] >= 0.70) &
        (out["q_pct_sector"] >= 0.55) &
        (out["v_pct_sector"] >= 0.35)
    )
    out["quality_too_low"] = out["q_pct_sector"] < 0.45
    return out

# ======================================================================
# FUNDAMENTALES (mínimo de batalla) → para VFQ
# ======================================================================

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

    # Flags de fuente
    out["__src_ev"]   = "ttm" if evttm is not None else None
    out["__src_gp"]   = "ttm" if gpttm is not None else None
    out["__src_ta"]   = "ttm" if tattm is not None else None
    out["__src_fcf"]  = "ttm" if fcf_t is not None else None
    out["__src_cfo"]  = "ttm" if cfo_t is not None else None
    out["__src_ebit"] = "ttm" if ebit_t is not None else None
    out["__src_roic"] = "ttm" if roic_t is not None else None
    out["__src_roa"]  = "ttm" if roa_t is not None else None
    out["__src_nmar"] = "ttm" if nmar_t is not None else None

    # Fallback annual si falta algo crítico
    need_annual = any(
        x is None for x in [out["evToEbitda"], out["grossProfitTTM"], out["totalAssetsTTM"],
                            out["fcf_ttm"], out["cfo_ttm"], out["ebit_ttm"], out["roic"], out["roa"], out["netMargin"]]
    )

    if need_annual:
        # key-metrics annual
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics/{s}", params={"period":"annual","limit":4})
            km = j[0] if isinstance(j, list) and j else {}
        except Exception:
            km = {}
        if out["evToEbitda"]     is None: out["evToEbitda"]     = _num(km.get("enterpriseValueOverEBITDA"))
        if out["grossProfitTTM"] is None: out["grossProfitTTM"] = _num(km.get("grossProfit"))
        if out["totalAssetsTTM"] is None: out["totalAssetsTTM"] = _num(km.get("totalAssets"))
        if out["marketCap"]      is None: out["marketCap"]      = _num(km.get("marketCap")) or (market_cap_hint if market_cap_hint else None)

        # ratios annual
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/ratios/{s}", params={"period":"annual","limit":4})
            rr = j[0] if isinstance(j, list) and j else {}
        except Exception:
            rr = {}
        if out["roic"]      is None: out["roic"]      = _num(rr.get("returnOnCapitalEmployed") or rr.get("returnOnInvestedCapital"))
        if out["roa"]       is None: out["roa"]       = _num(rr.get("returnOnAssets"))
        if out["netMargin"] is None: out["netMargin"] = _num(rr.get("netProfitMargin"))

        # cash-flow annual
        if out["cfo_ttm"] is None or out["fcf_ttm"] is None:
            try:
                cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{s}", params={"period":"annual","limit":1})
                cf0 = cf[0] if isinstance(cf, list) and cf else {}
            except Exception:
                cf0 = {}
            if out["cfo_ttm"] is None: out["cfo_ttm"] = _num(cf0.get("netCashProvidedByOperatingActivities"))
            if out["fcf_ttm"] is None: out["fcf_ttm"] = _num(cf0.get("freeCashFlow"))

        # income annual
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
    key = f"fund_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None and not dfc.empty:
            return dfc

    rows = []
    mc_map = market_caps or {}
    throttle = max(0.0, 60.0 / max(1, max_symbols_per_minute))
    for i, s in enumerate(symbols):
        if i > 0 and throttle > 0:
            time.sleep(throttle)
        try:
            rec = _fetch_min_battle_fmp(s, market_cap_hint=mc_map.get(s))
            rows.append(rec)
        except Exception as e:
            rows.append({"symbol": s, "__err_fund": str(e)[:180]})

    df = pd.DataFrame(rows).drop_duplicates("symbol")

    # Si literalmente no hay cobertura, intenta un segundo pase con muestra
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
        df = df.set_index("symbol").combine_first(df2.set_index("symbol")).reset_index()

    if key and _coverage_count(df) > 0:
        try: save_df(df, key)
        except Exception: pass

    return df

# Flag para progreso Streamlit (opcional)
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ======================================================================
# GUARDRAILS: descarga (paralela) + aplicación
# ======================================================================

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

    # KEY-METRICS TTM
    try:
        kttm = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}")
        kt0 = _first_obj(kttm)
        out["shares_out_ttm"] = _safe_float(kt0.get("sharesOutstanding"))
        out["net_debt_ttm"]   = _safe_float(kt0.get("netDebtTTM"))
        out["ebitda_ttm"]     = _safe_float(kt0.get("ebitdaTTM"))
    except Exception:
        pass

    # CFO/FCF TTM
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

    # EBIT TTM
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

    # Series anuales para growth/accruals/issuance
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

    # Net issuance
    shares_km = _yr_series(km, "sharesOutstanding")
    shares_bs = _yr_series(bal, "commonStockSharesOutstanding")
    seq = shares_km if len(shares_km) >= 2 else shares_bs
    if len(seq) >= 2:
        _, s0 = seq[-2]; _, s1 = seq[-1]
        out["net_issuance"] = (s1 - s0) / s0 if (s0 not in (None, 0)) else None

    # NetDebt/EBITDA
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

def _retry_download_guardrails(symbol: str, retries: int = 2, base_sleep: float = 0.6) -> Dict[str, Any]:
    """
    Llama download_guardrails(symbol) con reintentos exponenciales.
    Devuelve siempre un dict con al menos {"symbol": symbol, ...} o {"symbol": symbol, "__err_guard": "..."}.
    """
    for attempt in range(retries + 1):
        try:
            row = download_guardrails(symbol)
            if isinstance(row, dict):
                row.setdefault("symbol", symbol)
                return row
            elif isinstance(row, pd.Series):
                d = row.to_dict(); d.setdefault("symbol", symbol)
                return d
            else:
                return {"symbol": symbol, "__err_guard": f"Unexpected return type: {type(row).__name__}"}
        except Exception as e:
            if attempt < retries:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            return {"symbol": symbol, "__err_guard": str(e)[:180]}

def _norm_symbols(symbols: List[str]) -> List[str]:
    """Normaliza y deduplica preservando orden."""
    seen = set()
    out = []
    for s in symbols:
        if s is None:
            continue
        t = str(s).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def download_guardrails_batch(
    symbols: List[str],
    cache_key: str | None = None,
    force: bool = False,
    *,
    chunk_size: int = 150,
    max_workers: int = 8,
    pause_between_chunks: float = 0.6,
    retries: int = 2
) -> pd.DataFrame:
    """
    Descarga guardrails para muchos símbolos con:
      - chunks para controlar rate limits,
      - concurrencia limitada por chunk,
      - reintentos con backoff por símbolo,
      - caché opcional vía load_df/save_df.
    """
    key = f"guard_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc

    syms = _norm_symbols(symbols)
    if not syms:
        return pd.DataFrame(columns=["symbol"])

    # Progreso en Streamlit si está disponible
    prog = st.progress(0.0) if _HAS_ST else None
    status = st.empty() if _HAS_ST else None

    rows: list[Dict[str, Any]] = []
    total = len(syms)
    processed = 0

    for i in range(0, total, chunk_size):
        chunk = syms[i : i + chunk_size]

        if status:
            status.write(f"Guardrails: procesando {i+1}-{min(i+len(chunk), total)} / {total} símbolos...")

        # Ejecuta en paralelo con límite de workers
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_retry_download_guardrails, s, retries): s for s in chunk}
            for fut in cf.as_completed(futs):
                rows.append(fut.result())
                processed += 1
                if prog:
                    prog.progress(min(processed / total, 1.0))

        # Pausa corta entre chunks para evitar rate limits
        if i + chunk_size < total and pause_between_chunks > 0:
            time.sleep(pause_between_chunks)

    if status:
        status.write("Guardrails: consolidando resultados...")

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns:
        df["symbol"] = syms[:len(df)]
    df = df.drop_duplicates(subset=["symbol"], keep="first")
    order = {s: idx for idx, s in enumerate(syms)}
    df["_ord"] = df["symbol"].map(order)
    df = df.sort_values("_ord").drop(columns=["_ord"])

    if key:
        save_df(df, key)

    if status:
        status.write("Guardrails: listo ✅")
        prog.progress(1.0)

    return df

# ======================================================================
# VFQ clásico (merge universo + fundamentales) y dinámico
# ======================================================================

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
    # --- merge base
    dfu = df_universe.copy() if isinstance(df_universe, pd.DataFrame) else pd.DataFrame()
    dff = df_fund.copy()     if isinstance(df_fund, pd.DataFrame)     else pd.DataFrame()

    if dfu.empty or "symbol" not in dfu.columns:
        return pd.DataFrame(columns=["symbol","VFQ","coverage_count"])

    if "symbol" not in dff.columns:
        dff = pd.DataFrame(columns=["symbol"])

    df = dfu.merge(dff, on="symbol", how="left").copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # --- columnas de identificación
    for col in ["sector","industry"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df["sector"] = df["sector"].astype(str).replace({None: "Unknown"}).fillna("Unknown")

    # --- market cap unificado (robusto a _x/_y y variantes)
    def to_num(colname: str) -> pd.Series:
        return pd.to_numeric(df[colname], errors="coerce") if colname in df.columns else pd.Series(np.nan, index=df.index)

    mcap = pd.Series(np.nan, index=df.index)
    mcap_candidates = (
        ["marketCap", "marketCap_profile", "marketCap_ev"] +
        [c for c in df.columns if c.lower().startswith("marketcap")]
    )
    for c in mcap_candidates:
        if c in df.columns:
            mcap = mcap.fillna(to_num(c))

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

    # --- bucket por tamaño
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

    val_cols = [c for c in ["fcf_yield","inv_ev_ebitda"] if c in df.columns]
    q_cols   = [c for c in ["gross_profitability","roic","roa","netMargin"] if c in df.columns]

    # winsor suave
    for c in val_cols + q_cols:
        df[c] = _winsorize(df[c], 0.01)

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

    try:
        sec = df["sector"].astype(str).replace({None: "Unknown"}).fillna("Unknown")
        df["VFQ_pct_sector"] = df.groupby(sec)["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = df["VFQ_pct_sector"].clip(0.0, 1.0).fillna(1.0)

    return df

# ======================================================================
# Guardrails: aplicación de umbrales
# ======================================================================

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

    # flags de diagnóstico
    d["guard_profit"]   = profit_pass
    d["guard_issuance"] = issuance_pass
    d["guard_assets"]   = asset_pass
    d["guard_accruals"] = accruals_pass
    d["guard_leverage"] = lev_pass
    d["guard_all"]      = mask

    return d[mask].copy(), d

# ======================================================================
# VFQ dinámico (si quieres definir columnas ad-hoc)
# ======================================================================

def build_vfq_scores_dynamic(
    df: pd.DataFrame,
    value_metrics: list[str],
    quality_metrics: list[str],
    w_value: float = 0.5,
    w_quality: float = 0.5,
    method_intra: str = "mean",    # "mean" | "median" | "weighted_mean" (=mean)
    winsor_p: float = 0.01,
    size_buckets: int = 3,
    group_mode: str = "sector",    # "sector" | "sector|size"
) -> pd.DataFrame:
    df = df.copy()

    def _numcol(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    def _winsor_local(s: pd.Series, p: float):
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all() or p <= 0:
            return s
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)

    # Derivadas mínimas si faltan
    if "inv_ev_ebitda" in value_metrics and "inv_ev_ebitda" not in df.columns:
        ev = _numcol("evToEbitda")
        df["inv_ev_ebitda"] = (1.0 / ev).replace([np.inf, -np.inf], np.nan)

    if "fcf_yield" in value_metrics and "fcf_yield" not in df.columns:
        df["fcf_yield"] = (_numcol("fcf_ttm") / _numcol("marketCap_unified")).replace([np.inf, -np.inf], np.nan)

    if "gross_profitability" in quality_metrics and "gross_profitability" not in df.columns:
        df["gross_profitability"] = (_numcol("grossProfitTTM") / _numcol("totalAssetsTTM")).replace([np.inf, -np.inf], np.nan)

    V = [c for c in value_metrics if c in df.columns]
    Q = [c for c in quality_metrics if c in df.columns]

    for c in set(V + Q):
        df[c] = _winsor_local(df[c], winsor_p)

    use_cols = V + Q
    df["coverage_count"] = df[use_cols].notna().sum(axis=1) if use_cols else 0

    # size buckets
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

    def _rank_group(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.groupby(grp_key).rank(method="average", ascending=False, na_option="bottom")

    def _block_score(cols):
        if not cols:
            return pd.Series(np.nan, index=df.index)
        ranks = pd.concat([_rank_group(c) for c in cols], axis=1)
        if method_intra == "median":
            return ranks.median(axis=1)
        return ranks.mean(axis=1)

    df["ValueScore"]   = _block_score(V)
    df["QualityScore"] = _block_score(Q)

    w_sum = (w_value or 0) + (w_quality or 0)
    if w_sum == 0:
        w_value = w_quality = 0.5
        w_sum = 1.0
    df["VFQ"] = (df["ValueScore"] * w_value + df["QualityScore"] * w_quality) / w_sum

    try:
        df["VFQ_pct_sector"] = df.groupby("sector")["VFQ"].rank(pct=True)
    except Exception:
        df["VFQ_pct_sector"] = df["VFQ"].rank(pct=True)
    df["VFQ_pct_sector"] = df["VFQ_pct_sector"].clip(0.0, 1.0).fillna(1.0)

    return df
