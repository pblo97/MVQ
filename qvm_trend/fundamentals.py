# qvm_trend/fundamentals.py  — añadir debajo de lo existente
from .data_io import _http_get
import math
import pandas as pd

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def _yr_series(items, key):
    """Construye serie anual (list of dicts) -> list of (date, value) con coerción numérica."""
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
      - Net Issuance (12-24m)
      - Asset Growth (y/y)
      - Accruals/TA (anual)
      - NetDebt/EBITDA (últ. año)
    Devuelve un dict con métricas y flags básicos (no aplica umbrales aquí).
    """
    sym = symbol.strip().upper()
    out = {"symbol": sym}

    # --- TTM básicos para piso de rentabilidad
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
        # fallback: cash-flow statement último año
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}", params={"limit":1})
            if isinstance(cf, list) and cf:
                out["cfo_ttm"] = _safe_num(cf[0].get("netCashProvidedByOperatingActivities"))
                out["fcf_ttm"] = _safe_num(cf[0].get("freeCashFlow"))
        except Exception:
            pass

    # EBIT TTM (aprox con income ttm si está)
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        if isinstance(inc_ttm, dict) and inc_ttm:
            out["ebit_ttm"] = _safe_num(inc_ttm.get("ebitTTM") or inc_ttm.get("operatingIncomeTTM"))
    except Exception:
        # fallback: último anual
        try:
            inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period":"annual","limit":1})
            if isinstance(inc, list) and inc:
                out["ebit_ttm"] = _safe_num(inc[0].get("ebit") or inc[0].get("operatingIncome"))
        except Exception:
            pass

    # --- Series anuales para growth, accruals, issuance
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

    # Asset growth
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]
        _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0)/a0 if (a0 and a0!=0) else None

    # Accruals/TA: (ΔCA − ΔCash) − (ΔCL − ΔShortDebt) − Depreciation, escalar por activos medios
    # Implementación práctica: total accruals = NI − CFO; escalado por activos medios
    ni = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf, "netCashProvidedByOperatingActivities")
    ta  = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1]; _, ni0 = ni[-2]
        _, cfo1 = cfo[-1]; _, cfo0 = cfo[-2]
        accruals = (ni1 - cfo1)  # último año
        _, ta1 = ta[-1]; _, ta0 = ta[-2]
        avg_assets = ((ta1 or 0.0) + (ta0 or 0.0))/2.0 if (ta1 and ta0) else None
        out["accruals_ta"] = (accruals/avg_assets) if (avg_assets and avg_assets!=0) else None

    # Net issuance (Δ acciones 12–24m): usar key-metrics anual por disponibilidad
    shares = _yr_series(km, "sharesOutstanding")
    if len(shares) >= 2:
        _, s0 = shares[-2]; _, s1 = shares[-1]
        if s0 and s0 != 0:
            out["net_issuance"] = (s1 - s0)/s0
        else:
            out["net_issuance"] = None

    # NetDebt/EBITDA (último anual si hay)
    net_debt = None
    ebitda = None
    if km and isinstance(km, list):
        # toma el último con ambos
        for item in reversed(km):
            nd = _safe_num(item.get("netDebt"))
            eb = _safe_num(item.get("ebitda"))
            if nd is not None and eb is not None and eb != 0:
                net_debt = nd; ebitda = eb; break
    if net_debt is not None and ebitda not in (None, 0):
        out["netdebt_ebitda"] = net_debt / ebitda
    else:
        out["netdebt_ebitda"] = None

    return out


def download_guardrails_batch(symbols: list[str], cache_key: str | None = None, force: bool=False) -> pd.DataFrame:
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
    if key: save_df(df, key)
    return df


def apply_quality_guardrails(df: pd.DataFrame,
                             require_profit_floor=True,
                             profit_floor_min_hits=2,  # de {EBIT>0, CFO>0, FCF>0}
                             max_net_issuance=0.03,
                             max_asset_growth=0.20,
                             max_accruals_ta=0.10,
                             max_netdebt_ebitda=3.0) -> pd.DataFrame:
    """
    Devuelve subset que cumple los guardrails y añade columnas booleanas y 'profit_hits'.
    """
    d = df.copy()

    # profit floor
    ebit_ok = (d.get("ebit_ttm") > 0)
    cfo_ok  = (d.get("cfo_ttm")  > 0)
    fcf_ok  = (d.get("fcf_ttm")  > 0)
    d["profit_hits"] = ebit_ok.astype(int) + cfo_ok.astype(int) + fcf_ok.astype(int)
    profit_pass = (d["profit_hits"] >= int(profit_floor_min_hits)) if require_profit_floor else True

    # otros guardrails
    issuance_pass = (d.get("net_issuance").fillna(0) <= float(max_net_issuance))
    asset_pass    = (d.get("asset_growth").abs() <= float(max_asset_growth))  # puedes usar solo lado + si prefieres
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

# qvm_trend/fundamentals.py  — añadir debajo de lo existente
from .data_io import _http_get
import math

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def _yr_series(items, key):
    """Construye serie anual (list of dicts) -> list of (date, value) con coerción numérica."""
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
      - Net Issuance (12-24m)
      - Asset Growth (y/y)
      - Accruals/TA (anual)
      - NetDebt/EBITDA (últ. año)
    Devuelve un dict con métricas y flags básicos (no aplica umbrales aquí).
    """
    sym = symbol.strip().upper()
    out = {"symbol": sym}

    # --- TTM básicos para piso de rentabilidad
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
        # fallback: cash-flow statement último año
        try:
            cf = _http_get(f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}", params={"limit":1})
            if isinstance(cf, list) and cf:
                out["cfo_ttm"] = _safe_num(cf[0].get("netCashProvidedByOperatingActivities"))
                out["fcf_ttm"] = _safe_num(cf[0].get("freeCashFlow"))
        except Exception:
            pass

    # EBIT TTM (aprox con income ttm si está)
    try:
        inc_ttm = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement-ttm/{sym}")
        if isinstance(inc_ttm, dict) and inc_ttm:
            out["ebit_ttm"] = _safe_num(inc_ttm.get("ebitTTM") or inc_ttm.get("operatingIncomeTTM"))
    except Exception:
        # fallback: último anual
        try:
            inc = _http_get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period":"annual","limit":1})
            if isinstance(inc, list) and inc:
                out["ebit_ttm"] = _safe_num(inc[0].get("ebit") or inc[0].get("operatingIncome"))
        except Exception:
            pass

    # --- Series anuales para growth, accruals, issuance
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

    # Asset growth
    assets = _yr_series(bal, "totalAssets")
    if len(assets) >= 2:
        _, a0 = assets[-2]
        _, a1 = assets[-1]
        out["asset_growth"] = (a1 - a0)/a0 if (a0 and a0!=0) else None

    # Accruals/TA: (ΔCA − ΔCash) − (ΔCL − ΔShortDebt) − Depreciation, escalar por activos medios
    # Implementación práctica: total accruals = NI − CFO; escalado por activos medios
    ni = _yr_series(inc, "netIncome")
    cfo = _yr_series(cf, "netCashProvidedByOperatingActivities")
    ta  = _yr_series(bal, "totalAssets")
    if len(ni) >= 2 and len(cfo) >= 2 and len(ta) >= 2:
        _, ni1 = ni[-1]; _, ni0 = ni[-2]
        _, cfo1 = cfo[-1]; _, cfo0 = cfo[-2]
        accruals = (ni1 - cfo1)  # último año
        _, ta1 = ta[-1]; _, ta0 = ta[-2]
        avg_assets = ((ta1 or 0.0) + (ta0 or 0.0))/2.0 if (ta1 and ta0) else None
        out["accruals_ta"] = (accruals/avg_assets) if (avg_assets and avg_assets!=0) else None

    # Net issuance (Δ acciones 12–24m): usar key-metrics anual por disponibilidad
    shares = _yr_series(km, "sharesOutstanding")
    if len(shares) >= 2:
        _, s0 = shares[-2]; _, s1 = shares[-1]
        if s0 and s0 != 0:
            out["net_issuance"] = (s1 - s0)/s0
        else:
            out["net_issuance"] = None

    # NetDebt/EBITDA (último anual si hay)
    net_debt = None
    ebitda = None
    if km and isinstance(km, list):
        # toma el último con ambos
        for item in reversed(km):
            nd = _safe_num(item.get("netDebt"))
            eb = _safe_num(item.get("ebitda"))
            if nd is not None and eb is not None and eb != 0:
                net_debt = nd; ebitda = eb; break
    if net_debt is not None and ebitda not in (None, 0):
        out["netdebt_ebitda"] = net_debt / ebitda
    else:
        out["netdebt_ebitda"] = None

    return out


def download_guardrails_batch(symbols: list[str], cache_key: str | None = None, force: bool=False) -> pd.DataFrame:
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
    if key: save_df(df, key)
    return df


def apply_quality_guardrails(df: pd.DataFrame,
                             require_profit_floor=True,
                             profit_floor_min_hits=2,  # de {EBIT>0, CFO>0, FCF>0}
                             max_net_issuance=0.03,
                             max_asset_growth=0.20,
                             max_accruals_ta=0.10,
                             max_netdebt_ebitda=3.0) -> pd.DataFrame:
    """
    Devuelve subset que cumple los guardrails y añade columnas booleanas y 'profit_hits'.
    """
    d = df.copy()

    # profit floor
    ebit_ok = (d.get("ebit_ttm") > 0)
    cfo_ok  = (d.get("cfo_ttm")  > 0)
    fcf_ok  = (d.get("fcf_ttm")  > 0)
    d["profit_hits"] = ebit_ok.astype(int) + cfo_ok.astype(int) + fcf_ok.astype(int)
    profit_pass = (d["profit_hits"] >= int(profit_floor_min_hits)) if require_profit_floor else True

    # otros guardrails
    issuance_pass = (d.get("net_issuance").fillna(0) <= float(max_net_issuance))
    asset_pass    = (d.get("asset_growth").abs() <= float(max_asset_growth))  # puedes usar solo lado + si prefieres
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
