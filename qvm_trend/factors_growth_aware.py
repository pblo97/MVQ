from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------- Utils -----------------------------
def _to_float(s: pd.Series | pd.DataFrame | np.ndarray | None) -> pd.Series:
    """
    Fuerza a Serie 1D numérica, coalesciendo entradas 2D si llegan:
    - DataFrame con >1 columna: promedio por fila (skipna)
    - ndarray 2D: toma la 1ª columna
    """
    if s is None:
        return pd.Series(dtype=float)

    # DataFrame (posibles duplicados de nombre → 2D)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            # Coalesce: promedio fila a fila
            s = s.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    # ndarray
    if isinstance(s, np.ndarray):
        if s.ndim > 1:
            s = s[:, 0]
        s = pd.Series(s)

    # Cualquier otra cosa → Serie
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    s = pd.to_numeric(s, errors="coerce")
    return s.astype(float)


def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = _to_float(s)
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    s = _to_float(s)  # ahora tolera 2D
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
def _safe_div(a, b) -> pd.Series:
    a = _to_float(a)
    b = _to_float(b)
    out = a.div(b)
    return out.replace([np.inf, -np.inf], np.nan)

def _rank_pct(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    # si todos NaN → devolver ceros para no romper
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.0)

# ----------------------- Intangibles / I+D -----------------------
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
    req = {rd_col, "operating_income_ttm", "total_assets_ttm"}
    if not req.issubset(out.columns):
        out["rd_asset"] = np.nan
        out["op_income_xrd"] = out.get("operating_income_ttm", np.nan)
        out["assets_xrd"] = out.get("total_assets_ttm", np.nan)
        return out

    rd = _to_float(out[rd_col]).fillna(0.0)
    opi = _to_float(out["operating_income_ttm"]).fillna(0.0)
    ta = _to_float(out["total_assets_ttm"]).fillna(0.0)

    cap_ratio = 0.80
    rd_asset = cap_ratio * rd * amort_years
    amort = rd_asset / amort_years

    out["rd_asset"] = rd_asset
    out["op_income_xrd"] = opi + amort
    out["assets_xrd"] = ta + rd_asset
    return out

# ----------------------------- Value -----------------------------
def value_growth_aware(df: pd.DataFrame) -> pd.Series:
    """
    Value “growth-aware”:
      40% EV/EBITDA NTM (invertido)
      30% EV/Gross Profit TTM (invertido)
      30% EV/Sales NTM penalizado por Capex/Sales (invertido)
    + boost si FCF_yield_5y (ajustada por SBC) está en top 20% global
    Requiere: ev, ebitda_ntm, gross_profit_ttm, sales_ntm, capex_ttm, sbc_ttm
             y opcionalmente fcf_5y_median (si no, usa fcf_ttm)
    """
    out = df.copy()

    # Forzar numérico en columnas usadas
    for col in ["ev","ebitda_ntm","gross_profit_ttm","sales_ntm",
                "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median"]:
        if col in out.columns:
            out[col] = _to_float(out[col])

    ev          = out.get("ev")
    ebitda_ntm  = out.get("ebitda_ntm")
    gp_ttm      = out.get("gross_profit_ttm")
    sales_ntm   = out.get("sales_ntm")
    capex_ttm   = out.get("capex_ttm", pd.Series(index=out.index, data=np.nan))
    sbc_ttm     = out.get("sbc_ttm", pd.Series(index=out.index, data=0.0)).fillna(0.0)
    fcf_ttm     = out.get("fcf_ttm", pd.Series(index=out.index, data=np.nan))
    fcf5_med    = out.get("fcf_5y_median", fcf_ttm)

    ev_over_ebitda = _safe_div(ev, ebitda_ntm)
    ev_over_gp     = _safe_div(ev, gp_ttm)
    ev_over_sales  = _safe_div(ev, sales_ntm)
    capex_sales    = _safe_div(capex_ttm, sales_ntm).fillna(0.0)
    ev_over_sales_pen = ev_over_sales * (1 + capex_sales)

    v1 = _winsorize(1.0 / ev_over_ebitda.replace(0, np.nan), 0.01).fillna(0.0)
    v2 = _winsorize(1.0 / ev_over_gp.replace(0, np.nan), 0.01).fillna(0.0)
    v3 = _winsorize(1.0 / ev_over_sales_pen.replace(0, np.nan), 0.01).fillna(0.0)

    raw = 0.40 * _zscore(v1) + 0.30 * _zscore(v2) + 0.30 * _zscore(v3)

    # Boost por FCF 5y yield ajustado por SBC
    fcf_yield5 = _safe_div((fcf5_med - sbc_ttm), ev)
    f5_pct = _rank_pct(fcf_yield5)
    boost = (f5_pct >= 0.80).astype(float) * 0.25

    return (raw + boost).fillna(0.0)

# ---------------------------- Quality ----------------------------
def quality_intangible_aware(df: pd.DataFrame) -> pd.Series:
    """
    Quality ajustado por intangibles:
      - GP/Assets_xRD
      - ROIC_xRD (NOPAT_xRD / InvestedCapital_xRD)
      - Estabilidad de márgenes (inv. de la desviación 5y)
      - Accruals (NOA) bajos
      - NetCash/EBITDA
    """
    # Primero capitalizamos I+D
    out = capitalize_rd(df).copy()

    # Coerción a numérico DESPUÉS de capitalizar
    for col in [
        "gross_profit_ttm","assets_xrd","total_assets_ttm","ebitda_ttm",
        "ebitda_ntm","net_debt_ttm","noa_ttm","invested_capital_ttm",
        "current_liabilities_ttm","operating_income_ttm","op_income_xrd","tax_rate"
    ]:
        if col in out.columns:
            out[col] = _to_float(out[col])

    gp         = out.get("gross_profit_ttm")
    assets_xrd = out.get("assets_xrd", out.get("total_assets_ttm"))
    ebitda     = out.get("ebitda_ttm", out.get("ebitda_ntm"))
    net_debt   = out.get("net_debt_ttm")
    noa        = out.get("noa_ttm")
    ic         = out.get("invested_capital_ttm",
                  _to_float(out.get("total_assets_ttm", 0)) - _to_float(out.get("current_liabilities_ttm", 0)))

    tax_rate   = _to_float(out.get("tax_rate", pd.Series(index=out.index, data=0.20))).fillna(0.20)
    opi_xrd    = _to_float(out.get("op_income_xrd", out.get("operating_income_ttm", 0))).fillna(0.0)
    nopat_xrd  = opi_xrd * (1 - tax_rate)

    gp_assets = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd  = _winsorize(_safe_div(nopat_xrd, ic), 0.01)

    # Estabilidad de márgenes
    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(index=out.index, data=np.nan)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(_to_float(noa).fillna(_to_float(noa).median()) if noa is not None else pd.Series(index=out.index, data=0.0), 0.01)
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(_to_float(net_debt).fillna(0.0), _to_float(ebitda).abs() + 1e-9), 0.01)

    return (
        0.35 * _zscore(gp_assets) +
        0.35 * _zscore(roic_xrd)  +
        0.10 * stab               +
        0.10 * _zscore(netcash_ebitda) +
        0.10 * accruals_score
    ).fillna(0.0)

# ------------------- Sector & Cap Neutralization -----------------
def neutralize_by_sector_cap(df: pd.DataFrame, score_col: str, sector_col: str = "sector",
                             mcap_col: str = "market_cap",
                             buckets=(("Mega", 150e9, np.inf),
                                      ("Large", 10e9, 150e9),
                                      ("Mid",   2e9, 10e9),
                                      ("Small", 0,   2e9))) -> pd.Series:
    out = df.copy()

    # sector/cap fallbacks
    if sector_col not in out.columns:
        out[sector_col] = "Unknown"
    if mcap_col not in out.columns:
        out[mcap_col] = np.nan

    out[mcap_col] = _to_float(out[mcap_col])
    # Construimos cortes robustos
    edges = [b[1] for b in buckets] + [buckets[-1][2]]
    labels = [b[0] for b in buckets]
    out["_cap_bucket"] = pd.cut(out[mcap_col], bins=edges, labels=labels, include_lowest=True, right=False)

    def z_by(group):
        s = group[score_col]
        return _zscore(s)

    z_sector = out.groupby(sector_col, group_keys=False).apply(z_by).rename("z_sector")
    z_cap    = out.groupby("_cap_bucket", group_keys=False).apply(z_by).rename("z_cap")

    out["_z_sector"] = z_sector
    out["_z_cap"] = z_cap
    return (0.5 * out["_z_sector"] + 0.5 * out["_z_cap"]).fillna(0.0)

# ----------------------------- QVM -------------------------------
def compute_qvm_scores(df: pd.DataFrame, 
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap") -> pd.DataFrame:
    df = df.copy()

    # 🔧 evita que df["momentum_score"] devuelva un DataFrame por duplicados
    if hasattr(df, "columns"):
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    df["value_adj"]   = value_growth_aware(df)
    df["quality_adj"] = quality_intangible_aware(df)
    df["value_adj_neut"]   = neutralize_by_sector_cap(df, "value_adj",  sector_col, mcap_col)
    df["quality_adj_neut"] = neutralize_by_sector_cap(df, "quality_adj", sector_col, mcap_col)

    if momentum_col not in df.columns:
        df[momentum_col] = 0.0

    # ✅ ahora _to_float colapsa 2D de forma segura
    m = _zscore(_to_float(df[momentum_col]))

    df["qvm_score"] = (
        w_quality * _zscore(df["quality_adj_neut"]) +
        w_value   * _zscore(df["value_adj_neut"])   +
        w_momentum* m
    ).fillna(0.0)
    return df

# ----------------------- Guardrails/overrides --------------------
def apply_megacap_rules(df: pd.DataFrame,
                        momentum_col="momentum_score",
                        quality_col="quality_adj_neut",
                        value_col="value_adj_neut") -> pd.DataFrame:
    out = df.copy()
    for col in (momentum_col, quality_col, value_col):
        if col not in out.columns:
            out[col] = 0.0

    out["q_pct_sector"] = out.groupby("sector")[quality_col].transform(lambda s: s.rank(pct=True)).fillna(0.0)
    out["v_pct_sector"] = out.groupby("sector")[value_col].transform(lambda s: s.rank(pct=True)).fillna(0.0)
    out["m_pct_global"] = out[momentum_col].rank(pct=True).fillna(0.0)

    out["mega_exception_ok"] = (
        (out["m_pct_global"] >= 0.70) &
        (out["q_pct_sector"] >= 0.55) &
        (out["v_pct_sector"] >= 0.35)
    )
    out["quality_too_low"] = out["q_pct_sector"] < 0.45
    return out
