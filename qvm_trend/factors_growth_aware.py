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

def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return _to_float(df[name])
    return pd.Series(index=df.index, data=default, dtype=float)

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
    out = df.copy()
    ev         = _col(out, "ev")
    if ev.isna().all():  # fallback: usa market cap si no hay EV
        ev = _col(out, "market_cap")

    ebitda_ntm = _col(out, "ebitda_ntm")
    gp_ttm     = _col(out, "gross_profit_ttm")
    sales_ntm  = _col(out, "sales_ntm")

    capex_ttm  = _col(out, "capex_ttm", 0.0).fillna(0.0)
    sbc_ttm    = _col(out, "sbc_ttm",   0.0).fillna(0.0)
    fcf_ttm    = _col(out, "fcf_ttm")
    fcf5_med   = _col(out, "fcf_5y_median")
    if fcf5_med.isna().all():
        fcf5_med = fcf_ttm

    ev_over_ebitda = _safe_div(ev, ebitda_ntm)
    ev_over_gp     = _safe_div(ev, gp_ttm)
    ev_over_sales  = _safe_div(ev, sales_ntm)
    capex_sales    = _safe_div(capex_ttm, sales_ntm).fillna(0.0)
    ev_over_sales_pen = ev_over_sales * (1 + capex_sales)

    v1 = _winsorize(1.0 / ev_over_ebitda.replace(0, np.nan), 0.01)
    v2 = _winsorize(1.0 / ev_over_gp.replace(0, np.nan),     0.01)
    v3 = _winsorize(1.0 / ev_over_sales_pen.replace(0, np.nan), 0.01)

    raw = 0.40*_zscore(v1) + 0.30*_zscore(v2) + 0.30*_zscore(v3)

    fcf_yield5 = _safe_div((fcf5_med - sbc_ttm), ev)
    boost = (_rank_pct(fcf_yield5) >= 0.80).astype(float) * 0.25

    return (raw + boost).fillna(0.0).reindex(out.index)


# ---------------------------- Quality ----------------------------
def quality_intangible_aware(df: pd.DataFrame) -> pd.Series:
    out = capitalize_rd(df).copy()

    gp         = _col(out, "gross_profit_ttm")
    assets_xrd = _col(out, "assets_xrd")
    if assets_xrd.isna().all():
        assets_xrd = _col(out, "total_assets_ttm")

    ebitda     = _col(out, "ebitda_ttm")
    if ebitda.isna().all():
        ebitda = _col(out, "ebitda_ntm")

    net_debt   = _col(out, "net_debt_ttm").fillna(0.0)
    noa        = _col(out, "noa_ttm")
    ic         = _col(out, "invested_capital_ttm")
    if ic.isna().all():
        ic = _col(out, "total_assets_ttm") - _col(out, "current_liabilities_ttm", 0.0)

    tax_rate   = _col(out, "tax_rate", 0.20).fillna(0.20)
    opi_xrd    = _col(out, "op_income_xrd")
    if opi_xrd.isna().all():
        opi_xrd = _col(out, "operating_income_ttm").fillna(0.0)
    nopat_xrd  = opi_xrd * (1 - tax_rate)

    gp_assets  = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd   = _winsorize(_safe_div(nopat_xrd, ic),  0.01)

    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(np.nan, index=out.index)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(_col(out, "noa_ttm").fillna(_col(out, "noa_ttm").median()), 0.01)
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(net_debt, ebitda.abs() + 1e-9), 0.01)

    return (0.35*_zscore(gp_assets) + 0.35*_zscore(roic_xrd) +
            0.10*stab + 0.10*_zscore(netcash_ebitda) + 0.10*accruals_score).fillna(0.0).reindex(out.index)
# ------------------- Sector & Cap Neutralization -----------------
def neutralize_by_sector_cap(df: pd.DataFrame, 
                             score_col: str, 
                             sector_col: str = "sector",
                             mcap_col: str = "market_cap",
                             buckets=(("Mega", 150e9, np.inf),
                                      ("Large", 10e9, 150e9),
                                      ("Mid",   2e9, 10e9),
                                      ("Small", 0,   2e9))) -> pd.Series:
    """
    Estandariza (z-score) un 'score_col' intra-sector e intra-cap y mezcla 50/50.
    Devuelve una Serie 1D alineada al índice de df.
    """
    out = df.copy()

    # --- Columnas base y coerción ---
    if hasattr(out, "columns"):
        # Evita duplicados 2D tipo 'momentum_score' duplicado
        out = out.loc[:, ~out.columns.duplicated(keep="last")]

    if score_col not in out.columns:
        # Si falta, devuelve ceros (no rompemos el flujo)
        return pd.Series(0.0, index=out.index, name="z_neut")

    if sector_col not in out.columns:
        out[sector_col] = "Unknown"
    out[sector_col] = out[sector_col].astype(str).fillna("Unknown")

    if mcap_col not in out.columns:
        out[mcap_col] = np.nan
    out[mcap_col] = _to_float(out[mcap_col])

    # --- Score numérico 1D ---
    out["_score_"] = _to_float(out[score_col])

    # --- Buckets por market cap (robusto) ---
    try:
        b_sorted = sorted(list(buckets), key=lambda t: float(t[1]))
        lows  = [float(b[1]) for b in b_sorted]
        highs = [float(b[2]) for b in b_sorted]
        labels = [str(b[0]) for b in b_sorted]

        for lo, hi in zip(lows, highs):
            if not (lo < hi):
                raise ValueError("Cada bucket debe cumplir low < high")

        bins = lows + [highs[-1]]
        bins = np.array(bins, dtype=float)
        if not np.all(np.diff(bins) > 0):
            uniq = np.unique(bins)
            if len(uniq) < 2:
                raise ValueError("Bins inválidos")
            bins = uniq

        if len(labels) != (len(bins) - 1):
            labels = [f"Cap_{i+1}" for i in range(len(bins) - 1)]

        out["_cap_bucket"] = pd.cut(out[mcap_col], bins=bins, labels=labels, include_lowest=True, right=False)
        if out["_cap_bucket"].notna().sum() == 0:
            out["_cap_bucket"] = "Cap_All"
    except Exception:
        q = out[mcap_col]
        if q.notna().nunique() < 2:
            out["_cap_bucket"] = "Cap_All"
        else:
            try:
                out["_cap_bucket"] = pd.qcut(q, q=4, duplicates="drop")
            except Exception:
                out["_cap_bucket"] = "Cap_All"

    # --- z intra-sector / intra-cap ---
    def _z_by(group: pd.DataFrame) -> pd.Series:
        return _zscore(group["_score_"])

    # Importante: align al índice original SIEMPRE (evita problemas posteriores)
    z_sector = out.groupby(sector_col, group_keys=False).apply(_z_by)
    z_sector = z_sector.reindex(out.index)  # alineación explícita
    z_cap    = out.groupby("_cap_bucket", group_keys=False).apply(_z_by)
    z_cap    = z_cap.reindex(out.index)

    # Fallback si z_cap es todo NaN (poco/mala cobertura)
    if _to_float(z_cap).notna().sum() == 0:
        z_neut = _to_float(z_sector).fillna(0.0)
    else:
        z_neut = (0.5 * _to_float(z_sector) + 0.5 * _to_float(z_cap)).fillna(0.0)

    z_neut.name = "z_neut"
    return z_neut

# ----------------------------- QVM -------------------------------
def compute_qvm_scores(df: pd.DataFrame, 
                       w_quality: float = 0.40,
                       w_value: float = 0.25,
                       w_momentum: float = 0.35,
                       momentum_col: str = "momentum_score",
                       sector_col: str = "sector",
                       mcap_col: str = "market_cap") -> pd.DataFrame:
    """
    Calcula value/quality growth-aware, neutraliza por sector+cap y compone QVM.
    Devuelve df con columnas:
      value_adj, quality_adj, value_adj_neut, quality_adj_neut, qvm_score
    (y deja momentum zscoreado embebido en la fórmula)
    """
    d = df.copy()

    # Limpia duplicados que crean columnas 2D
    if hasattr(d, "columns"):
        d = d.loc[:, ~d.columns.duplicated(keep="last")]

    # Sector seguro
    if sector_col not in d.columns:
        d[sector_col] = "Unknown"
    d[sector_col] = d[sector_col].astype(str).fillna("Unknown")

    # Momentum seguro (1D numérico)
    if momentum_col not in d.columns:
        d[momentum_col] = 0.0
    d[momentum_col] = _to_float(d[momentum_col])

    # Value / Quality “growth/intangible aware”
    d["value_adj"]   = value_growth_aware(d)
    d["quality_adj"] = quality_intangible_aware(d)

    # Neutralización por sector+cap (blindea índices/nombres)
    d["value_adj_neut"]   = neutralize_by_sector_cap(d, "value_adj",  sector_col, mcap_col)
    d["quality_adj_neut"] = neutralize_by_sector_cap(d, "quality_adj", sector_col, mcap_col)

    # Z de momentum (global)
    m_z = _zscore(d[momentum_col])

    # Composición QVM
    d["qvm_score"] = (
        w_quality * _zscore(d["quality_adj_neut"]) +
        w_value   * _zscore(d["value_adj_neut"])   +
        w_momentum* m_z
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return d


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
