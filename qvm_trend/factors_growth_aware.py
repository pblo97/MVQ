from __future__ import annotations
import numpy as np
import pandas as pd

# ============================= Utils =============================

def _to_float(s: pd.Series | np.ndarray | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
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

# ----------------------- Intangibles / I+D -----------------------

def capitalize_rd(
    df: pd.DataFrame,
    rd_col: str = "rd_expense_ttm",
    amort_years: int = 3
) -> pd.DataFrame:
    """
    Capitaliza I+D (80%) y genera:
      - rd_asset
      - op_income_xrd (operating_income_ttm + amort. I+D)
      - assets_xrd (total_assets_ttm + rd_asset)
    Requiere: rd_expense_ttm, operating_income_ttm, total_assets_ttm
    """
    out = df.copy()
    required = {rd_col, "operating_income_ttm", "total_assets_ttm"}
    if not required.issubset(out.columns):
        return out.assign(
            rd_asset=np.nan,
            op_income_xrd=np.nan,
            assets_xrd=out.get("total_assets_ttm", np.nan),
        )

    rd = pd.to_numeric(out[rd_col], errors="coerce").fillna(0.0)
    op_inc = pd.to_numeric(out["operating_income_ttm"], errors="coerce").fillna(0.0)
    tot_assets = pd.to_numeric(out["total_assets_ttm"], errors="coerce").fillna(0.0)

    cap_ratio = 0.80
    rd_asset = cap_ratio * rd * amort_years
    amort = rd_asset / amort_years

    out["rd_asset"] = rd_asset
    out["op_income_xrd"] = op_inc + amort
    out["assets_xrd"] = tot_assets + rd_asset
    return out

# ----------------------------- Value -----------------------------

def value_growth_aware(df: pd.DataFrame) -> pd.Series:
    """
    Value growth-aware:
      40% inv(EV/EBITDA NTM) + 30% inv(EV/GP TTM) + 30% inv(EV/Sales NTM penalizado por Capex/Sales)
      + boost si FCF_yield_5y (ajustado por SBC) >= p80
    """
    out = df.copy()

    # fuerza numérico
    for col in ["ev", "ebitda_ntm", "gross_profit_ttm", "sales_ntm",
                "capex_ttm", "sbc_ttm", "fcf_ttm", "fcf_5y_median"]:
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

    ev_over_ebitda = _safe_div(ev, ebitda_ntm)
    ev_over_gp = _safe_div(ev, gp)
    ev_over_sales = _safe_div(ev, sales_ntm)
    capex_sales = _safe_div(capex, sales_ntm).fillna(0.0)
    ev_over_sales_pen = ev_over_sales * (1.0 + capex_sales)

    v1 = _winsorize(1.0 / ev_over_ebitda.replace(0, np.nan))
    v2 = _winsorize(1.0 / ev_over_gp.replace(0, np.nan))
    v3 = _winsorize(1.0 / ev_over_sales_pen.replace(0, np.nan))

    raw = 0.40 * _zscore(v1.fillna(0)) + 0.30 * _zscore(v2.fillna(0)) + 0.30 * _zscore(v3.fillna(0))

    fcf_yield5 = _safe_div((fcf_5y_median - sbc), ev)
    out["_fcf_yield5_pct"] = _rank_pct(fcf_yield5)
    boost = (out["_fcf_yield5_pct"] >= 0.80).astype(float) * 0.25
    return raw + boost

# ---------------------------- Quality ----------------------------

def quality_intangible_aware(df: pd.DataFrame) -> pd.Series:
    """
    Quality ajustado por intangibles:
      - GP/Assets_xRD
      - ROIC_xRD (NOPAT_xRD / InvestedCapital_xRD)
      - Estabilidad de márgenes (inv. de std 5y)
      - Accruals (NOA) bajos
      - NetCash/EBITDA
    """
    out = capitalize_rd(df).copy()

    # coerción numérica
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
    ic = out.get("invested_capital_ttm",
                 out.get("total_assets_ttm", pd.Series(index=out.index, data=np.nan))
                 - out.get("current_liabilities_ttm", 0))

    tax_rate = out.get("tax_rate", pd.Series(index=out.index, data=0.20)).fillna(0.20)
    nopat_xrd = out.get("op_income_xrd", out.get("operating_income_ttm", 0)).fillna(0.0) * (1 - tax_rate)

    gp_assets = _winsorize(_safe_div(gp, assets_xrd))
    roic_xrd = _winsorize(_safe_div(nopat_xrd, ic))

    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0)
            if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(index=out.index, data=np.nan)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median())))

    accruals = _winsorize(noa.fillna(noa.median()) if noa is not None else pd.Series(index=out.index, data=0.0))
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(net_debt.fillna(0), _to_float(ebitda).abs() + 1e-9))

    return (
        0.35 * _zscore(gp_assets) +
        0.35 * _zscore(roic_xrd) +
        0.10 * stab +
        0.10 * _zscore(netcash_ebitda) +
        0.10 * accruals_score
    )

# ------------------- Sector & Cap Neutralization -----------------

def neutralize_by_sector_cap(
    df: pd.DataFrame,
    score_col: str,
    sector_col: str = "sector",
    mcap_col: str = "market_cap",
    buckets=(
        ("Mega", 150e9, np.inf),
        ("Large", 10e9, 150e9),
        ("Mid",   2e9,  10e9),
        ("Small", 0,    2e9),
    ),
) -> pd.Series:
    """
    Devuelve score neutralizado por sector y por bucket de market cap:
        final = 0.5 * z_sector + 0.5 * z_capbucket
    """
    out = df.copy()
    out[sector_col] = out.get(sector_col, "Unknown").fillna("Unknown").astype(str)
    out[mcap_col] = pd.to_numeric(out.get(mcap_col), errors="coerce").fillna(0.0)

    # buckets por mcap
    bins = [b[1] for b in buckets] + [buckets[-1][2]]
    labels = [b[0] for b in buckets]
    out["_cap_bucket"] = pd.cut(out[mcap_col], bins=bins, labels=labels, include_lowest=True, right=False)

    def z_by(group: pd.DataFrame) -> pd.Series:
        return _zscore(group[score_col])

    z_sector = out.groupby(sector_col, group_keys=False).apply(z_by).rename("z_sector")
    z_cap = out.groupby("_cap_bucket", group_keys=False).apply(z_by).rename("z_cap")

    return 0.5 * z_sector + 0.5 * z_cap

# ----------------------------- QVM -------------------------------

def compute_qvm_scores(
    df: pd.DataFrame,
    w_quality: float = 0.40,
    w_value:   float = 0.25,
    w_momentum:float = 0.35,
    momentum_col: str = "momentum_score",
    sector_col:   str = "sector",
    mcap_col:     str = "market_cap",
) -> pd.DataFrame:
    """
    Calcula Value y Quality growth-aware, neutraliza por sector+cap
    y devuelve: value_adj, quality_adj, value_adj_neut, quality_adj_neut, qvm_score
    """
    df = df.copy()

    # tipos seguros
    df[momentum_col] = pd.to_numeric(df.get(momentum_col), errors="coerce")
    df[sector_col] = df.get(sector_col, "Unknown").fillna("Unknown").astype(str)
    df[mcap_col] = pd.to_numeric(df.get(mcap_col), errors="coerce")

    df["value_adj"] = value_growth_aware(df)
    df["quality_adj"] = quality_intangible_aware(df)

    df["value_adj_neut"] = neutralize_by_sector_cap(df, "value_adj", sector_col, mcap_col)
    df["quality_adj_neut"] = neutralize_by_sector_cap(df, "quality_adj", sector_col, mcap_col)

    m = _zscore(df[momentum_col].fillna(df[momentum_col].median()))
    df["qvm_score"] = (
        w_quality * _zscore(df["quality_adj_neut"]) +
        w_value   * _zscore(df["value_adj_neut"])   +
        w_momentum* m
    )
    return df

# ----------------------- Guardrails/overrides --------------------

def apply_megacap_rules(
    df: pd.DataFrame,
    momentum_col: str = "momentum_score",
    quality_col:  str = "quality_adj_neut",
    value_col:    str = "value_adj_neut",
) -> pd.DataFrame:
    """
    Reglas:
      - Permite peso si Value ∈ [35p,45p] y Momentum≥70p y Quality≥55p (sector).
      - Marca 'quality_too_low' si Quality<45p.
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
