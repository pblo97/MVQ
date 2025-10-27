from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------- Utils -----------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.isfinite(b) & (b != 0))

def _rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

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
    if not set([rd_col, "operating_income_ttm", "total_assets_ttm"]).issubset(out.columns):
        return out.assign(rd_asset=np.nan, op_income_xrd=np.nan, assets_xrd=out.get("total_assets_ttm", np.nan))

    rd = out[rd_col].fillna(0.0)
    cap_ratio = 0.80
    rd_asset = cap_ratio * rd * amort_years
    amort = rd_asset / amort_years

    out["rd_asset"] = rd_asset
    out["op_income_xrd"] = out["operating_income_ttm"].fillna(0) + amort
    out["assets_xrd"] = out["total_assets_ttm"].fillna(0) + rd_asset
    return out

# ----------------------------- Value -----------------------------
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
    ev_over_sales_pen = ev_over_sales * (1 + capex_sales)

    v1 = _winsorize(1 / (ev_over_ebitda.replace(0, np.nan)), 0.01).fillna(0)
    v2 = _winsorize(1 / (ev_over_gp.replace(0, np.nan)), 0.01).fillna(0)
    v3 = _winsorize(1 / (ev_over_sales_pen.replace(0, np.nan)), 0.01).fillna(0)
    raw = 0.40 * _zscore(v1) + 0.30 * _zscore(v2) + 0.30 * _zscore(v3)

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
      - Estabilidad de márgenes (inv. de la desviación 5y)
      - Accruals (NOA) bajos
      - NetCash/EBITDA
    """
    out = capitalize_rd(df)

    gp = out.get("gross_profit_ttm")
    assets_xrd = out.get("assets_xrd", out.get("total_assets_ttm"))
    ebitda = out.get("ebitda_ttm", out.get("ebitda_ntm"))
    net_debt = out.get("net_debt_ttm")
    noa = out.get("noa_ttm")
    ic = out.get("invested_capital_ttm", out.get("total_assets_ttm") - out.get("current_liabilities_ttm", 0))

    tax_rate = out.get("tax_rate", pd.Series(index=out.index, data=0.20)).fillna(0.20)
    nopat_xrd = out.get("op_income_xrd", out.get("operating_income_ttm", 0)) * (1 - tax_rate)

    gp_assets = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd = _winsorize(_safe_div(nopat_xrd, ic), 0.01)

    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(index=out.index, data=np.nan)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(noa.fillna(noa.median()) if noa is not None else pd.Series(index=out.index, data=0), 0.01)
    accruals_score = -_zscore(accruals)

    netcash_ebitda = _winsorize(-_safe_div(net_debt.fillna(0), ebitda.abs() + 1e-9), 0.01)

    return (
        0.35 * _zscore(gp_assets) +
        0.35 * _zscore(roic_xrd) +
        0.10 * stab +
        0.10 * _zscore(netcash_ebitda) +
        0.10 * accruals_score
    )

# ------------------- Sector & Cap Neutralization -----------------
def neutralize_by_sector_cap(df: pd.DataFrame, score_col: str, sector_col: str = "sector",
                             mcap_col: str = "market_cap", buckets=(("Mega", 150e9, np.inf),
                                                                    ("Large", 10e9, 150e9),
                                                                    ("Mid", 2e9, 10e9),
                                                                    ("Small", 0, 2e9))) -> pd.Series:
    """
    Devuelve score neutralizado por sector y bucket de market cap:
      final = 0.5*z_sector + 0.5*z_capbucket
    """
    out = df.copy()
    out["_cap_bucket"] = pd.cut(out[mcap_col].astype(float),
                                bins=[b[1] for b in buckets] + [buckets[-1][2]],
                                labels=[b[0] for b in buckets],
                                include_lowest=True, right=False)
    def z_by(group):
        return _zscore(group[score_col])
    z_sector = out.groupby(sector_col, group_keys=False).apply(z_by).rename("z_sector")
    z_cap = out.groupby("_cap_bucket", group_keys=False).apply(z_by).rename("z_cap")
    return 0.5 * z_sector + 0.5 * z_cap

# ----------------------------- QVM -------------------------------
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
    m = _zscore(df[momentum_col].astype(float))
    df["qvm_score"] = (
        w_quality * _zscore(df["quality_adj_neut"]) +
        w_value   * _zscore(df["value_adj_neut"])   +
        w_momentum* m
    )
    return df

# ----------------------- Guardrails/overrides --------------------
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
