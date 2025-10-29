# ======================= QVM CORE (todo en uno) =======================
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------- Utils -----------------------------
def _to_float(s: pd.Series | pd.DataFrame | np.ndarray | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    elif isinstance(s, np.ndarray):
        s = s[:, 0] if s.ndim > 1 else s
        s = pd.Series(s)
    elif not isinstance(s, pd.Series):
        s = pd.Series(s)
    return pd.to_numeric(s, errors="coerce").astype(float)

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
    out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _safe_div(a, b) -> pd.Series:
    a = _to_float(a)
    b = _to_float(b)
    out = a.div(b)
    return out.replace([np.inf, -np.inf], np.nan)

def _rank_pct(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    return s.rank(pct=True, method="average").fillna(0.0)

def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return _to_float(df[name])
    return pd.Series(default, index=df.index, dtype=float)

def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    return _to_float(df.get(col))

# --------------------- Estandarizador de inputs ---------------------
class FundamentalStandardizer:
    ALIAS = {
        "symbol": ["symbol","ticker","sym","Symbol"],
        "sector": ["sector","Sector","industry","Industry","gicsSector","GICS_Sector"],
        "market_cap": ["market_cap","marketCap_unified","marketCap","MarketCap","mkt_cap"],
        # value
        "ev": ["ev","enterpriseValue","EnterpriseValue","EV"],
        "ebitda_ttm": ["ebitda_ttm","EBITDA_TTM","ebitdaTrailingTwelveMonths","ebitdaTTM","EBITDAttm","ebitda"],
        "ebitda_ntm": ["ebitda_ntm","EBITDA_NTM","ebitdaForward","ebitdaNextTwelveMonths"],
        "gross_profit_ttm": ["gross_profit_ttm","grossProfitTTM","GrossProfitTTM","gp_ttm"],
        "sales_ntm": ["sales_ntm","revenueNTM","RevenueNTM","salesForward","revenueForward"],
        "capex_ttm": ["capex_ttm","capexTTM","CapExTTM","capitalExpenditureTTM"],
        "sbc_ttm": ["sbc_ttm","stockBasedCompTTM","shareBasedCompTTM","SBC_TTM","stock_based_comp_ttm"],
        "fcf_ttm": ["fcf_ttm","freeCashFlowTTM","FCF_TTM"],
        "fcf_5y_median": ["fcf_5y_median","FCF_5Y_MEDIAN","fcfMedian5Y"],
        # quality / intangible
        "rd_expense_ttm": ["rd_expense_ttm","researchDevelopmentTTM","R&D_TTM","researchAndDevTTM"],
        "operating_income_ttm": ["operating_income_ttm","operatingIncomeTTM","OperatingIncomeTTM","ebitTTM","EBIT_TTM"],
        "total_assets_ttm": ["total_assets_ttm","totalAssetsTTM","TotalAssetsTTM","totalAssets"],
        "current_liabilities_ttm": ["current_liabilities_ttm","currentLiabilitiesTTM","CurrentLiabilitiesTTM","current_liabilities"],
        "invested_capital_ttm": ["invested_capital_ttm","InvestedCapitalTTM","investedCapitalTTM","invested_capital"],
        "net_debt_ttm": ["net_debt_ttm","NetDebtTTM","netDebt"],
        "total_debt_ttm": ["total_debt_ttm","TotalDebtTTM","total_debt"],
        "cash_ttm": ["cash_ttm","cashAndEquivalentsTTM","CashAndEquivalentsTTM","cash"],
        "noa_ttm": ["noa_ttm","netOperatingAssetsTTM","NOA_TTM"],
        "tax_rate": ["tax_rate","effectiveTaxRate","effectiveTaxRateTTM"],
        "op_margin_hist": ["op_margin_hist","operatingMarginHistory","opMarginHistory"],
        # técnico
        "BreakoutScore": ["BreakoutScore"],
        "RVOL20": ["RVOL20"],
        "UDVol20": ["UDVol20"],
        "hits": ["hits"],
        "P52": ["P52"],
        "ClosePos": ["ClosePos"],
        "momentum_score": ["momentum_score","mom","mom_sig","mom_px","momentum_score_prices"],
    }

    REQUIRED = [
        "symbol","sector","market_cap","ev",
        "ebitda_ttm","ebitda_ntm","gross_profit_ttm","sales_ntm",
        "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median",
        "rd_expense_ttm","operating_income_ttm","total_assets_ttm",
        "current_liabilities_ttm","invested_capital_ttm",
        "net_debt_ttm","noa_ttm","tax_rate","op_margin_hist",
        "BreakoutScore","RVOL20","UDVol20","hits","P52","ClosePos",
        "momentum_score"
    ]

    def _apply_alias(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for target, cands in self.ALIAS.items():
            if target in d.columns:  # ya está
                continue
            for c in cands:
                if c in d.columns:
                    d[target] = d[c]; break
        return d

    def _ensure_required(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in self.REQUIRED:
            if c not in d.columns:
                d[c] = np.nan
        return d

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # sector
        d["sector"] = d.get("sector", "Unknown")
        d["sector"] = d["sector"].astype(str).replace({"nan": "Unknown"}).fillna("Unknown")
        # market cap
        if d.get("market_cap") is None or _to_float(d["market_cap"]).isna().all():
            for alt in ("marketCap_unified","marketCap","MarketCap"):
                if alt in d.columns:
                    d["market_cap"] = pd.to_numeric(d[alt], errors="coerce"); break
        # numéricos
        for c in [x for x in self.REQUIRED if x not in ("symbol","sector","op_margin_hist")]:
            d[c] = _to_float(d.get(c))
        # op_margin_hist a lista o NaN
        if "op_margin_hist" in d.columns:
            def _sanitize_hist(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return list(pd.to_numeric(pd.Series(x), errors="coerce"))
                return np.nan
            d["op_margin_hist"] = d["op_margin_hist"].apply(_sanitize_hist)
        else:
            d["op_margin_hist"] = np.nan
        # momentum: coalesce si vino por varias
        cand_moms = [c for c in self.ALIAS["momentum_score"] if c in d.columns and c != "momentum_score"]
        if cand_moms:
            d["momentum_score"] = d[cand_moms].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            d.drop(columns=[c for c in cand_moms if c in d.columns], inplace=True, errors="ignore")
        d["momentum_score"] = pd.to_numeric(d.get("momentum_score", 0.0), errors="coerce").fillna(0.0)
        return d

    def _derive_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if d["net_debt_ttm"].isna().all():
            if "total_debt_ttm" in d.columns and "cash_ttm" in d.columns:
                d["net_debt_ttm"] = _to_float(d["total_debt_ttm"]) - _to_float(d["cash_ttm"])
        if d["ev"].isna().all():
            d["ev"] = _to_float(d["market_cap"])
        d["tax_rate"] = _to_float(d["tax_rate"]).fillna(0.20)
        return d

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.loc[:, ~d.columns.duplicated(keep="last")]
        d = self._apply_alias(d)
        d = self._ensure_required(d)
        d = self._coerce_types(d)
        d = self._derive_missing(d)
        # última pasada: asegurar 1D numérica donde aplica
        for col in [c for c in self.REQUIRED if c not in ("symbol","sector","op_margin_hist")]:
            d[col] = _to_float(d[col])
        return d

# ----------------------- Intangibles / I+D -----------------------
def capitalize_rd(df: pd.DataFrame, rd_col="rd_expense_ttm", amort_years: int = 3) -> pd.DataFrame:
    out = df.copy()
    req = {rd_col, "operating_income_ttm", "total_assets_ttm"}
    if not req.issubset(out.columns):
        out["rd_asset"] = np.nan
        out["op_income_xrd"] = out.get("operating_income_ttm", np.nan)
        out["assets_xrd"] = out.get("total_assets_ttm", np.nan)
        return out
    rd  = _to_float(out[rd_col]).fillna(0.0)
    opi = _to_float(out["operating_income_ttm"]).fillna(0.0)
    ta  = _to_float(out["total_assets_ttm"]).fillna(0.0)
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
    if ev.isna().all(): ev = _col(out, "market_cap")
    ebitda_ntm = _col(out, "ebitda_ntm")
    gp_ttm     = _col(out, "gross_profit_ttm")
    sales_ntm  = _col(out, "sales_ntm")
    capex_ttm  = _col(out, "capex_ttm", 0.0).fillna(0.0)
    sbc_ttm    = _col(out, "sbc_ttm",   0.0).fillna(0.0)
    fcf_ttm    = _col(out, "fcf_ttm")
    fcf5_med   = _col(out, "fcf_5y_median")
    if fcf5_med.isna().all(): fcf5_med = fcf_ttm

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
    if assets_xrd.isna().all(): assets_xrd = _col(out, "total_assets_ttm")

    ebitda = _col(out, "ebitda_ttm")
    if ebitda.isna().all(): ebitda = _col(out, "ebitda_ntm")

    net_debt = _col(out, "net_debt_ttm").fillna(0.0)
    ic = _col(out, "invested_capital_ttm")
    if ic.isna().all():
        ic = _col(out, "total_assets_ttm") - _col(out, "current_liabilities_ttm", 0.0)

    tax_rate = _col(out, "tax_rate", 0.20).fillna(0.20)
    opi_xrd  = _col(out, "op_income_xrd")
    if opi_xrd.isna().all(): opi_xrd = _col(out, "operating_income_ttm").fillna(0.0)
    nopat_xrd = opi_xrd * (1 - tax_rate)

    gp_assets = _winsorize(_safe_div(gp, assets_xrd), 0.01)
    roic_xrd  = _winsorize(_safe_div(nopat_xrd, ic),  0.01)

    if "op_margin_hist" in out.columns:
        std_margin = out["op_margin_hist"].apply(
            lambda xs: np.nanstd(np.asarray(xs), ddof=0) if isinstance(xs, (list, tuple, np.ndarray)) else np.nan
        )
    else:
        std_margin = pd.Series(np.nan, index=out.index)
    stab = -_zscore(_winsorize(std_margin.fillna(std_margin.median()), 0.01))

    accruals = _winsorize(_col(out, "noa_ttm").fillna(_col(out, "noa_ttm").median()), 0.01)
    accruals_score = -_zscore(accruals)

    ebitda_series = _to_float(ebitda)              # <- evita numpy.float64.abs
    netcash_ebitda = _winsorize(-_safe_div(net_debt, ebitda_series.abs() + 1e-9), 0.01)

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
    out = df.copy()
    if score_col not in out.columns:
        return pd.Series(0.0, index=out.index, name="z_neut")

    out["_score_"] = _to_float(out[score_col])
    out[sector_col] = out.get(sector_col, "Unknown")
    out[sector_col] = out[sector_col].astype(str).fillna("Unknown")
    out[mcap_col] = _to_float(out.get(mcap_col))

    def _z(s):
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = s.mean(skipna=True), s.std(skipna=True)
        return (s - mu) / (sd if (sd and sd > 0) else 1.0)

    try:
        b_sorted = sorted(list(buckets), key=lambda t: float(t[1]))
        lows  = [float(b[1]) for b in b_sorted]
        highs = [float(b[2]) for b in b_sorted]
        bins = lows + [highs[-1]]
        out["_cap_bucket"] = pd.cut(out[mcap_col], bins=bins, labels=[b[0] for b in b_sorted],
                                    include_lowest=True, right=False)
        if out["_cap_bucket"].isna().all():
            out["_cap_bucket"] = "Cap_All"
    except Exception:
        out["_cap_bucket"] = "Cap_All"

    z_sector = out.groupby(sector_col, group_keys=False)["_score_"].apply(_z).reindex(out.index)
    z_cap    = out.groupby("_cap_bucket", group_keys=False)["_score_"].apply(_z).reindex(out.index)

    if _to_float(z_sector).abs().sum(skipna=True) == 0:
        z_sector = _z(out["_score_"])
    if _to_float(z_cap).abs().sum(skipna=True) == 0:
        z_cap = _z(out["_score_"])

    z_neut = (0.5*_to_float(z_sector) + 0.5*_to_float(z_cap)).fillna(0.0)
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
    d = df.copy()
    d = d.loc[:, ~d.columns.duplicated(keep="last")]

    if sector_col not in d.columns: d[sector_col] = "Unknown"
    d[sector_col] = d[sector_col].astype(str).fillna("Unknown")

    if momentum_col not in d.columns: d[momentum_col] = 0.0
    d[momentum_col] = _to_float(d[momentum_col])

    d["value_adj"]   = value_growth_aware(d)
    d["quality_adj"] = quality_intangible_aware(d)

    d["value_adj_neut"]   = neutralize_by_sector_cap(d, "value_adj",  sector_col, mcap_col)
    d["quality_adj_neut"] = neutralize_by_sector_cap(d, "quality_adj", sector_col, mcap_col)

    m_z = _zscore(d[momentum_col])
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
# ===================== FIN QVM CORE (todo en uno) =====================


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
