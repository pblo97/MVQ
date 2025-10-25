# qvm_trend/pm/orchestrator.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.covariance import LedoitWolf  # (opcional; usado si activas Kelly vectorial)

from qvm_trend.data_io import load_prices_panel
from qvm_trend.macro.macro_score import z_to_regime


# =========================
# Helpers de retornos
# =========================
def monthly_rets(ret_daily: pd.Series) -> pd.Series:
    """Convierte retornos diarios a mensuales (compuestos)."""
    r = pd.to_numeric(ret_daily, errors="coerce").dropna()
    if r.empty:
        return pd.Series(dtype=float)
    r_m = (1.0 + r).resample("M").apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    return r_m.dropna()


# =========================
# Kelly "pro" por símbolo
# =========================
def _winsor(s: pd.Series, p: float = 0.02) -> pd.Series:
    if s is None or s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1.0 - p)
    return s.clip(lo, hi)

def _ewm_mean_std(x: pd.Series, span: int = 12):
    m = x.ewm(span=span, min_periods=max(3, span // 3)).mean()
    v = (x - m).pow(2).ewm(span=span, min_periods=max(3, span // 3)).mean()
    return m, np.sqrt(v)

def kelly_metrics_single(
    asset_excess_m: pd.Series,
    *,
    costs_per_period: float = 0.001,  # 0.1% mensual por costos/derrape
    winsor_p: float = 0.02,
    shrink_kappa: int = 20,
    ewm_span: int = 12,
    min_months: int = 36,
) -> Dict[str, float]:
    """
    Métricas Kelly robustas por símbolo usando retornos MENSUALES en EXCESO al benchmark.
    Devuelve: n, p, payoff, mu, sigma, k_bin, k_cont, k_raw
    """
    r = pd.to_numeric(asset_excess_m, errors="coerce").dropna()
    if r.size < min_months:
        return dict(n=r.size, p=np.nan, payoff=np.nan, mu=np.nan, sigma=np.nan,
                    k_bin=0.0, k_cont=0.0, k_raw=0.0)

    r = _winsor(r, p=winsor_p)

    # p(hit) y payoff (en exceso)
    gains = r[r > 0]
    losses = -r[r < 0]
    hits = gains.size
    misses = losses.size
    p_emp = hits / (hits + misses) if (hits + misses) > 0 else 0.5
    payoff_emp = (gains.mean() / losses.mean()) if (hits > 0 and losses.size > 0 and losses.mean() > 0) else 1.0

    # Shrinkage bayesiano hacia 0.5 y 1.0
    n = r.size
    p_hat = (p_emp * n + 0.5 * shrink_kappa) / (n + shrink_kappa)
    payoff_hat = (payoff_emp * n + 1.0 * shrink_kappa) / (n + shrink_kappa)

    # Kelly binomial (cap [0,1])
    k_bin = p_hat - (1.0 - p_hat) / max(payoff_hat, 1e-6)
    k_bin = float(np.clip(k_bin, 0.0, 1.0))

    # Kelly continuo ~ mu/sigma^2 usando EWMA (restando costos)
    mu_ewm, sigma_ewm = _ewm_mean_std(r, span=ewm_span)
    mu = float(mu_ewm.iloc[-1] - costs_per_period)
    sigma = float(sigma_ewm.iloc[-1])
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-8:
        k_cont = 0.0
    else:
        k_cont = float(np.clip(mu / (sigma ** 2), 0.0, 1.0))

    # Mezcla
    k_raw = float(0.5 * k_bin + 0.5 * k_cont)

    return dict(n=int(n), p=float(p_hat), payoff=float(payoff_hat),
                mu=float(mu if np.isfinite(mu) else np.nan),
                sigma=float(sigma if np.isfinite(sigma) else np.nan),
                k_bin=float(k_bin), k_cont=float(k_cont), k_raw=float(k_raw))

def penalize_by_corr(k_series: pd.Series, ret_excess_df: pd.DataFrame, lambda_corr: float = 0.5) -> pd.Series:
    """
    Penaliza k por similitud con el proto-portfolio (promedio de activos con k>0).
    k' = k / (1 + λ * max(0, ρ_i,proto))
    """
    k = k_series.clip(lower=0.0).copy()
    keep = k[k > 0].index
    if len(keep) < 2:
        return k.fillna(0.0)
    proto = ret_excess_df[keep].mean(axis=1).dropna()
    pen = {}
    for s in k.index:
        a = ret_excess_df.get(s)
        if a is None:
            pen[s] = 1.0
            continue
        c = pd.concat([proto, a], axis=1).dropna()
        if c.shape[0] < 12:
            pen[s] = 1.0
        else:
            rho = float(c.corr().iloc[0, 1])
            pen[s] = 1.0 / (1.0 + lambda_corr * max(0.0, rho))
    return (k * pd.Series(pen)).fillna(0.0)


# =========================
# Otros helpers
# =========================
def _beta_vs_bench(x: pd.Series, y: pd.Series) -> float:
    """β ~ Cov(x, y) / Var(y) sobre índice común (mensual)."""
    idx = x.index.intersection(y.index)
    if len(idx) < 12:
        return np.nan
    xm = x.reindex(idx).dropna()
    ym = y.reindex(idx).dropna()
    idx2 = xm.index.intersection(ym.index)
    if len(idx2) < 12:
        return np.nan
    xv = xm.reindex(idx2).values
    yv = ym.reindex(idx2).values
    vy = float(np.var(yv, ddof=1))
    if vy == 0:
        return np.nan
    cov = float(np.cov(xv, yv, ddof=1)[0, 1])
    return cov / vy

def _quality_tilt(q: pd.Series | np.ndarray, alpha: float) -> pd.Series:
    """
    exp(alpha * zscore(q)) acotado [0.5, 2.0].
    Si q vacío → 1.0.
    """
    if q is None:
        return pd.Series([], dtype=float)
    q = pd.Series(q).astype(float)
    if q.dropna().empty:
        return pd.Series(1.0, index=q.index)
    z = (q - q.mean()) / (q.std(ddof=1) + 1e-12)
    mult = np.exp(alpha * z).clip(0.5, 2.0)
    return pd.Series(mult, index=q.index)

def _apply_beta_cap(w: np.ndarray, betas: np.ndarray, beta_cap: float) -> np.ndarray:
    betas = np.nan_to_num(betas, nan=1.0)
    total_beta_w = float(np.sum(w * betas))
    if total_beta_w > beta_cap and total_beta_w > 0:
        return w * (beta_cap / total_beta_w)
    return w

def _finalize_weights(
    w_base: np.ndarray,
    betas: np.ndarray,
    beta_cap: float,
    pos_cap: float,
    enforce_sum1: bool = True,
) -> np.ndarray:
    w = np.minimum(w_base, float(pos_cap))
    w = _apply_beta_cap(w, betas, float(beta_cap))
    if enforce_sum1 and w.sum() > 0:
        w = w / w.sum()
    return w

def _pick_alpha(reg_label: str, alpha_off: float, alpha_neu: float, alpha_on: float) -> float:
    return alpha_off if reg_label == "OFF" else (alpha_on if reg_label == "ON" else alpha_neu)


# =========================
# Orchestrator principal
# =========================
def build_portfolio(
    symbols: List[str],
    bench: str,
    start: str,
    end: str,
    *,
    # Kelly pro
    base_kelly: float = 0.20,
    winsor_p: float = 0.02,
    min_months: int = 36,
    costs_per_period: float = 0.001,
    shrink_kappa: int = 20,
    ewm_span: int = 12,
    lambda_corr: float = 0.5,
    # Mezcla anterior (compat)
    t0: float = 2.0,
    lam_blend: float = 0.2,
    # Macro / quality
    macro_z: float = 0.0,
    quality_df: pd.DataFrame | None = None,   # ['symbol','QualityScore'] o ['symbol','VFQ']
    alpha_off: float = 0.30,
    alpha_neu: float = 0.10,
    alpha_on: float = 0.00,
    # Caps y gating
    enforce_sum1: bool = True,
    pos_cap: float = 0.05,
    beta_cap_user: float = 1.0,
    allow_new_when_z_below: float = -0.5,
    current_holdings: List[str] | None = None,
) -> pd.DataFrame:
    """
    Devuelve DF ordenado por 'weight' con:
    ['symbol','p','payoff','k_bin','k_cont','k_raw','k_pen','mu','sigma','beta','n','weight','beta_w']
    """
    # 1) Precios y retornos
    pnl = load_prices_panel(symbols + [bench], start, end, cache_key="pm_panel")
    bench_df = pnl.get(bench)
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return pd.DataFrame()

    bench_d = bench_df["close"].pct_change().dropna()
    bench_m = monthly_rets(bench_d)

    ret_m, ret_excess, rows, used = {}, {}, [], []
    for s in symbols:
        df = pnl.get(s)
        if df is None or df.empty or "close" not in df.columns:
            continue
        a_d = df["close"].pct_change().dropna()
        a_m = monthly_rets(a_d)
        common = a_m.index.intersection(bench_m.index)
        if len(common) < int(min_months):
            continue

        a_m = a_m.loc[common]
        b_m = bench_m.loc[common]
        ex_m = (a_m - b_m).dropna()
        if ex_m.size < int(min_months):
            continue

        ret_m[s] = a_m
        ret_excess[s] = ex_m

        km = kelly_metrics_single(
            ex_m,
            costs_per_period=costs_per_period,
            winsor_p=winsor_p,
            shrink_kappa=shrink_kappa,
            ewm_span=ewm_span,
            min_months=min_months,
        )
        beta = _beta_vs_bench(a_m, b_m)
        rows.append({"symbol": s, **km, "beta": beta})
        used.append(s)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 2) Penalización por correlación → k_pen
    ret_excess_df = pd.DataFrame(ret_excess)  # cols=símbolos
    k_pen = penalize_by_corr(
        df.set_index("symbol")["k_raw"],
        ret_excess_df,
        lambda_corr=lambda_corr
    )

    # asegurar índice y orden
    df = df.set_index("symbol")
    if not isinstance(k_pen, pd.Series):
        k_pen = pd.Series(k_pen, index=df.index)
    else:
        k_pen = k_pen.reindex(df.index)

    # ---- floor suave + mínimo de activos + mezcla diversificada ----
    # a) floor suave (anti-ruido)
    k_pen = k_pen.clip(lower=0.0)
    k_floor = 0.001  # 0.1% de Kelly
    k_pen = k_pen.where(k_pen >= k_floor, 0.0)

    # b) diversificación mínima
    min_active = 6
    actives = int((k_pen > 0).sum())
    if actives < min_active:
        # relaja floor: cuenta todo >0
        k_pen = k_pen.where(k_pen > 0.0, 0.0)
        actives = int((k_pen > 0).sum())
        if actives < min_active:
            # enciende por μ de exceso los top faltantes
            need = min_active - actives
            already = set(k_pen[k_pen > 0].index)
            top_mu = df.loc[~df.index.isin(already), "mu"].sort_values(ascending=False)
            push = top_mu.head(max(need, 0)).index
            eps = 1e-4
            k_pen.loc[push] = np.maximum(k_pen.loc[push], eps)

    df["k_pen"] = k_pen

    # c) normaliza Kelly fraccionado
    k_base = df["k_pen"].fillna(0.0).clip(lower=0.0).values
    if k_base.sum() <= 0:
        w_k = np.ones(len(df)) / len(df)
    else:
        w_k = k_base / k_base.sum()
    w_k = base_kelly * w_k

    # d) mezcla diversificada estable sobre top-μ
    eta = 0.15                   # 15% diversificado
    N_u = max(min_active, min(8, len(df)))
    u_idx = df["mu"].sort_values(ascending=False).head(N_u).index
    u = pd.Series(0.0, index=df.index)
    u.loc[u_idx] = 1.0 / N_u
    w_k = (1.0 - eta) * w_k + eta * u.values

    # 3) Overlay macro y caps efectivos
    reg = z_to_regime(float(macro_z))
    M_macro = reg.m_multiplier
    beta_cap_eff = min(float(beta_cap_user), float(reg.beta_cap))
    pos_cap_eff  = min(float(pos_cap), float(reg.vol_cap))

    # 4) Quality tilt
    if quality_df is not None and not quality_df.empty:
        if "QualityScore" in quality_df.columns:
            qmap = dict(zip(quality_df["symbol"].astype(str).str.upper(), quality_df["QualityScore"]))
        elif "VFQ" in quality_df.columns:
            qmap = dict(zip(quality_df["symbol"].astype(str).str.upper(), quality_df["VFQ"]))
        else:
            qmap = {}
        q_series = pd.Series(df.index).astype(str).str.upper().map(qmap)
        q_series.index = df.index
        q_series = q_series.fillna(df.get("mu", 0.0))
    else:
        q_series = df.get("mu", pd.Series(0.0, index=df.index))

    alpha = _pick_alpha(reg.label, alpha_off, alpha_neu, alpha_on)
    q_mult = _quality_tilt(q_series, alpha).reindex(df.index).fillna(1.0).values

    # 5) Gate táctico (bloqueo de nuevas si macro malo)
    T_gate = np.ones(len(df))
    if current_holdings is not None and float(macro_z) < float(allow_new_when_z_below):
        holdset = {h.upper() for h in current_holdings}
        is_new = ~pd.Series(df.index, index=df.index).astype(str).str.upper().isin(holdset)
        T_gate[is_new.values] = 0.0

    # 6) Combinar y normalizar
    w_base = w_k * M_macro * q_mult * T_gate
    if w_base.sum() <= 0:
        w_base = w_k
    else:
        w_base = w_base / max(w_base.sum(), 1e-12)

    # 7) Caps finales y suma=1
    w_final = _finalize_weights(
        w_base,
        betas=df["beta"].fillna(1.0).values,
        beta_cap=beta_cap_eff,
        pos_cap=pos_cap_eff,
        enforce_sum1=enforce_sum1,
    )

    df["weight"] = w_final
    df["beta_w"] = df["beta"].fillna(1.0) * df["weight"]
    df = df.reset_index()

    # columnas para UI
    cols = ["symbol",
            "p", "payoff", "k_bin", "k_cont", "k_raw", "k_pen",
            "mu", "sigma", "beta", "n",
            "weight", "beta_w"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c != "symbol" else df.get(c, "")
    return df[cols].sort_values("weight", ascending=False).reset_index(drop=True)
