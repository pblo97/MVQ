# qvm_trend/pm/orchestrator.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from qvm_trend.data_io import load_prices_panel
from qvm_trend.macro.macro_score import z_to_regime


# =========================
# Helpers estadísticos base
# =========================
def monthly_rets(ret_daily: pd.Series) -> pd.Series:
    """Convierte retornos diarios a mensuales (compuestos)."""
    r = pd.to_numeric(ret_daily, errors="coerce").dropna()
    if r.empty:
        return pd.Series(dtype=float)
    r_m = (1.0 + r).resample("M").apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    return r_m.dropna()

def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if s is None or s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1.0 - p)
    return s.clip(lo, hi)

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

def kelly_stats(
    asset_m: pd.Series,
    bench_m: pd.Series,
    *,
    winsor_p: float = 0.02,
    t0: float = 2.0,
    min_months: int = 36,
    lam_blend: float = 0.2,
) -> dict:
    """
    Calcula métricas robustas por activo (mensual).
    - Kelly clásico (payout) + prudencia por t-stat + mezcla con μ/σ².
    - Devuelve dict con: p, payoff, k_raw, k_blend, mu, sigma, t, beta, n, valid.
    """
    a = pd.to_numeric(asset_m, errors="coerce").dropna()
    b = pd.to_numeric(bench_m, errors="coerce").dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < int(min_months):
        return {"valid": False}

    a = a.reindex(idx).dropna()
    b = b.reindex(idx).dropna()
    n = len(a)
    if n < int(min_months):
        return {"valid": False}

    # Winsor para robustez
    a_w = _winsorize(a, winsor_p)

    # Kelly clásico por payoff (ganar/perder)
    pos = a_w[a_w > 0.0]
    neg = a_w[a_w < 0.0]
    p_hit = float((a_w > 0.0).mean()) if n > 0 else np.nan
    avg_win = float(pos.mean()) if len(pos) > 0 else np.nan
    avg_loss = float(neg.mean()) if len(neg) > 0 else np.nan
    payoff = float(abs(avg_win / avg_loss)) if (avg_loss not in (0.0, None, np.nan)) else np.nan

    k_raw = 0.0
    if payoff is not None and np.isfinite(payoff) and payoff > 0:
        k_raw = max(0.0, min(1.0, p_hit - (1.0 - p_hit) / payoff))

    # Estimadores μ, σ, t-stat (mensual)
    mu = float(a_w.mean())
    sigma = float(a_w.std(ddof=1)) if a_w.std(ddof=1) > 0 else np.nan
    tstat = float((mu / (sigma / np.sqrt(n))) if (sigma and sigma > 0) else 0.0)

    # Mezcla con μ/σ² para reflejar info de escala/estabilidad
    mu_sig2 = 0.0
    if sigma and sigma > 0:
        mu_sig2 = float(max(0.0, min(1.0, lam_blend * (mu / (sigma * sigma)))))

    # Prudencia por t-stat: si t < t0, atenúa
    prudence = float(min(1.0, max(0.0, tstat / float(t0 if t0 > 0 else 1.0))))

    # Blend final (acotado)
    k_blend = float(np.clip(0.5 * k_raw + 0.5 * mu_sig2, 0.0, 1.0) * prudence)

    beta = _beta_vs_bench(a, b)

    return {
        "valid": True,
        "p": p_hit,
        "payoff": payoff,
        "k_raw": k_raw,
        "k_blend": k_blend,
        "mu": mu,
        "sigma": sigma,
        "t": tstat,
        "beta": beta,
        "n": int(n),
    }


# =========================
# Mezclas, caps y tilts
# =========================
def _quality_tilt(q: pd.Series | np.ndarray, alpha: float) -> pd.Series:
    """
    Genera multiplicador exp(alpha * zscore(q)), acotado [0.5, 2.0].
    Si q está vacío, devuelve 1.0.
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
    base_kelly: float = 0.5,
    winsor_p: float = 0.02,
    t0: float = 2.0,
    min_months: int = 36,
    lam_blend: float = 0.2,
    macro_z: float = 0.0,
    quality_df: pd.DataFrame | None = None,  # ['symbol', 'QualityScore'] o ['symbol','VFQ']
    alpha_off: float = 0.30,
    alpha_neu: float = 0.10,
    alpha_on: float = 0.00,
    enforce_sum1: bool = True,
    pos_cap: float = 0.05,
    beta_cap_user: float = 1.0,
    allow_new_when_z_below: float = -0.5,
    current_holdings: List[str] | None = None,
) -> pd.DataFrame:
    """
    Devuelve DataFrame ordenado por 'weight' con:
    ['symbol','p','payoff','k_raw','k_blend','mu','sigma','t','beta','n','weight','beta_w']
    Lógica:
      1) Precios → retornos mensuales → 'kelly_stats' por símbolo.
      2) Peso base = normalización de k_blend * base_kelly.
      3) Macro: z_to_regime → multiplicador M_macro y caps efectivos.
      4) Quality tilt (alpha por régimen).
      5) Gate táctico: si macro_z < umbral, nuevas entradas = 0 (mantiene holdings).
      6) Caps (posición/beta) y normalización final.
    """
    # 1) Datos
    pnl = load_prices_panel(symbols + [bench], start, end, cache_key="pm_panel")
    bench_df = pnl.get(bench)
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return pd.DataFrame()

    bench_d = bench_df["close"].pct_change().dropna()
    bench_m = monthly_rets(bench_d)

    rows = []
    used = []
    for s in symbols:
        df = pnl.get(s)
        if df is None or df.empty or "close" not in df.columns:
            continue
        a_d = df["close"].pct_change().dropna()
        a_m = monthly_rets(a_d)
        common = a_m.index.intersection(bench_m.index)
        if len(common) < int(min_months):
            continue

        stt = kelly_stats(
            a_m.loc[common],
            bench_m.loc[common],
            winsor_p=winsor_p,
            t0=t0,
            min_months=min_months,
            lam_blend=lam_blend,
        )
        if not stt.get("valid", False):
            continue
        rows.append({"symbol": s, **stt})
        used.append(s)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 2) Peso base por Kelly (fraccionado)
    w_k = base_kelly * df["k_blend"].fillna(0.0).values
    if w_k.sum() <= 0:
        w_k = np.ones(len(df)) / len(df)
    else:
        w_k = w_k / w_k.sum()

    # 3) Macro → multiplicador + caps sugeridos
    reg = z_to_regime(float(macro_z))
    M_macro = reg.m_multiplier
    beta_cap_eff = min(float(beta_cap_user), float(reg.beta_cap))
    pos_cap_eff = min(float(pos_cap), float(reg.vol_cap))

    # 4) Quality tilt
    if quality_df is not None and not quality_df.empty:
        if "QualityScore" in quality_df.columns:
            qmap = dict(zip(quality_df["symbol"].astype(str).str.upper(), quality_df["QualityScore"]))
        elif "VFQ" in quality_df.columns:
            qmap = dict(zip(quality_df["symbol"].astype(str).str.upper(), quality_df["VFQ"]))
        else:
            qmap = {}
        q_series = df["symbol"].astype(str).str.upper().map(qmap)
        # fallback a mu mensual si no hay calidad
        q_series = q_series.fillna(df.get("mu", 0.0))
    else:
        q_series = df.get("mu", pd.Series(0.0, index=df.index))

    alpha = _pick_alpha(reg.label, alpha_off, alpha_neu, alpha_on)
    q_mult = _quality_tilt(q_series, alpha).reindex(df.index).fillna(1.0).values

    # 5) Gate táctico: bloquear NUEVAS si macro < umbral (mantener holdings)
    T_gate = np.ones(len(df))
    if current_holdings is not None and float(macro_z) < float(allow_new_when_z_below):
        holdset = {h.upper() for h in current_holdings}
        is_new = ~df["symbol"].astype(str).str.upper().isin(holdset)
        T_gate[is_new.values] = 0.0

    # 6) Combinar y caps finales
    w_base = w_k * M_macro * q_mult * T_gate
    if w_base.sum() <= 0:
        w_base = w_k
    else:
        w_base = w_base / w_base.sum()

    w_final = _finalize_weights(
        w_base,
        betas=df["beta"].fillna(1.0).values,
        beta_cap=beta_cap_eff,
        pos_cap=pos_cap_eff,
        enforce_sum1=enforce_sum1,
    )

    df = df.copy()
    df["weight"] = w_final
    df["beta_w"] = df["beta"].fillna(1.0) * df["weight"]
    return df.sort_values("weight", ascending=False).reset_index(drop=True)
