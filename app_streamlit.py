# /mount/src/mvq/app_streamlit.py
from __future__ import annotations
import os
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st

# ------------------ Imports del proyecto ------------------
try:
    from qvm_trend.data_io import (
        run_fmp_screener, filter_universe, load_prices_panel,
        load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END, fmp_probe
    )
except Exception:
    # fallback por si el paquete no está
    from data_io import (
        run_fmp_screener, filter_universe, load_prices_panel,
        load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END, fmp_probe
    )

try:
    from qvm_trend.fundamentals import (
        download_fundamentals, build_vfq_scores,
        download_guardrails_batch, apply_quality_guardrails
    )
except Exception:
    from fundamentals import (
        download_fundamentals, build_vfq_scores,
        download_guardrails_batch, apply_quality_guardrails
    )

try:
    from qvm_trend.pipeline import apply_trend_filter, enrich_with_breakout
except Exception:
    from pipeline import apply_trend_filter, enrich_with_breakout

try:
    from qvm_trend.scoring import DEFAULT_TH
except Exception:
    DEFAULT_TH = {}

try:
    from qvm_trend.cache_io import save_df, load_df, save_panel, load_panel
except Exception:
    # no-ops si no existe cache_io
    def save_df(*args, **kwargs): ...
    def load_df(*args, **kwargs): return None
    def save_panel(*args, **kwargs): ...
    def load_panel(*args, **kwargs): return None

# ------------------ Config de página ------------------
st.set_page_config(page_title="VFQ + Tendencia + Breakouts", layout="wide")
st.title("Screener VFQ + Tendencia + Breakouts")

# ======================== Sidebar ========================
st.sidebar.header("Parámetros")

# Universo (Screener)
limit = st.sidebar.slider("Límite screener (símbolos)", 50, 500, 300, step=50)
min_mcap = float(st.sidebar.number_input("Market Cap mínimo (USD)", 1e7, 1e12, value=5e8, step=1e8, format="%.0f"))
ipo_days = st.sidebar.slider("Antigüedad IPO mínima (días)", 0, 2000, 365, step=30)

# Guardrails
st.sidebar.subheader("Guardrails (calidad)")
profit_floor_hits = st.sidebar.slider("Pisos de rentabilidad (hits de {EBIT, CFO, FCF})", 0, 3, 2, step=1)
max_issuance = st.sidebar.slider("Dilución máx. (Net issuance)", 0.0, 0.20, 0.03, step=0.01)
max_asset_g = st.sidebar.slider("Asset growth máx. (abs)", 0.0, 1.0, 0.20, step=0.05)
max_accruals = st.sidebar.slider("Accruals/TA máx. (abs)", 0.0, 0.5, 0.10, step=0.01)
max_ndeb = st.sidebar.slider("Net Debt / EBITDA máx.", 0.0, 10.0, 3.0, step=0.5)

# VFQ
st.sidebar.subheader("VFQ")
top_pct = st.sidebar.slider("Top % intra-sector (VFQ)", 0.05, 1.0, 0.35, step=0.05)

# Tendencia + Breakout
st.sidebar.subheader("Tendencia / Señales")
start = st.sidebar.date_input("Inicio precios", value=pd.to_datetime(DEFAULT_START).date())
end = st.sidebar.date_input("Fin precios", value=pd.to_datetime(DEFAULT_END).date())
use_and = st.sidebar.checkbox("Requerir (MA200 AND Mom 12–1>0)", value=False)
bench = st.sidebar.text_input("Benchmark RS (SPY/QQQ/IPSA)", value="SPY")

# Breakout thresholds (mínimo potente)
st.sidebar.subheader("Breakout (mínimo potente)")
rvol_th = st.sidebar.slider("RVOL 20d (umbral)", 1.0, 3.0, 1.5, step=0.1)
closepos_th = st.sidebar.slider("ClosePos (0–1)", 0.0, 1.0, 0.6, step=0.05)
p52_th = st.sidebar.slider("P52 (proximidad a 52W high)", 0.80, 1.05, 0.95, step=0.01)
updown_vol_th = st.sidebar.slider("Up/Down Vol Ratio (20d)", 0.5, 3.0, 1.2, step=0.1)
min_hits = st.sidebar.slider("Breakout: mín. checks verdaderos (0-5)", 1, 5, 3, step=1)
use_rs_slope = st.sidebar.checkbox("Usar RS slope > 0 como check extra", value=False)
# Cache / Acciones
st.sidebar.subheader("Cache / Acciones")
cache_tag = st.sidebar.text_input("Cache key", value="run1")
force_universe = st.sidebar.checkbox("Refrescar universo", value=False)
force_guard = st.sidebar.checkbox("Refrescar guardrails", value=False)
force_fund = st.sidebar.checkbox("Refrescar fundamentals", value=False)

# ======================== Paso 0: Sonda FMP ========================
st.header("0) Sonda FMP")
try:
    probe = fmp_probe("AAPL")
    st.json(probe)
    if not (probe.get("key_metrics_ttm_ok") or probe.get("ratios_ttm_ok")):
        st.warning("La API FMP no está respondiendo para TTM. Verifica API key/cuota. "
                   "Usa 'Refrescar' o cambia Cache key.")
except Exception as e:
    st.info(f"No se pudo ejecutar sonda FMP: {e}")

# ======================== Paso 1: Universo ========================
st.header("1) Universo (Screener FMP → Limpieza)")
try:
    uni_raw = run_fmp_screener(limit=limit)
    uni = filter_universe(uni_raw, min_mcap=min_mcap, ipo_min_days=ipo_days)
    st.write(f"Universo limpio: {len(uni)} símbolos")
    st.dataframe(uni.head(50), width="stretch")
except Exception as e:
    st.error(f"Error en screener: {e}")
    st.stop()

# ======================== Paso 2: Guardrails ========================
st.header("Guardrails (calidad contable / disciplina)")
try:
    syms = uni["symbol"].tolist()
    df_guard = download_guardrails_batch(syms, cache_key=cache_tag, force=force_guard)

    # Diagnóstico de errores de guardrails
    if "__err_guard" in df_guard.columns:
        st.warning("Algunas descargas de guardrails fallaron; mostrando columna __err_guard")
        bad = df_guard[df_guard["__err_guard"].notna()][["symbol", "__err_guard"]]
        if not bad.empty:
            st.dataframe(bad.head(50), width="stretch")

    df_merge = uni.merge(df_guard, on="symbol", how="left")
    kept, diag = apply_quality_guardrails(
        df_merge,
        require_profit_floor=(profit_floor_hits > 0),
        profit_floor_min_hits=profit_floor_hits,
        max_net_issuance=max_issuance,
        max_asset_growth=max_asset_g,
        max_accruals_ta=max_accruals,
        max_netdebt_ebitda=max_ndeb
    )
    st.write(f"Tras guardrails: {len(kept)} / {len(uni)}")
    st.dataframe(diag.sort_values("guard_all", ascending=False).head(100), width="stretch")
except Exception as e:
    st.error(f"Error en guardrails: {e}")
    st.stop()



# ======================== Paso 3: Fundamentals VFQ ========================
st.header("Fundamentals mínimos (VFQ) + Scores")
try:
    kept_syms = kept["symbol"].tolist()
    if len(kept_syms) == 0:
        st.warning("No hay símbolos tras guardrails. Relaja umbrales o aumenta el universo.")
        st.stop()

    mc_map = uni.set_index("symbol")["marketCap"].to_dict() if "marketCap" in uni.columns else {}
    df_fund = download_fundamentals(kept_syms, market_caps=mc_map, cache_key=cache_tag, force=force_fund)

    # Diagnóstico de cobertura por columna
    st.caption("Cobertura (no nulos) fundamentals")
    try:
        cols_cov = ["evToEbitda","fcf_ttm","cfo_ttm","ebit_ttm","grossProfitTTM","totalAssetsTTM","roic","roa","netMargin"]
        cov = df_fund.reindex(columns=[c for c in cols_cov if c in df_fund.columns]).notna().sum().sort_values(ascending=False)
        st.write(cov.to_frame("count").T)
    except Exception:
        pass

    # Muestra ejemplos con datos (si hay)
    non_null_any = df_fund.dropna(how="all", subset=[c for c in cols_cov if c in df_fund.columns])
    if not non_null_any.empty:
        st.dataframe(non_null_any.head(10), width="stretch")
    else:
        st.warning("⚠️ Aún sin cobertura en fundamentals. Prueba con Cache key nuevo y activa 'Refrescar fundamentals'. "
                "Reduce el tamaño del screener (100–150) para evitar 429.")

    # Diagnóstico de errores de fundamentals
    if "__err_fund" in df_fund.columns:
        st.warning("Algunas descargas de fundamentals fallaron; mostrando columna __err_fund")
        badf = df_fund[df_fund["__err_fund"].notna()][["symbol", "__err_fund"]]
        if not badf.empty:
            st.dataframe(badf.head(50), width="stretch")

    base_for_vfq = uni[uni["symbol"].isin(kept_syms)].copy()
    df_vfq = build_vfq_scores(base_for_vfq, df_fund)

    # Top % intra-sector
    if "VFQ" in df_vfq.columns and "sector" in df_vfq.columns:
        cutoff = df_vfq.groupby("sector")["VFQ"].transform(lambda s: s.quantile(1 - top_pct))
        df_vfq_sel = df_vfq[df_vfq["VFQ"] >= cutoff].copy()
    else:
        df_vfq_sel = df_vfq.copy()

    st.write(
        f"Elegibles por VFQ & cobertura: {len(df_vfq_sel)}  "
        f"(Con cobertura ≥1 métrica VFQ: {(df_vfq.get('coverage_count', pd.Series(0)).gt(0)).sum()})"
    )
    st.dataframe(df_vfq_sel.sort_values("VFQ", ascending=False).head(100), width="stretch")
except Exception as e:
    st.error(f"Error en VFQ: {e}")
    st.stop()


# Cobertura por columna (rápido)
st.caption("Cobertura (no nulos) fundamentals")
try:
    cov = df_fund[["evToEbitda","fcf_ttm","cfo_ttm","ebit_ttm","grossProfitTTM","totalAssetsTTM","roic","roa","netMargin"]].notna().sum().sort_values(ascending=False)
    st.write(cov.to_frame("count").T)
except Exception:
    pass

# Mostrar filas con error si existieran
if "__err_fund" in df_fund.columns:
    badf = df_fund[df_fund["__err_fund"].notna()][["symbol","__err_fund"]]
    if not badf.empty:
        st.warning("Errores al descargar fundamentals (primeros):")
        st.dataframe(badf.head(50), width="stretch")


# ======================== Paso 4: Tendencia & Breakout ========================
st.header("4) Tendencia (MA200 / Mom 12–1) + Señal de Breakout")

try:
    syms_vfq = df_vfq_sel["symbol"].dropna().astype(str).tolist()
    if len(syms_vfq) == 0:
        st.warning("No hay símbolos tras VFQ; relaja guardrails/VFQ o amplía universo.")
        st.stop()

    # --- Carga de precios y benchmark
    panel = load_prices_panel(
        syms_vfq,
        start.isoformat(),
        end.isoformat(),
        cache_key=cache_tag,
        force=False
    )
    bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())

    # --- Señales de tendencia
    trend = apply_trend_filter(panel, use_and_condition=use_and)  # devuelve 'symbol','signal_trend', métricas

    # --- Señales de breakout
    brk = enrich_with_breakout(
        panel,
        rvol_lookback=20,
        rvol_th=rvol_th,
        closepos_th=closepos_th,
        p52_th=p52_th,
        updown_vol_th=updown_vol_th,
        bench_series=bench_px["close"] if isinstance(bench_px, pd.DataFrame) and "close" in bench_px.columns else None,
        # estos dos son opcionales; inclúyelos solo si tu enrich_with_breakout ya los acepta
        **({"min_hits": min_hits} if "min_hits" in enrich_with_breakout.__code__.co_varnames else {}),
        **({"use_rs_slope": use_rs_slope, "rs_min_slope": 0.0} if "use_rs_slope" in enrich_with_breakout.__code__.co_varnames else {}),
    )

    # --- Base para señales (VFQ + metadata)
    base_cols = [c for c in ["symbol","sector","marketCap","VFQ","ValueScore","QualityScore","coverage_count"] if c in df_vfq.columns]
    base_for_signals = df_vfq[base_cols].drop_duplicates("symbol") if base_cols else df_vfq[["symbol"]].drop_duplicates()

    # --- Merge único (NO volver a crear df_sig más abajo)
    df_sig = (
        base_for_signals
        .merge(trend if isinstance(trend, pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_trend"]), on="symbol", how="left")
        .merge(brk   if isinstance(brk,   pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_breakout"]), on="symbol", how="left")
    )

    # --- Garantiza columnas booleanas y diagnósticas
    for col in ("signal_trend","signal_breakout"):
        if col not in df_sig.columns: df_sig[col] = False
        df_sig[col] = df_sig[col].fillna(False).astype(bool)

    for c in ["RVOL20","ClosePos","P52","UDVol20","rs_ma20_slope","c_RVOL","c_ClosePos","c_P52","c_UDVol","c_RSslope","hits"]:
        if c not in df_sig.columns: df_sig[c] = np.nan

    # --- Regla de entrada (ENTRY) siempre creada antes de usarla
    require_breakout = st.sidebar.checkbox("Exigir breakout para ENTRY", value=False)
    df_sig["ENTRY"] = (df_sig["signal_trend"] & df_sig["signal_breakout"]) if require_breakout else df_sig["signal_trend"]

    # --- Conteo y diagnóstico
    st.caption("Conteo señales por etapa")
    st.write({
        "n_total": int(len(df_sig)),
        "trend_true": int(df_sig["signal_trend"].sum()),
        "breakout_true": int(df_sig["signal_breakout"].sum()),
        "entry_true": int(df_sig["ENTRY"].sum()),
    })

    dbg_cols = ["symbol","RVOL20","ClosePos","P52","UDVol20","rs_ma20_slope",
                "c_RVOL","c_ClosePos","c_P52","c_UDVol","c_RSslope","hits",
                "signal_trend","signal_breakout","ENTRY"]
    dbg_cols = [c for c in dbg_cols if c in df_sig.columns]

    st.subheader("Diagnóstico de métricas de breakout (muestra)")
    st.dataframe(
        df_sig[dbg_cols].sort_values(
            ["ENTRY","signal_breakout","hits","RVOL20","ClosePos","P52","UDVol20"],
            ascending=[False, False, False, False, False, False, False]
        ).head(100),
        width="stretch"
    )

    # --- Tabla de candidatas
    st.subheader("Candidatas (ENTRY = True)")

    # filtramos primero por ENTRY
    df_candidates = df_sig.loc[df_sig["ENTRY"]].copy()

    # columnas para ordenar (las que existan)
    sort_cols = [c for c in ["BreakoutScore", "VFQ", "ValueScore", "QualityScore"] if c in df_candidates.columns]
    if not sort_cols:
        # fallback por si faltan scores (no rompe)
        sort_cols = [c for c in ["VFQ", "ValueScore", "QualityScore"] if c in df_candidates.columns]

    # orden descendente para todas
    asc = [False] * len(sort_cols)

    st.dataframe(
        df_candidates.sort_values(sort_cols, ascending=asc),
        width="stretch"   # si Streamlit te advierte, cambia a width="content" o "stretch" según tu versión
    )


except Exception as e:
    st.error(f"Error en señales: {e}")
    st.stop()

# ======================== Paso 5: Export ========================
st.header("5) Export / Guardar corrida")
if st.button("Guardar tablas (cache_io)"):
    try:
        save_df(uni, f"uni_{cache_tag}")
        save_df(diag, f"guard_diag_{cache_tag}")
        save_df(df_vfq, f"vfq_{cache_tag}")
        save_df(df_sig, f"signals_{cache_tag}")
        st.success("Tablas guardadas en cache_io.")
    except Exception as e:
        st.error(f"No se pudo guardar: {e}")
 #======================== Paso 6: Resumen ========================
def perf_summary_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna().astype(float)
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year
    cagr = eq.iloc[-1]**(1/yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    sharpe = (r.mean()*periods_per_year) / r.std() if r.std() > 0 else np.nan
    dd = eq/eq.cummax() - 1
    maxdd = dd.min()
    hit = (r > 0).mean()
    avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
    payoff = (avg_win/abs(avg_loss)) if (avg_win and avg_loss) else np.nan
    expct = (hit*avg_win + (1-hit)*avg_loss) if (not np.isnan(hit) and avg_win is not None and avg_loss is not None) else np.nan
    return {
        "CAGR": float(cagr), "Vol_anual": float(vol), "Sharpe": float(sharpe),
        "MaxDD": float(maxdd), "HitRate": float(hit), "AvgWin": float(avg_win),
        "AvgLoss": float(avg_loss), "Payoff": float(payoff), "Expectancy": float(expct),
        "Periodos": int(len(r))
    }
st.header("5) Resumen de performance (sin Monte Carlo)")
rets_any = None
for cand in ["rets2_m","rets2_q","rets"]:
    if cand in locals() and isinstance(locals()[cand], pd.Series) and len(locals()[cand])>0:
        rets_any = locals()[cand].dropna().astype(float); break

if rets_any is None or len(rets_any) < 6:
    st.info("Necesitas un backtest con al menos ~6 periodos para ver el resumen.")
else:
    k = 12 if rets_any.index.freqstr in ("M","ME") or len(rets_any)>30 else 4  # heurística
    summ = perf_summary_from_returns(rets_any, periods_per_year=k)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("CAGR", f"{summ['CAGR']*100:.2f}%")
        st.metric("Sharpe", f"{summ['Sharpe']:.2f}")
        st.metric("Hit Rate", f"{summ['HitRate']*100:.1f}%")
    with c2:
        st.metric("Max Drawdown", f"{summ['MaxDD']*100:.1f}%")
        st.metric("Payoff", f"{summ['Payoff']:.2f}")
        st.metric("Expectancy", f"{summ['Expectancy']*100:.2f}%")
    with c3:
        st.metric("Vol anual", f"{summ['Vol_anual']*100:.1f}%")
        st.metric("Periodos", f"{summ['Periodos']}")