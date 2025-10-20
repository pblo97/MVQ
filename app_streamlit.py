# /mount/src/mvq/app_streamlit.py
from __future__ import annotations
import os
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st

# --- Imports del proyecto (dos rutas posibles) ---
try:
    from qvm_trend.data_io import (
        run_fmp_screener, filter_universe, load_prices_panel,
        load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END
    )
    from qvm_trend.fundamentals import (
        download_fundamentals, build_vfq_scores,
        download_guardrails_batch, apply_quality_guardrails
    )
    from qvm_trend.pipeline import apply_trend_filter, enrich_with_breakout
    from qvm_trend.scoring import DEFAULT_TH
    from qvm_trend.cache_io import save_df, load_df, save_panel, load_panel
except Exception:
    # Alternativa si tu paquete se llama distinto
    from data_io import (
        run_fmp_screener, filter_universe, load_prices_panel,
        load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END
    )
    from fundamentals import (
        download_fundamentals, build_vfq_scores,
        download_guardrails_batch, apply_quality_guardrails
    )
    from pipeline import apply_trend_filter, enrich_with_breakout
    from scoring import DEFAULT_TH
    from cache_io import save_df, load_df, save_panel, load_panel

st.set_page_config(page_title="VFQ + Tendencia + Breakouts", layout="wide")

# ======================== Sidebar (controles) ========================
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

# VFQ (no tienen sliders; se calcula y se usa % intra-sector)
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

# Cache / Acciones
st.sidebar.subheader("Acciones")
cache_tag = st.sidebar.text_input("Cache key", value="run1")
force_universe = st.sidebar.checkbox("Refrescar universo", value=False)
force_guard = st.sidebar.checkbox("Refrescar guardrails", value=False)
force_fund = st.sidebar.checkbox("Refrescar fundamentals", value=False)

# ======================== App title ========================
st.title("Screener VFQ + Tendencia + Breakouts")

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
st.header("2) Guardrails (calidad contable / disciplina)")
try:
    syms = uni["symbol"].tolist()
    df_guard = download_guardrails_batch(syms, cache_key=cache_tag, force=force_guard)
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
    st.dataframe(diag.sort_values("guard_all", ascending=False).head(50), width="stretch")
except Exception as e:
    st.error(f"Error en guardrails: {e}")
    st.stop()

# ======================== Paso 3: Fundamentals VFQ ========================
st.header("3) Fundamentals mínimos (VFQ) + Scores")
try:
    kept_syms = kept["symbol"].tolist()
    mc_map = uni.set_index("symbol")["marketCap"].to_dict() if "marketCap" in uni.columns else {}
    df_fund = download_fundamentals(kept_syms, market_caps=mc_map, cache_key=cache_tag, force=force_fund)

    base_for_vfq = uni[uni["symbol"].isin(kept_syms)].copy()
    df_vfq = build_vfq_scores(base_for_vfq, df_fund)

    # Top % intra-sector
    cutoff = df_vfq.groupby("sector")["VFQ"].transform(lambda s: s.quantile(1 - top_pct))
    df_vfq_sel = df_vfq[df_vfq["VFQ"] >= cutoff].copy()

    st.write(f"Elegibles por VFQ & cobertura: {len(df_vfq_sel)} "
             f"(Con cobertura ≥1 métrica VFQ: {(df_vfq['coverage_count']>0).sum()})")
    st.dataframe(df_vfq_sel.sort_values("VFQ", ascending=False).head(100), width="stretch")
except Exception as e:
    st.error(f"Error en VFQ: {e}")
    st.stop()

# ======================== Paso 4: Tendencia & Breakout ========================
st.header("4) Tendencia (MA200 / Mom 12–1) + Señal de Breakout")
try:
    syms_vfq = df_vfq_sel["symbol"].tolist()
    if len(syms_vfq) == 0:
        st.warning("No hay símbolos tras VFQ; relaja guardrails/VFQ o amplía universo.")
        st.stop()

    # Precios panel
    panel = load_prices_panel(syms_vfq, start.isoformat(), end.isoformat(), cache_key=cache_tag, force=False)
    bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())

    # Señales de tendencia
    trend = apply_trend_filter(panel, use_and_condition=use_and)  # debe devolver df con flags por símbolo/fecha o al menos flag actual
    # Breakout enrich (usa tus métricas: RVOL, ClosePos, P52, Up/Down Vol Ratio…)
    brk = enrich_with_breakout(
        panel,
        rvol_lookback=20,
        rvol_th=rvol_th,
        closepos_th=closepos_th,
        p52_th=p52_th,
        updown_vol_th=updown_vol_th,
        bench_series=bench_px["close"] if isinstance(bench_px, pd.DataFrame) and "close" in bench_px.columns else None
    )

    # Join señales a los scores
    df_sig = df_vfq_sel.merge(trend, on="symbol", how="left").merge(brk, on="symbol", how="left")

    # Señales de entrada (ejemplo: todas deben ser True)
    # Ajusta con tus nombres de columnas devueltos en apply_trend_filter / enrich_with_breakout
    trend_col = "signal_trend"          # <- asegúrate que pipeline.py exporta esta col
    breakout_col = "signal_breakout"    # <- idem
    if trend_col not in df_sig.columns:
        df_sig[trend_col] = False
    if breakout_col not in df_sig.columns:
        df_sig[breakout_col] = False

    df_sig["ENTRY"] = df_sig[trend_col].fillna(False) & df_sig[breakout_col].fillna(False)

    st.subheader("Candidatas (ENTRY = True)")
    st.dataframe(df_sig[df_sig["ENTRY"]].sort_values("VFQ", ascending=False), width="stretch")

    st.subheader("Diagnóstico de señales (top 100)")
    st.dataframe(df_sig.sort_values(["ENTRY", "VFQ"], ascending=[False, False]).head(100), width="stretch")

except Exception as e:
    st.error(f"Error en señales: {e}")
    st.stop()

# ======================== Paso 5 (opcional): Export ========================
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
