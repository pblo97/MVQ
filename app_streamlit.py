import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

# ==================== CONFIG BÁSICO ====================
st.set_page_config(
    page_title="Sistema QVM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS suave
st.markdown("""
<style>
/* contenedor */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
/* títulos */
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1rem 0; }
/* hover tablas */
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(59,130,246,.08) !important; }
/* captions más claras */
[data-testid="stCaptionContainer"] { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ============== IMPORTS DE TU PIPELINE (mantén tus módulos) ==============
from qvm_trend.data_io import (
    run_fmp_screener, filter_universe, load_prices_panel, load_benchmark,
    DEFAULT_START, DEFAULT_END
)
from qvm_trend.fundamentals import (
    download_fundamentals, build_vfq_scores,
    download_guardrails_batch, apply_quality_guardrails
)
from qvm_trend.pipeline import (
    apply_trend_filter, enrich_with_breakout,
    market_regime_on  # helper de régimen (SPY>MA200 & breadth)
)
# si tienes helpers de performance sin MonteCarlo:
def perf_summary_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna().astype(float)
    if r.empty: return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year if periods_per_year else np.nan
    cagr = eq.iloc[-1]**(1/yrs) - 1 if yrs and yrs>0 else np.nan
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

# ==================== HEADER ====================
l, r = st.columns([0.85, 0.15])
with l:
    st.markdown("<h1 style='margin-bottom:0'>QVM Screener</h1>", unsafe_allow_html=True)
    st.caption("Momentum estructural + Breakout técnico + Value/Quality (VFQ)")
with r:
    st.caption(datetime.now().strftime("Actualizado: %d %b %Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)

# ==================== SIDEBAR (pro con presets) ====================
with st.sidebar:
    st.markdown("### ⚙️ Controles")

    preset = st.segmented_control("Preset", options=["Laxo","Balanceado","Estricto"], default="Balanceado")

    with st.expander("Universo & Screener", expanded=True):
        limit = st.slider("Límite del universo", 50, 800, 300, 50)
        min_mcap = st.number_input("MarketCap mínimo (USD)", value=5e8, step=1e8, format="%.0f")
        ipo_days = st.slider("Antigüedad IPO (días)", 90, 1500, 365, 30)

    with st.expander("Fundamentales & Guardrails", expanded=False):
        min_cov = st.slider("Cobertura VFQ mínima (# métricas)", 1, 4, 2)
        profit_hits = st.slider("Pisos de rentabilidad (hits EBIT/CFO/FCF)", 0, 3, 2)
        max_issuance = st.slider("Net issuance máx.", 0.00, 0.10, 0.03, 0.01)
        max_assets = st.slider("Asset growth |y/y| máx.", 0.00, 0.50, 0.20, 0.01)
        max_accr = st.slider("Accruals/TA | | máx.", 0.00, 0.25, 0.10, 0.01)
        max_ndeb = st.slider("NetDebt/EBITDA máx.", 0.0, 6.0, 3.0, 0.5)

    with st.expander("Técnico — Tendencia & Breakout", expanded=True):
        use_and = st.toggle("MA200 Y Mom 12–1", value=False)
        require_breakout = st.toggle("Exigir Breakout para ENTRY", value=False)
        rvol_th = st.slider("RVOL (20d) mín.", 0.8, 2.5, 1.2, 0.1)
        closepos_th = st.slider("ClosePos mín.", 0.0, 1.0, 0.60, 0.05)
        p52_th = st.slider("Cercanía 52W High", 0.80, 1.00, 0.95, 0.01)
        updown_vol_th = st.slider("Up/Down Vol Ratio (20d)", 0.8, 3.0, 1.2, 0.1)
        min_hits = st.slider("Mínimo checks breakout (K de 4)", 1, 4, 3)
        atr_pct_min = st.slider("ATR pct (6–12m) mín.", 0.0, 1.0, 0.6, 0.05)
        use_rs_slope = st.toggle("Exigir RS slope > 0 (MA20)", value=False)

    with st.expander("Régimen & Fechas", expanded=False):
        bench = st.selectbox("Benchmark", ["SPY","QQQ","^GSPC"], index=0)
        risk_on = st.toggle("Exigir mercado Risk-ON", value=True)
        start = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end = st.date_input("Fin", value=pd.to_datetime(DEFAULT_END).date())

    st.markdown("---")
    run_btn = st.button("🚀 Ejecutar / Refrescar", use_container_width=True)

# Aplica presets en sliders (sin pisar lo que el usuario ya cambió)
if preset == "Laxo":
    rvol_th = min(rvol_th, 1.0); closepos_th = min(closepos_th, 0.55); p52_th = min(p52_th, 0.92); min_hits = min(min_hits, 2)
elif preset == "Estricto":
    rvol_th = max(rvol_th, 1.5); closepos_th = max(closepos_th, 0.65); p52_th = max(p52_th, 0.97); min_hits = max(min_hits, 3)

# cache tag por corrida
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}"

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Universo", "Guardrails", "VFQ", "Señales", "Export"]
)

# ====== Paso 1: UNIVERSO ======
with tab1:
    st.subheader("Universo inicial")
    try:
        with st.status("Cargando universo del screener…", expanded=False) as status:
            uni_raw = run_fmp_screener(limit=limit)
            uni = filter_universe(uni_raw, min_mcap=min_mcap, ipo_min_days=ipo_days)
            status.update(label=f"Universo listo: {len(uni)} símbolos", state="complete")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Screener", f"{len(uni_raw):,}")
        c2.metric("Tras filtros básicos", f"{len(uni):,}")
        if "sector" in uni.columns:
            st.bar_chart(uni["sector"].value_counts().head(12), use_container_width=True)
        st.dataframe(uni.head(200), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error cargando universo: {e}")

# ====== Paso 2: FUNDAMENTALES & GUARDRAILS ======
with tab2:
    st.subheader("Guardrails")
    try:
        syms = uni["symbol"].dropna().astype(str).unique().tolist()
        with st.status("Descargando guardrails/fundamentales (cacheados)…", expanded=False) as status:
            df_guard = download_guardrails_batch(syms, cache_key=cache_tag, force=False)
            kept, diag = apply_quality_guardrails(
                df_guard,
                require_profit_floor=(profit_hits>0),
                profit_floor_min_hits=profit_hits,
                max_net_issuance=max_issuance,
                max_asset_growth=max_assets,
                max_accruals_ta=max_accr,
                max_netdebt_ebitda=max_ndeb
            )
            status.update(label=f"Guardrails OK: {len(kept)} / {len(uni)}", state="complete")
        c1,c2 = st.columns(2)
        c1.metric("Pasan guardrails", f"{len(kept):,}")
        c2.metric("Rechazados", f"{len(uni)-len(kept):,}")
        st.dataframe(diag.merge(uni[["symbol","sector"]], on="symbol", how="left"), use_container_width=True, hide_index=True)
        st.caption("Nota: si ves columnas '__err_guard' o NaN, son símbolos con datos faltantes; quedan fuera.")
    except Exception as e:
        st.error(f"Error en guardrails: {e}")

with tab3:
    st.subheader("3) Ranking VFQ")
    try:
        kept_syms = kept["symbol"].dropna().astype(str).unique().tolist()
        with st.status("Descargando fundamentales VFQ (TTM)…", expanded=False) as status:
            df_fund = download_fundamentals(kept_syms, cache_key=cache_tag, force=False)
            base_for_vfq = uni.merge(df_fund, on="symbol", how="right")  # right asegura que no pierdas símbolos con fundas
            df_vfq = build_vfq_scores(base_for_vfq, base_for_vfq)
            status.update(label="VFQ calculado", state="complete")

        # diagnóstico cobertura
        vfq_fields = [c for c in ["fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"] if c in df_vfq.columns]
        st.caption("Cobertura por métrica (no nulos)")
        if vfq_fields:
            st.bar_chart(df_vfq[vfq_fields].notna().sum().sort_values(ascending=False), use_container_width=True)
        else:
            st.info("No llegó ninguna métrica VFQ: revisa la API key/ratelimit o los nombres mapeados en download_fundamentals.")

        # filtro por cobertura mínima
        df_vfq_sel = df_vfq[df_vfq.get("coverage_count", 0) >= int(min_cov)].copy()

        st.metric("VFQ elegibles", f"{len(df_vfq_sel):,}")
        cols_show = [c for c in ["symbol","sector","marketCap_unified","coverage_count","VFQ","ValueScore","QualityScore","fcf_yield","inv_ev_ebitda","gross_profitability","netMargin"] if c in df_vfq_sel.columns]
        st.dataframe(
            df_vfq_sel[cols_show].sort_values(["VFQ","ValueScore","QualityScore"], ascending=False).head(300),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"Error en VFQ: {e}")



# ====== Paso 4: SEÑALES (Tendencia & Breakout) ======
with tab4:
    st.subheader("Tendencia & Rompimiento")
    try:
        syms_vfq = df_vfq_sel["symbol"].dropna().astype(str).tolist()
        if len(syms_vfq) == 0:
            st.warning("Sin símbolos tras VFQ; relaja filtros o amplía universo.")
        else:
            # Precios y benchmark
            panel = load_prices_panel(syms_vfq, start.isoformat(), end.isoformat(), cache_key=cache_tag, force=False)
            bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())

            # Señales
            trend = apply_trend_filter(panel, use_and_condition=use_and)
            brk = enrich_with_breakout(
                panel,
                rvol_lookback=20,
                rvol_th=rvol_th,
                closepos_th=closepos_th,
                p52_th=p52_th,
                updown_vol_th=updown_vol_th,
                bench_series=bench_px["close"] if isinstance(bench_px, pd.DataFrame) and "close" in bench_px.columns else None,
                **({"min_hits": min_hits} if "min_hits" in enrich_with_breakout.__code__.co_varnames else {}),
                **({"use_rs_slope": use_rs_slope, "rs_min_slope": 0.0} if "use_rs_slope" in enrich_with_breakout.__code__.co_varnames else {}),
            )

            # Merge base
            base_cols = [c for c in ["symbol","sector","marketCap","VFQ","ValueScore","QualityScore","coverage_count"] if c in df_vfq.columns]
            base_for_signals = df_vfq[base_cols].drop_duplicates("symbol") if base_cols else df_vfq[["symbol"]].drop_duplicates()

            df_sig = (
                base_for_signals
                .merge(trend if isinstance(trend, pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_trend"]), on="symbol", how="left")
                .merge(brk if isinstance(brk, pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_breakout"]), on="symbol", how="left")
            )

            # columnas obligatorias
            for col in ("signal_trend","signal_breakout"):
                if col not in df_sig.columns: df_sig[col] = False
                df_sig[col] = df_sig[col].fillna(False).astype(bool)
            # columnas de diagnóstico
            for c in ["RVOL20","ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope","BreakoutScore","hits","c_RVOL","c_ClosePos","c_P52","c_UDVol","c_RSslope"]:
                if c not in df_sig.columns: df_sig[c] = np.nan

            # ENTRY
            df_sig["ENTRY"] = (df_sig["signal_trend"] & df_sig["signal_breakout"]) if require_breakout else df_sig["signal_trend"]
            # gating por régimen
            if risk_on and not market_regime_on(bench_px, panel, ma_bench=200, breadth_ma=50, breadth_min=0.5):
                st.warning("Régimen OFF (bench ≤ MA200 o breadth ≤ 50%): bloqueando nuevas entradas.")
                df_sig["ENTRY"] = False

            # Conteos
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("En tendencia", f"{int(df_sig['signal_trend'].sum())}")
            k2.metric("Breakout", f"{int(df_sig['signal_breakout'].sum())}")
            k3.metric("ENTRY", f"{int(df_sig['ENTRY'].sum())}")
            k4.metric("VFQ elegibles", f"{len(df_vfq_sel):,}")

            # Diagnóstico muestra
            dbg_cols = [c for c in ["symbol","RVOL20","ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope","hits","signal_trend","signal_breakout","ENTRY"] if c in df_sig.columns]
            st.caption("Diagnóstico (muestra)")
            st.dataframe(
                df_sig[dbg_cols].sort_values(["ENTRY","signal_breakout","hits","RVOL20","ClosePos","P52","UDVol20"], ascending=[False,False,False,False,False,False,False]).head(120),
                use_container_width=True, hide_index=True
            )

            # Candidatas ordenadas pro
            st.subheader("Candidatas (ENTRY = True)")
            df_candidates = df_sig.loc[df_sig["ENTRY"]].copy()
            sort_cols = [c for c in ["BreakoutScore","VFQ","ValueScore","QualityScore"] if c in df_candidates.columns]
            asc = [False] * len(sort_cols) if sort_cols else [False]
            st.dataframe(
                df_candidates.sort_values(sort_cols, ascending=asc),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "BreakoutScore": st.column_config.NumberColumn("BreakoutScore", help="0–100", format="%.1f"),
                    "VFQ": st.column_config.NumberColumn("VFQ", format="%.1f"),
                    "ValueScore": st.column_config.NumberColumn("Value", format="%.1f"),
                    "QualityScore": st.column_config.NumberColumn("Quality", format="%.1f"),
                    "ClosePos": st.column_config.NumberColumn("ClosePos", format="%.2f"),
                    "P52": st.column_config.NumberColumn("P52", format="%.3f"),
                    "UDVol20": st.column_config.NumberColumn("UDVol20", format="%.2f"),
                    "ATR_pct": st.column_config.ProgressColumn("ATR pct", format="%.0f%%", min_value=0, max_value=1),
                }
            )

            # guarda en session para Export
            st.session_state["uni"] = uni
            st.session_state["guard_diag"] = diag
            st.session_state["vfq"] = df_vfq
            st.session_state["signals"] = df_sig

    except Exception as e:
        st.error(f"Error en señales: {e}")

# ====== Paso 5: EXPORT ======
with tab5:
    st.subheader("Exportar / Guardar ")
    uni_s  = st.session_state.get("uni")
    gdiag  = st.session_state.get("guard_diag")
    vfq_s  = st.session_state.get("vfq")
    sig_s  = st.session_state.get("signals")

    def _dl_btn(df, label, fname):
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.download_button(label, df.to_csv(index=False).encode(), file_name=fname, mime="text/csv", use_container_width=True)
        else:
            st.button(label, disabled=True, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        _dl_btn(uni_s, "Descargar universo (CSV)", "universo.csv")
        _dl_btn(vfq_s, "Descargar VFQ (CSV)", "vfq.csv")
    with c2:
        _dl_btn(gdiag, "Descargar guardrails diag (CSV)", "guardrails_diag.csv")
        _dl_btn(sig_s, "Descargar señales (CSV)", "senales.csv")
