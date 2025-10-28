# --- poner esto ARRIBA DE TODO ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # o "none" si prefieres desactivar
# ---------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Tuple

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
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1rem 0; }
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(59,130,246,.08) !important; }
[data-testid="stCaptionContainer"] { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ============== IMPORTS DE TU PIPELINE ==============
from qvm_trend.scoring import (
    blend_breakout_qvm, build_momentum_proxy
)
from qvm_trend.data_io import (
    run_fmp_screener, filter_universe, load_prices_panel, load_benchmark,
    DEFAULT_START, DEFAULT_END
)
from qvm_trend.fundamentals import (
    download_fundamentals, build_vfq_scores_dynamic,
    download_guardrails_batch, apply_quality_guardrails
)
from qvm_trend.pipeline import (
    apply_trend_filter, enrich_with_breakout,
    market_regime_on
)
from qvm_trend.backtests import backtest_many

# NUEVOS IMPORTS (growth-aware)
from qvm_trend.factors_growth_aware import compute_qvm_scores, apply_megacap_rules

# ------------------ CACHÉ DE I/O ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_run_fmp_screener(limit: int) -> pd.DataFrame:
    return run_fmp_screener(limit=limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_guardrails(symbols: Tuple[str, ...], cache_key: str) -> pd.DataFrame:
    return download_guardrails_batch(list(symbols), cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_fundamentals(symbols: Tuple[str, ...], cache_key: str) -> pd.DataFrame:
    return download_fundamentals(list(symbols), cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_prices_panel(symbols, start, end, cache_key=""):
    return load_prices_panel(symbols, start, end, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_benchmark(bench, start, end):
    return load_benchmark(bench, start, end)

# ------------------ PERF HELPERS ------------------
def perf_summary_from_returns(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna().astype(float)
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year if periods_per_year else np.nan
    cagr = eq.iloc[-1]**(1/yrs) - 1 if yrs and yrs > 0 else np.nan
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

# ------------------ RANK HELPERS ------------------
def _probability_from_percentile(pct: pd.Series, beta: float = 6.0) -> pd.Series:
    s = pd.to_numeric(pct, errors="coerce").fillna(0.5).clip(0, 1)
    return 1.0 / (1.0 + np.exp(-beta * (s - 0.5)))

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Controles")
    preset = st.segmented_control("Preset", options=["Laxo", "Balanceado", "Estricto"], default="Balanceado")

    with st.expander("Universo & Screener", expanded=True):
        limit = st.slider("Límite del universo", 50, 1000, 300, 50)
        min_mcap = st.number_input("MarketCap mínimo (USD)", value=5e8, step=1e8, format="%.0f")
        ipo_days = st.slider("Antigüedad IPO (días)", 90, 1500, 365, 30)

    with st.expander("Fundamentales & Guardrails", expanded=False):
        min_cov_guard = st.slider("Cobertura VFQ mínima (# métricas)", 1, 4, 2)
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
        bench = st.selectbox("Benchmark", ["SPY", "QQQ", "^GSPC"], index=0)
        risk_on = st.toggle("Exigir mercado Risk-ON", value=True)
        start = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end = st.date_input("Fin", value=pd.to_datetime(DEFAULT_END).date())

    with st.expander("Ranking avanzado", expanded=False):
        beta_prob = st.slider("Sensibilidad probabilidad (β)", 1.0, 12.0, 6.0, 0.5)
        top_n_show = st.slider("Top N a resaltar", 10, 100, 25, 5)

    st.markdown("---")
    run_btn = st.button("Ejecutar", use_container_width=True)

# Presets (sin pisar cambios del usuario)
if preset == "Laxo":
    rvol_th = min(rvol_th, 1.0); closepos_th = min(closepos_th, 0.55); p52_th = min(p52_th, 0.92); min_hits = min(min_hits, 2)
elif preset == "Estricto":
    rvol_th = max(rvol_th, 1.5); closepos_th = max(closepos_th, 0.65); p52_th = max(p52_th, 0.97); min_hits = max(min_hits, 3)

# cache tag por corrida
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}"

# Estado del pipeline
if "pipeline_ready" not in st.session_state:
    st.session_state["pipeline_ready"] = False

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Universo", "Guardrails", "VFQ", "Señales", "QVM (growth-aware)", "Export", "Backtesting"]
)

# ==================== VFQ sidebar extra ====================
with st.sidebar:
    st.markdown("⚙️ Fundamentos (VFQ)")

    # Ajusta estas opciones a las columnas reales que produce tu DF de fundamentales
    value_metrics_opts = ["ev_ebitda", "fcf_yield", "pe_ttm", "pb"]
    quality_metrics_opts = ["roic", "roa", "gross_margin", "oper_margin"]

    sel_value = st.multiselect("Métricas Value", options=value_metrics_opts, default=["ev_ebitda", "fcf_yield"])
    sel_quality = st.multiselect("Métricas Quality", options=quality_metrics_opts, default=["roic", "gross_margin"])

    c1, c2 = st.columns(2)
    with c1: w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2: w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)

    method_intra = st.radio("Agregación intra-bloque", ["mean", "median", "weighted_mean"], index=0, horizontal=True)
    winsor_p = st.slider("Winsor p (cola)", 0.0, 0.10, 0.01, 0.005)
    size_buckets = st.slider("Buckets por tamaño", 1, 5, 3, 1)
    group_mode = st.selectbox("Agrupar por", ["sector", "sector|size"], index=1)
    min_cov = st.slider("Cobertura mín. (# métricas)", 0, 8, 1, 1)
    min_pct = st.slider("VFQ pct (intra-sector) mín.", 0.00, 1.00, 0.00, 0.01)

vfq_cfg = dict(
    value_metrics=sel_value,
    quality_metrics=sel_quality,
    w_value=float(w_value),
    w_quality=float(w_quality),
    method_intra=method_intra,
    winsor_p=float(winsor_p),
    size_buckets=int(size_buckets),
    group_mode=group_mode,
)

# ====== Paso 1: UNIVERSO ======
with tab1:
    st.subheader("Universo inicial")
    try:
        if run_btn:
            with st.status("Cargando universo del screener…", expanded=False) as status:
                uni_raw = _cached_run_fmp_screener(limit=limit)
                uni = filter_universe(uni_raw, min_mcap=min_mcap, ipo_min_days=ipo_days)
                status.update(label=f"Universo listo: {len(uni)} símbolos", state="complete")
            st.session_state["uni_raw"] = uni_raw
            st.session_state["uni"] = uni
            st.session_state["pipeline_ready"] = False
        elif "uni" in st.session_state:
            uni = st.session_state["uni"]
            uni_raw = st.session_state.get("uni_raw", pd.DataFrame())
        else:
            st.info("Presiona **Ejecutar** para cargar el universo.")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Screener", f"{len(st.session_state.get('uni_raw', pd.DataFrame())):,}")
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
        if run_btn and "uni" in st.session_state:
            uni = st.session_state["uni"]
            syms = uni["symbol"].dropna().astype(str).unique().tolist()
            with st.status("Descargando guardrails/fundamentales (cacheados)…", expanded=False) as status:
                df_guard = _cached_download_guardrails(tuple(sorted(syms)), cache_tag)
                kept, diag = apply_quality_guardrails(
                    df_guard,
                    require_profit_floor=(profit_hits > 0),
                    profit_floor_min_hits=profit_hits,
                    max_net_issuance=max_issuance,
                    max_asset_growth=max_assets,
                    max_accruals_ta=max_accr,
                    max_netdebt_ebitda=max_ndeb
                )
                status.update(label=f"Guardrails OK: {len(kept)} / {len(uni)}", state="complete")
            st.session_state["kept"] = kept
            st.session_state["guard_diag"] = diag
        elif "kept" in st.session_state:
            kept = st.session_state["kept"]
            uni = st.session_state["uni"]
            diag = st.session_state.get("guard_diag", pd.DataFrame())
        else:
            st.info("Primero ejecuta **Universo** (botón Ejecutar).")
            st.stop()

        c1, c2 = st.columns(2)
        c1.metric("Pasan guardrails", f"{len(kept):,}")
        c2.metric("Rechazados", f"{len(st.session_state['uni']) - len(kept):,}")

        st.dataframe(
            diag.merge(uni[["symbol", "sector"]], on="symbol", how="left"),
            use_container_width=True, hide_index=True
        )
        st.caption("Nota: si ves '__err_guard' o NaN, son símbolos con datos faltantes; quedan fuera.")
    except Exception as e:
        st.error(f"Error en guardrails: {e}")

# ====== Paso 3: VFQ ======
with tab3:
    st.subheader("VFQ")
    try:
        if run_btn and "kept" in st.session_state:
            uni = st.session_state["uni"]
            kept = st.session_state["kept"]
            kept_syms = kept["symbol"].dropna().astype(str).unique().tolist()
            with st.status("Descargando fundamentales VFQ (TTM)…", expanded=False) as status:
                df_fund = _cached_download_fundamentals(tuple(sorted(kept_syms)), cache_key=cache_tag)
                base_for_vfq = uni.merge(df_fund, on="symbol", how="right")

                # Calcula VFQ usando tu función dinámica
                df_vfq = build_vfq_scores_dynamic(
                    base_for_vfq,
                    value_metrics=vfq_cfg["value_metrics"],
                    quality_metrics=vfq_cfg["quality_metrics"],
                    w_value=vfq_cfg["w_value"],
                    w_quality=vfq_cfg["w_quality"],
                    method_intra=vfq_cfg["method_intra"],
                    winsor_p=vfq_cfg["winsor_p"],
                    size_buckets=vfq_cfg["size_buckets"],
                    group_mode=vfq_cfg["group_mode"],
                )
                status.update(label="VFQ calculado", state="complete")

            # -------- FIX 1: filtro de cobertura tolerante --------
            if "coverage_count" in df_vfq.columns:
                mask_cov = pd.to_numeric(df_vfq["coverage_count"], errors="coerce").fillna(0) >= int(min_cov)
            else:
                # Si no hay coverage_count, no bloquees por cobertura
                mask_cov = pd.Series(True, index=df_vfq.index)

            # -------- FIX 2: filtro por percentil dentro de sector --------
            # build_vfq_scores_dynamic produce 'VFQ' y 'VFQ_pct_sector'
            vfq_pct_col = "VFQ_pct_sector" if "VFQ_pct_sector" in df_vfq.columns else None
            if vfq_pct_col:
                mask_pct = pd.to_numeric(df_vfq[vfq_pct_col], errors="coerce").fillna(1.0) >= float(min_pct)
            else:
                mask_pct = pd.Series(True, index=df_vfq.index)

            df_vfq_sel = df_vfq.loc[mask_cov & mask_pct].copy()

            st.session_state["vfq"] = df_vfq
            st.session_state["vfq_sel"] = df_vfq_sel

        elif "vfq" in st.session_state and "vfq_sel" in st.session_state:
            df_vfq = st.session_state["vfq"]
            df_vfq_sel = st.session_state["vfq_sel"]
        else:
            st.info("Primero corre **Guardrails** (botón Ejecutar).")
            st.stop()

        # -------- FIX 3: ordenar por la columna correcta --------
        sort_col = "VFQ" if "VFQ" in df_vfq_sel.columns else (
            "VFQ_score" if "VFQ_score" in df_vfq_sel.columns else None
        )
        view_df = (
            df_vfq_sel.sort_values(sort_col, ascending=False).head(300)
            if sort_col else df_vfq_sel.head(300)
        )

        c1, c2 = st.columns([0.6, 0.4])
        with c1:
            st.metric("Con VFQ calculado", f"{len(df_vfq):,}")
            st.dataframe(view_df, use_container_width=True, hide_index=True)
        with c2:
            st.caption("Distribución por sector (seleccionados)")
            if "sector" in df_vfq_sel.columns and not df_vfq_sel.empty:
                st.bar_chart(df_vfq_sel["sector"].value_counts().head(15), use_container_width=True)

    except Exception as e:
        st.error(f"Error en VFQ: {e}")

# ====== Paso 4: SEÑALES (placeholder si tu lógica está en otro módulo) ======
with tab4:
    st.subheader("Señales (Técnico)")
    try:
        if run_btn and "vfq_sel" in st.session_state:
            # Carga precios + señales usando tus helpers reales
            syms = st.session_state["vfq_sel"]["symbol"].dropna().astype(str).unique().tolist()
            with st.status("Cargando precios y calculando señales…", expanded=False) as status:
                panel = _cached_load_prices_panel(syms, start=str(start), end=str(end), cache_key=cache_tag)
                # Aplica filtros/indicadores según tus funciones
                sig_df = apply_trend_filter(panel, use_and=use_and)
                sig_df = enrich_with_breakout(sig_df,
                                              rvol_th=rvol_th,
                                              closepos_th=closepos_th,
                                              p52_th=p52_th,
                                              updown_vol_th=updown_vol_th,
                                              min_hits=min_hits,
                                              atr_pct_min=atr_pct_min,
                                              use_rs_slope=use_rs_slope,
                                              require_breakout=require_breakout)
                if risk_on:
                    bench_df = _cached_load_benchmark(bench, start=str(start), end=str(end))
                    sig_df = market_regime_on(sig_df, bench_df)
                status.update(label="Señales listas", state="complete")

            st.session_state["signals"] = sig_df
            st.session_state["panel_prices"] = panel

        elif "signals" in st.session_state:
            sig_df = st.session_state["signals"]
        else:
            st.info("Corre **VFQ** y luego vuelve a esta pestaña.")
            st.stop()

        st.dataframe(sig_df.head(300), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error calculando señales: {e}")

# ====== Paso 5: QVM (growth-aware) ======
with tab5:
    st.subheader("QVM (growth-aware)")
    try:
        sig_df = st.session_state.get("signals", pd.DataFrame())
        vfq_df = st.session_state.get("vfq", pd.DataFrame())
        uni_df = st.session_state.get("uni", pd.DataFrame())
        kept_df = st.session_state.get("kept", pd.DataFrame())
        panel_prices = st.session_state.get("panel_prices")  # guardado en tab4

        if sig_df.empty:
            st.info("Primero corre **Señales**.")
            st.stop()

        # --- Base: columnas clave por symbol ---
        base_cols = []
        for c in ["symbol", "sector", "marketCap", "marketCap_unified", "BreakoutScore", "ClosePos", "P52", "rs_ma20_slope"]:
            if c in sig_df.columns:
                base_cols.append(c)

        base = sig_df[["symbol"] + [c for c in base_cols if c != "symbol"]].drop_duplicates("symbol")

        if isinstance(vfq_df, pd.DataFrame) and not vfq_df.empty:
            base = base.merge(vfq_df, on="symbol", how="left", suffixes=("", "_vfq"))
        if isinstance(uni_df, pd.DataFrame) and {"symbol", "sector", "marketCap"}.issubset(uni_df.columns):
            base = base.merge(uni_df[["symbol", "sector", "marketCap"]], on="symbol", how="left", suffixes=("", "_uni"))
        if isinstance(kept_df, pd.DataFrame) and "symbol" in kept_df.columns:
            base = base.merge(kept_df.drop_duplicates("symbol")[["symbol"]], on="symbol", how="right")

        mom_proxy = build_momentum_proxy(sig_df)
        if not mom_proxy.empty:
            base = base.merge(mom_proxy.rename("momentum_score"), on="symbol", how="left")

        # Normaliza market cap / sector
        if "market_cap" not in base.columns:
            if "marketCap_unified" in base.columns:
                base["market_cap"] = pd.to_numeric(base["marketCap_unified"], errors="coerce")
            else:
                base["market_cap"] = pd.to_numeric(base.get("marketCap"), errors="coerce")
        if "sector" not in base.columns and "sector_vfq" in base.columns:
            base["sector"] = base["sector_vfq"]

        # --- Momentum desde precios (si está el panel) ---
        momentum = None
        try:
            if isinstance(panel_prices, pd.DataFrame):
                df_long = panel_prices.rename(columns={"ticker": "symbol"} if "ticker" in panel_prices.columns else {})
                momentum = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
            elif isinstance(panel_prices, dict):
                frames = []
                for sym, dfp in panel_prices.items():
                    if isinstance(dfp, pd.DataFrame) and {"close"}.issubset(dfp.columns):
                        tmp = dfp.reset_index().rename(columns={"index": "date"} if "date" not in dfp.columns else {})
                        tmp["symbol"] = sym
                        frames.append(tmp[["symbol", "date", "close"]])
                if frames:
                    df_long = pd.concat(frames, ignore_index=True)
                    momentum = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
        except Exception:
            momentum = None

        if isinstance(momentum, pd.Series) and not momentum.empty:
            base = base.merge(momentum.rename("momentum_score_prices"), on="symbol", how="left")
            base["momentum_score"] = base[["momentum_score", "momentum_score_prices"]].mean(axis=1, skipna=True)
        else:
            if "momentum_score" not in base.columns:
                base["momentum_score"] = 0.0  # neutro

        # --- QVM growth-aware ---
        qvm_df = compute_qvm_scores(
            base.rename(columns={"marketCap": "market_cap"}),
            w_quality=0.40, w_value=0.25, w_momentum=0.35,
            momentum_col="momentum_score",
            sector_col="sector",
            mcap_col="market_cap"
        )

        # Reglas megacaps (opcionales)
        qvm_df = apply_megacap_rules(
            qvm_df,
            momentum_col="momentum_score",
            quality_col="quality_adj_neut",
            value_col="value_adj_neut"
        )

        # Blend con breakout (si está disponible). Percentil a "final_alpha".
        if "BreakoutScore" in base.columns:
            blend = blend_breakout_qvm(
                pd.DataFrame({
                    "qvm_score": qvm_df["qvm_score"],
                    "breakout_score": pd.to_numeric(base["BreakoutScore"], errors="coerce"),
                    "momentum_score": qvm_df.get("momentum_score")
                }),
                col_qvm="qvm_score",
                col_breakout="breakout_score",
                w_qvm=0.70,
                w_breakout=0.30,
                to_percentile=True
            )
            qvm_df["final_alpha"] = blend
        else:
            s = qvm_df["qvm_score"]
            qvm_df["final_alpha"] = s.rank(pct=True, method="average")

        if "final_alpha" in qvm_df.columns:
            pct = qvm_df["final_alpha"].rank(pct=True, method="average")
            qvm_df["final_alpha_pct"] = pct
            qvm_df["prob_up"] = _probability_from_percentile(pct, beta=beta_prob)

        st.metric("Con QVM calculado", f"{len(qvm_df):,}")
        show_cols = [c for c in [
            "symbol", "sector", "market_cap", "qvm_score", "final_alpha",
            "value_adj_neut", "quality_adj_neut", "mega_exception_ok",
            "final_alpha_pct", "prob_up", "quality_too_low",
            "BreakoutScore", "momentum_score"
        ] if c in qvm_df.columns]

        st.dataframe(
            qvm_df[show_cols].sort_values(["final_alpha", "qvm_score"], ascending=False).head(300),
            use_container_width=True, hide_index=True
        )

        if "prob_up" in qvm_df.columns:
            st.subheader(f"Top {top_n_show} por probabilidad de alza")
            top_cols = [c for c in show_cols if c in qvm_df.columns]
            st.dataframe(
                qvm_df.sort_values(["prob_up", "final_alpha"], ascending=False).head(top_n_show)[top_cols],
                use_container_width=True,
                hide_index=True
            )

        st.session_state["qvm"] = qvm_df
    except Exception as e:
        st.error(f"Error en QVM growth-aware: {e}")

# ====== Paso 6: EXPORT ======
with tab6:
    st.subheader("Exportar / Guardar ")
    uni_s  = st.session_state.get("uni")
    gdiag  = st.session_state.get("guard_diag")
    vfq_s  = st.session_state.get("vfq")
    sig_s  = st.session_state.get("signals")

    def _dl_btn(df, label, fname):
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.download_button(
                label,
                df.to_csv(index=False).encode(),
                file_name=fname,
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button(label, disabled=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _dl_btn(uni_s, "Descargar universo (CSV)", "universo.csv")
        _dl_btn(vfq_s, "Descargar VFQ (CSV)", "vfq.csv")
    with c2:
        _dl_btn(gdiag, "Descargar guardrails diag (CSV)", "guardrails_diag.csv")
        _dl_btn(sig_s, "Descargar señales (CSV)", "senales.csv")

# ====== Paso 7: BACKTESTING (placeholder) ======
with tab7:
    st.subheader("Backtesting")
    st.caption("Integra aquí tu lógica de backtests usando `backtest_many` si corresponde.")
