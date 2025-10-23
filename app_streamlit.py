import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

# ==== QVM / VFQ ====
from qvm_trend.fundamentals import build_vfq_scores_dynamic

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
    market_regime_on
)
from qvm_trend.backtests import backtest_many
from qvm_trend.stats import beta_vs_bench, win_loss_stats, expectancy

# ------------------ PERSISTENCIA DE CARTERA ------------------
PORTFOLIO_PATH = "portfolio_symbols.csv"

def _portfolio_empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "date_added"]).astype({"symbol":"string","date_added":"string"})

def load_portfolio_df() -> pd.DataFrame:
    try:
        if os.path.exists(PORTFOLIO_PATH):
            df = pd.read_csv(PORTFOLIO_PATH)
            if "symbol" not in df.columns:
                return _portfolio_empty_df()
            df["symbol"] = df["symbol"].astype(str).str.upper()
            if "date_added" not in df.columns:
                df["date_added"] = datetime.now().strftime("%Y-%m-%d")
            return df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        else:
            return _portfolio_empty_df()
    except Exception:
        return _portfolio_empty_df()

def save_portfolio_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        _portfolio_empty_df().to_csv(PORTFOLIO_PATH, index=False)
    else:
        out = df.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        if "date_added" not in out.columns:
            out["date_added"] = datetime.now().strftime("%Y-%m-%d")
        out.drop_duplicates(subset=["symbol"]).to_csv(PORTFOLIO_PATH, index=False)

def add_symbols_to_portfolio(symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return load_portfolio_df()
    df = load_portfolio_df()
    add = pd.DataFrame({
        "symbol": [s.strip().upper() for s in symbols if s and str(s).strip()],
        "date_added": datetime.now().strftime("%Y-%m-%d")
    })
    merged = pd.concat([df, add], ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    save_portfolio_df(merged)
    return merged

def remove_symbols_from_portfolio(symbols: list[str]) -> pd.DataFrame:
    df = load_portfolio_df()
    if df.empty or not symbols:
        return df
    keep = df[~df["symbol"].isin([s.strip().upper() for s in symbols])]
    save_portfolio_df(keep)
    return keep

# ------------------ CACHÉ DE I/O (NUEVO) ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_prices_panel(symbols, start, end, cache_key=""):
    return load_prices_panel(symbols, start, end, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_benchmark(bench, start, end):
    return load_benchmark(bench, start, end)

# ------------------ PERF HELPERS ------------------
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

# ==================== SIDEBAR ====================
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
    run_btn = st.button("Ejecutar", use_container_width=True)

# Aplica presets (sin pisar cambios del usuario)
if preset == "Laxo":
    rvol_th = min(rvol_th, 1.0); closepos_th = min(closepos_th, 0.55); p52_th = min(p52_th, 0.92); min_hits = min(min_hits, 2)
elif preset == "Estricto":
    rvol_th = max(rvol_th, 1.5); closepos_th = max(closepos_th, 0.65); p52_th = max(p52_th, 0.97); min_hits = max(min_hits, 3)

# cache tag por corrida
cache_tag = f"{int(min_mcap)}_{ipo_days}_{limit}"

# Estado del pipeline (para evitar recomputar en cada rerun)
if "pipeline_ready" not in st.session_state:
    st.session_state["pipeline_ready"] = False

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Universo", "Guardrails", "VFQ", "Señales", "Export", "Backtesting", "Gestión de cartera"]
)

# ==================== VFQ sidebar extra ====================
with st.sidebar:
    st.markdown("⚙️ Fundamentos (VFQ)")
    sel_value = st.multiselect(
        "Value metrics (↑ mejor)",
        ["fcf_yield", "inv_ev_ebitda", "earnings_yield", "shareholder_yield"],
        default=["fcf_yield", "inv_ev_ebitda"]
    )
    sel_quality = st.multiselect(
        "Quality metrics (↑ mejor)",
        ["gross_profitability", "roic", "roa", "netMargin"],
        default=["gross_profitability", "roic", "roa", "netMargin"]
    )
    c1, c2 = st.columns(2)
    with c1:  w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2:  w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)
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
                uni_raw = run_fmp_screener(limit=limit)
                uni = filter_universe(uni_raw, min_mcap=min_mcap, ipo_min_days=ipo_days)
                status.update(label=f"Universo listo: {len(uni)} símbolos", state="complete")
            st.session_state["uni_raw"] = uni_raw
            st.session_state["uni"] = uni
            st.session_state["pipeline_ready"] = False  # aún falta seguir pasos
        elif "uni" in st.session_state:
            uni = st.session_state["uni"]
            uni_raw = st.session_state.get("uni_raw", pd.DataFrame())
        else:
            st.info("Presiona **Ejecutar** para cargar el universo.")
            st.stop()

        c1,c2,c3,c4 = st.columns(4)
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
            st.session_state["kept"] = kept
            st.session_state["guard_diag"] = diag
        elif "kept" in st.session_state:
            kept = st.session_state["kept"]
            uni = st.session_state["uni"]
            diag = st.session_state.get("guard_diag", pd.DataFrame())
        else:
            st.info("Primero ejecuta **Universo** (botón Ejecutar).")
            st.stop()

        c1,c2 = st.columns(2)
        c1.metric("Pasan guardrails", f"{len(kept):,}")
        c2.metric("Rechazados", f"{len(st.session_state['uni'])-len(kept):,}")
        st.dataframe(diag.merge(uni[["symbol","sector"]], on="symbol", how="left"), use_container_width=True, hide_index=True)
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
                df_fund = download_fundamentals(kept_syms, cache_key=cache_tag, force=False)
                base_for_vfq = uni.merge(df_fund, on="symbol", how="right")
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
            # filtros finales
            mask_cov = pd.to_numeric(df_vfq.get("coverage_count", 0), errors="coerce").fillna(0) >= int(min_cov)
            mask_pct = pd.to_numeric(df_vfq.get("VFQ_pct_sector", 1.0), errors="coerce").fillna(1.0) >= float(min_pct)
            df_vfq_sel = df_vfq.loc[mask_cov & mask_pct].copy()

            st.session_state["vfq"] = df_vfq
            st.session_state["vfq_sel"] = df_vfq_sel
        elif "vfq" in st.session_state and "vfq_sel" in st.session_state:
            df_vfq = st.session_state["vfq"]
            df_vfq_sel = st.session_state["vfq_sel"]
        else:
            st.info("Primero corre **Guardrails** (botón Ejecutar).")
            st.stop()

        st.metric("VFQ elegibles", f"{len(df_vfq_sel):,}")
        cols_show = [c for c in [
            "symbol","sector","marketCap_unified","coverage_count","VFQ","ValueScore","QualityScore",
            "fcf_yield","inv_ev_ebitda","gross_profitability","roic","roa","netMargin"
        ] if c in df_vfq_sel.columns]
        st.dataframe(
            df_vfq_sel[cols_show].sort_values(["VFQ","ValueScore","QualityScore"], ascending=False).head(300),
            use_container_width=True, hide_index=True
        )

        n_total = len(df_vfq)
        n_cov   = int(pd.to_numeric(df_vfq.get("coverage_count", 0), errors="coerce").fillna(0).ge(min_cov).sum())
        n_pct   = int(pd.to_numeric(df_vfq.get("VFQ_pct_sector", 1.0), errors="coerce").fillna(1.0).ge(min_pct).sum())
        st.info(f"Embudo VFQ → total={n_total} | cobertura≥{min_cov}: {n_cov} | pct≥{min_pct}: {n_pct} | elegibles={len(df_vfq_sel)}")
    except Exception as e:
        st.error(f"Error en VFQ: {e}")

# ====== Paso 4: SEÑALES ======
with tab4:
    st.subheader("Tendencia & Rompimiento")
    try:
        if not ("vfq_sel" in st.session_state and len(st.session_state["vfq_sel"])>0):
            st.info("Primero corre **VFQ** (botón Ejecutar).")
            st.stop()

        df_vfq_sel = st.session_state["vfq_sel"]
        syms_vfq = df_vfq_sel["symbol"].dropna().astype(str).tolist()
        if len(syms_vfq) == 0:
            st.warning("Sin símbolos tras VFQ; ajusta filtros.")
            st.stop()

        # ↓↓↓ USAR CACHÉ ↓↓↓
        panel = _cached_load_prices_panel(syms_vfq, start.isoformat(), end.isoformat(), cache_key=cache_tag)
        bench_px = _cached_load_benchmark(bench, start.isoformat(), end.isoformat())

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

        base_cols_src = st.session_state.get("vfq", pd.DataFrame())
        base_cols = [c for c in ["symbol","sector","marketCap","VFQ","ValueScore","QualityScore","coverage_count"] if c in base_cols_src.columns]
        base_for_signals = base_cols_src[base_cols].drop_duplicates("symbol") if base_cols else pd.DataFrame({"symbol": syms_vfq})

        df_sig = (
            base_for_signals
            .merge(trend if isinstance(trend, pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_trend"]), on="symbol", how="left")
            .merge(brk if isinstance(brk, pd.DataFrame) else pd.DataFrame(columns=["symbol","signal_breakout"]), on="symbol", how="left")
        )

        for col in ("signal_trend","signal_breakout"):
            if col not in df_sig.columns: df_sig[col] = False
            df_sig[col] = df_sig[col].fillna(False).astype(bool)

        for c in ["RVOL20","ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope","BreakoutScore","hits","c_RVOL","c_ClosePos","c_P52","c_UDVol","c_RSslope"]:
            if c not in df_sig.columns: df_sig[c] = np.nan

        df_sig["ENTRY"] = (df_sig["signal_trend"] & df_sig["signal_breakout"]) if require_breakout else df_sig["signal_trend"]

        if risk_on and not market_regime_on(bench_px, panel, ma_bench=200, breadth_ma=50, breadth_min=0.5):
            st.warning("Régimen OFF (bench ≤ MA200 o breadth ≤ 50%): bloqueando nuevas entradas.")
            df_sig["ENTRY"] = False

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("En tendencia", f"{int(df_sig['signal_trend'].sum())}")
        k2.metric("Breakout", f"{int(df_sig['signal_breakout'].sum())}")
        k3.metric("ENTRY", f"{int(df_sig['ENTRY'].sum())}")
        k4.metric("VFQ elegibles", f"{len(df_vfq_sel):,}")

        dbg_cols = [c for c in ["symbol","RVOL20","ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope","hits","signal_trend","signal_breakout","ENTRY"] if c in df_sig.columns]
        st.caption("Diagnóstico (muestra)")
        st.dataframe(
            df_sig[dbg_cols].sort_values(["ENTRY","signal_breakout","hits","RVOL20","ClosePos","P52","UDVol20"], ascending=[False]*7).head(120),
            use_container_width=True, hide_index=True
        )

        st.subheader("Entradas")
        df_candidates = df_sig.loc[df_sig["ENTRY"]].copy()
        sort_cols = [c for c in ["BreakoutScore","VFQ","ValueScore","QualityScore"] if c in df_candidates.columns]
        asc = [False] * len(sort_cols) if sort_cols else [False]
        st.dataframe(
            df_candidates.sort_values(sort_cols, ascending=asc),
            use_container_width=True, hide_index=True,
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

        # guarda en session para otras pestañas
        st.session_state["guard_diag"] = st.session_state.get("guard_diag", pd.DataFrame())
        st.session_state["signals"] = df_sig
        st.session_state["pipeline_ready"] = True  # pipeline completo disponible

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

# ====== Pestaña 6: BACKTESTING ======
with tab6:
    st.subheader("🔎 Backtesting (por activo)")
    st.markdown(
        "Regla evaluada: **MA200 OR Momentum 12–1 > 0** (o **AND** si marcas el check). "
        "Rebalanceo **M/W**, **lag** opcional (días) y **coste** por turnover (bps). "
        "Métricas por símbolo: **CAGR, Sharpe, Sortino, MaxDD, Turnover, Trades**."
    )

    sig_df = st.session_state.get("signals", pd.DataFrame())
    vfq_sel_df = st.session_state.get("vfq_sel", pd.DataFrame())
    default_syms = []
    try:
        if not sig_df.empty and "ENTRY" in sig_df.columns:
            default_syms = sig_df.loc[sig_df["ENTRY"], "symbol"].dropna().astype(str).unique().tolist()
        if not default_syms and not vfq_sel_df.empty:
            default_syms = vfq_sel_df["symbol"].dropna().astype(str).unique().tolist()
        if not default_syms and 'uni' in st.session_state and isinstance(st.session_state['uni'], pd.DataFrame):
            default_syms = st.session_state['uni']["symbol"].dropna().astype(str).unique().tolist()[:50]
    except Exception:
        pass

    syms_text = st.text_input(
        "Símbolos a backtestear (coma-separados). Si lo dejas vacío uso la selección final.",
        value=",".join(default_syms) if default_syms else ""
    ).strip()
    symbols_bt = [s.strip().upper() for s in syms_text.split(",") if s.strip()] or default_syms

    c1, c2, c3, c4 = st.columns(4)
    cost_bps = c1.number_input("Coste (bps)", min_value=0, max_value=100, value=10, step=1)
    lag_days = c2.number_input("Lag (días)", min_value=0, max_value=30, value=0, step=1)
    use_and_bt = c3.checkbox("Regla AND (MA200 y Mom>0)", value=False)
    freq_bt = c4.selectbox("Frecuencia", options=["M", "W"], index=0)

    run_bt = st.button("▶️ Correr Backtest", use_container_width=True)
    if run_bt:
        if not symbols_bt:
            st.warning("No hay símbolos seleccionados.")
        else:
            import pandas as pd
            extend_days = 420
            start_ext = (pd.to_datetime(start) - pd.Timedelta(days=extend_days)).date().isoformat()
            end_iso = end.isoformat()

            # ↓↓↓ USAR CACHÉ ↓↓↓
            panel_bt = _cached_load_prices_panel(symbols_bt, start_ext, end_iso, cache_key="bt_panel")
            if not panel_bt:
                st.error("No pude cargar precios para los símbolos.")
            else:
                metrics, curves = backtest_many(
                    panel_bt, symbols_bt,
                    cost_bps=cost_bps, lag_days=lag_days,
                    use_and_condition=use_and_bt, rebalance_freq=freq_bt
                )
                st.subheader("Métricas por símbolo")
                st.dataframe(metrics, use_container_width=True)
                st.subheader("Equity curves")
                for s, eq in curves.items():
                    st.line_chart(eq.rename(s), use_container_width=True)

# ====== Pestaña 7: GESTIÓN DE CARTERA (PERSISTENTE) ======
with tab7:
    st.subheader("📊 Gestión de Cartera — Persistente (Kelly fracc. + límite por beta)")
    st.markdown(
        "Calcula pesos **solo** con los símbolos de tu **cartera persistente** (CSV). "
        "Añade desde **Entradas (ENTRY=True)** o manualmente; remueve cuando vendas.  \n"
        "**Sizing**: Kelly `f* = p − (1−p)/b` con `b = AvgWin/|AvgLoss|` → **Kelly fracc.** + **cap por símbolo**, "
        "y normalización `∑(β·w) ≤ beta_cap`."
    )

    # --- Cartera persistente (gestión rápida) ---
    st.markdown("#### Cartera persistente (símbolos en posición)")
    portfolio_df = load_portfolio_df()
    st.dataframe(portfolio_df if not portfolio_df.empty else _portfolio_empty_df(), use_container_width=True, hide_index=True)
    cpa, cpb, cpc = st.columns([0.45, 0.35, 0.20])

    sig_df = st.session_state.get("signals", pd.DataFrame())
    entry_syms = sig_df.loc[sig_df["ENTRY"], "symbol"].dropna().astype(str).unique().tolist() if ("ENTRY" in sig_df.columns and not sig_df.empty) else []
    add_from_entry = cpa.multiselect("Añadir desde entradas (ENTRY=True)", options=entry_syms, default=[])
    manual_add = cpb.text_input("Añadir manual (coma-separado)", value="").strip()
    btn_add = cpc.button("➕ Añadir", use_container_width=True)
    if btn_add:
        syms_to_add = add_from_entry + [s.strip().upper() for s in manual_add.split(",") if s.strip()]
        if syms_to_add:
            portfolio_df = add_symbols_to_portfolio(syms_to_add)
            st.success(f"Agregados: {', '.join(syms_to_add)}")
        else:
            st.info("No hay símbolos para agregar.")

    st.markdown("##### Quitar de la cartera")
    rm_sel = st.multiselect("Selecciona símbolos a remover", options=portfolio_df["symbol"].tolist() if not portfolio_df.empty else [], default=[])
    if st.button("🗑️ Remover seleccionados", use_container_width=True):
        if rm_sel:
            portfolio_df = remove_symbols_from_portfolio(rm_sel)
            st.success(f"Removidos: {', '.join(rm_sel)}")
        else:
            st.info("No seleccionaste símbolos.")

    st.markdown("---")

    # --- Cálculo de pesos (solo cartera persistente) ---
    st.markdown("#### Cálculo de pesos sobre la cartera persistente")
    syms_port = portfolio_df["symbol"].tolist() if not portfolio_df.empty else []
    bench_ticker = st.text_input("Benchmark para beta (ej: SPY)", value=st.session_state.get("bench", "SPY")).strip().upper() or "SPY"
    base_kelly = st.slider("Fracción de Kelly base", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    vol_cap = st.number_input("Cap por posición (proxy vol) [% equity]", min_value=0.01, max_value=0.10, value=0.03, step=0.01, format="%.2f")
    beta_cap = st.number_input("Cap ∑(beta·peso) <=", min_value=0.25, max_value=2.0, value=1.0, step=0.05)

    run_pm = st.button("🧮 Calcular pesos (cartera persistente)", use_container_width=True)
    if run_pm:
        if not syms_port:
            st.warning("Tu cartera persistente está vacía. Agrega símbolos primero.")
        else:
            import pandas as pd
            extend_days = 420
            start_ext = (pd.to_datetime(start) - pd.Timedelta(days=extend_days)).date().isoformat()
            end_iso = end.isoformat()

            # ↓↓↓ USAR CACHÉ ↓↓↓
            pnl = _cached_load_prices_panel(syms_port + [bench_ticker], start_ext, end_iso, cache_key="pm_panel")
            if bench_ticker not in pnl:
                st.error("No pude cargar el benchmark para calcular betas.")
            else:
                bench_daily = pnl[bench_ticker]['close'].pct_change().dropna()
                bench_m = bench_daily.resample('M').apply(lambda x: (1+x).prod()-1).dropna()

                rows = []
                for s in syms_port:
                    if s not in pnl:
                        continue
                    ser = pnl[s]['close'].pct_change().dropna()
                    ser_m = ser.resample('M').apply(lambda x: (1+x).prod()-1).dropna()

                    p, avg_win, avg_loss = win_loss_stats(ser_m)
                    b = abs(avg_win / avg_loss) if (avg_loss not in (0, None)) else np.nan
                    kelly = 0.0
                    if b and b > 0:
                        kelly = max(0.0, min(1.0, p - (1.0 - p)/b))
                    f_prop = min(base_kelly * kelly, vol_cap)

                    beta = beta_vs_bench(ser_m, bench_m)
                    rows.append({
                        'symbol': s,
                        'HitRate': p,
                        'AvgWin': avg_win,
                        'AvgLoss': avg_loss,
                        'Payoff': (abs(avg_win/avg_loss) if (avg_loss not in (0, None)) else np.nan),
                        'Kelly*': kelly,
                        'Propuesta_w': f_prop,
                        'Beta': beta
                    })

                dfw = pd.DataFrame(rows)
                if dfw.empty:
                    st.warning("No se pudieron calcular pesos (verifica datos).")
                else:
                    dfw['beta_w'] = dfw['Beta'].fillna(1.0) * dfw['Propuesta_w'].fillna(0.0)
                    total_beta_w = dfw['beta_w'].sum()
                    scale = 1.0
                    if total_beta_w > beta_cap and total_beta_w > 0:
                        scale = beta_cap / total_beta_w
                    dfw['Peso_final'] = dfw['Propuesta_w'] * scale

                    st.subheader("Pesos propuestos (cartera persistente)")
                    st.dataframe(
                        dfw[['symbol','HitRate','AvgWin','AvgLoss','Payoff','Kelly*','Beta','Propuesta_w','Peso_final']],
                        use_container_width=True
                    )
                    st.caption("Kelly fracc. + cap por símbolo, normalizado por β total ≤ beta_cap.")
