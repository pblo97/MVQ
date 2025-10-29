# --- poner esto ARRIBA DE TODO ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"  # o "none" si prefieres desactivar
# ---------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Tuple
import altair as alt

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
    # fetch_profiles=True baja sector/industry del perfil en la misma pasada
    return run_fmp_screener(limit=limit, fetch_profiles=True)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_guardrails(symbols: Tuple[str, ...], cache_key: str) -> pd.DataFrame:
    return download_guardrails_batch(list(symbols), cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_download_fundamentals(
    symbols: Tuple[str, ...],
    cache_key: str,
    mc_pairs: Tuple[Tuple[str, float], ...] | None = None,
) -> pd.DataFrame:
    mc_map = dict(mc_pairs or ())
    return download_fundamentals(
        list(symbols),
        market_caps=mc_map,          # <-- hint
        cache_key=cache_key,
        force=False
    )

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

def _enrich_sector_industry(uni_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    out = uni_df.copy()
    need_sector = ("sector" not in out.columns) or (out["sector"].isna().mean() > 0.8 if "sector" in out.columns else True)
    have_cols = [c for c in ["sector", "industry"] if c in src_df.columns]
    if need_sector and have_cols:
        map_df = (
            src_df[["symbol"] + have_cols]
            .dropna(subset=["symbol"])
            .drop_duplicates("symbol", keep="last")
        )
        out = out.drop(columns=have_cols, errors="ignore").merge(map_df, on="symbol", how="left")


        # ⬇️ claves: no llamar fillna sobre un string
    if "sector" in out.columns:
        out["sector"] = out["sector"].astype(str).replace({"": "Unknown"}).fillna("Unknown")
    else:
        out["sector"] = "Unknown"

    if "industry" in out.columns:
        out["industry"] = out["industry"].astype(str).fillna("")
    else:
        out["industry"] = ""

    return out

def _as_series(x, index=None):
    import pandas as pd
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=index)

def _ensure_sector_strings(df: pd.DataFrame, sector_col="sector", industry_col="industry") -> pd.DataFrame:
    import numpy as np, pandas as pd
    if sector_col not in df.columns:
        df[sector_col] = pd.Series(["Unknown"] * len(df), index=df.index)
    else:
        s = _as_series(df[sector_col], df.index)
        s = s.astype(str)
        s = s.replace({"": "Unknown"})
        s = s.where(~s.isna(), "Unknown")
        df[sector_col] = s

    if industry_col in df.columns:
        t = _as_series(df[industry_col], df.index)
        df[industry_col] = t.astype(str).where(~t.isna(), "")
    return df

def _as_list(x):
    return x if isinstance(x, (list, tuple, pd.Index, np.ndarray)) else [x]

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
    # Nombres que sí existen/deriva build_vfq_scores_dynamic
    value_metrics_opts   = ["inv_ev_ebitda", "fcf_yield"]
    quality_metrics_opts = ["gross_profitability", "roic", "roa", "netMargin"]

    sel_value = st.multiselect("Métricas Value", options=value_metrics_opts, default=["inv_ev_ebitda", "fcf_yield"])
    sel_quality = st.multiselect("Métricas Quality", options=quality_metrics_opts, default=["gross_profitability", "roic"])

    c1, c2 = st.columns(2)
    with c1: w_value = st.slider("Peso Value", 0.0, 1.0, 0.5, 0.05)
    with c2: w_quality = st.slider("Peso Quality", 0.0, 1.0, 0.5, 0.05)

    method_intra = st.radio("Agregación intra-bloque", ["mean", "median", "weighted_mean"], index=0, horizontal=True)
    winsor_p = st.slider("Winsor p (cola)", 0.0, 0.10, 0.01, 0.005)
    size_buckets = st.slider("Buckets por tamaño", 1, 5, 3, 1)
    group_mode = st.selectbox("Agrupar por", ["sector", "sector|size"], index=1)
    min_cov = st.slider("Cobertura mín. (# métricas)", 0, 8, 1, 1)
    min_pct = st.slider("VFQ pct (intra-sector) mín.", 0.00, 1.00, 0.00, 0.01)

    st.session_state["min_cov"] = int(min_cov)
    st.session_state["min_pct"] = float(min_pct)

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
# ====== Paso 3: VFQ (PARCHE COMPLETO + UI BONITA) ======


def _fmt_mcap(x):
    try:
        x = float(x)
        if x >= 1e12:  return f"${x/1e12:.2f}T"
        if x >= 1e9:   return f"${x/1e9:.2f}B"
        if x >= 1e6:   return f"${x/1e6:.2f}M"
        return f"${x:,.0f}"
    except Exception:
        return ""

def _numcol(df: pd.DataFrame, col: str) -> pd.Series:
    import pandas as pd
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

# ====== Paso 3: VFQ (PARCHE COMPLETO) ======
# ====== Paso 3: VFQ (PARCHE COMPLETO) ======
with tab3:
    st.subheader("VFQ")

    # --- helper local para no romper en rutas warm ---
    def _build_mask_sane(df: pd.DataFrame) -> pd.Series:
        rev = _numcol(df, "revenue_ttm")
        gp  = _numcol(df, "gross_profit_ttm")
        yoy = _numcol(df, "revenue_yoy")
        mask_rev_ok = (rev > 0) | (gp > 0)
        mask_yoy_ok = (yoy.isna()) | (yoy > -0.05)
        out = (mask_rev_ok & mask_yoy_ok).fillna(False)
        if len(df) and not out.any():
            # si todos False por datos faltantes, relaja a True para no vaciar universo
            out = pd.Series(True, index=df.index)
        return out

    try:
        # ===================== RUTA "RUN" =====================
        if run_btn and "kept" in st.session_state:
            uni  = st.session_state["uni"]
            kept = st.session_state["kept"]
            kept_syms = (
                kept.get("symbol", pd.Series(dtype=str))
                    .dropna().astype(str).unique().tolist()
            )
            if not kept_syms:
                st.warning("No hay símbolos en 'kept'. Ajusta filtros antes de ejecutar VFQ.")
                st.stop()

            # --- hint de market cap desde el universo ---
            mc_hint_df = (
                uni.loc[uni["symbol"].isin(kept_syms), ["symbol", "marketCap"]]
                   .dropna(subset=["symbol"])
                   .copy()
            )
            mc_hint_df["symbol"]    = mc_hint_df["symbol"].astype(str)
            mc_hint_df["marketCap"] = pd.to_numeric(mc_hint_df["marketCap"], errors="coerce")
            mc_pairs = tuple(
                (row.symbol, float(row.marketCap))
                for _, row in mc_hint_df.dropna(subset=["marketCap"]).iterrows()
            )

            with st.status("Descargando fundamentales VFQ (TTM)…", expanded=False) as status:
                # ÚNICA llamada, pasando mc_pairs
                df_fund = _cached_download_fundamentals(
                    tuple(sorted(kept_syms)),
                    cache_key=cache_tag,
                    mc_pairs=mc_pairs
                )

                # sector/industry desde fundamentals + normalización
                uni_enriched = _enrich_sector_industry(uni, df_fund)
                uni_enriched = _ensure_sector_strings(uni_enriched)

                # --- OPTIMIZACIÓN: usa df_fund como driver y trae solo sector/industry del universo ---
                _cols = ["symbol"]
                if "sector" in uni_enriched.columns:   _cols.append("sector")
                if "industry" in uni_enriched.columns: _cols.append("industry")

                base_for_vfq = df_fund.merge(
                    uni_enriched[_cols].drop_duplicates("symbol"),
                    on="symbol",
                    how="left"
                )
                base_for_vfq = _ensure_sector_strings(base_for_vfq)

                # ===== cálculo VFQ =====
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

                # === COALESCE de sector/industry: preferir lo que ya viene en df_vfq y completar con universo ===
                keys = ["sector", "industry"]
                tmp_uni = uni_enriched[["symbol"] + [c for c in keys if c in uni_enriched.columns]].drop_duplicates("symbol")
                df_vfq = df_vfq.merge(tmp_uni, on="symbol", how="left", suffixes=("", "_uni"))
                for c in keys:
                    cu = f"{c}_uni"
                    if c in df_vfq.columns and cu in df_vfq.columns:
                        df_vfq[c] = df_vfq[c].where(
                            df_vfq[c].notna() & (df_vfq[c].astype(str).str.len() > 0),
                            df_vfq[cu]
                        )
                        df_vfq.drop(columns=[cu], inplace=True)

                df_vfq = _ensure_sector_strings(df_vfq)
                df_vfq["symbol"] = df_vfq["symbol"].astype(str)

                # ===== score y percentil (único cálculo, robusto con fallback global) =====
                score_col = "VFQ" if "VFQ" in df_vfq.columns else ("VFQ_score" if "VFQ_score" in df_vfq.columns else None)
                if score_col is None:
                    st.error("No encontré columna de score ('VFQ' o 'VFQ_score') en df_vfq.")
                    st.stop()

                tmp = _ensure_sector_strings(df_vfq.copy())
                grp_sz  = tmp.groupby("sector")["symbol"].transform("size") if "sector" in tmp.columns else pd.Series(0, index=tmp.index)
                pct_sec = tmp.groupby("sector")[score_col].rank(pct=True) if "sector" in tmp.columns else tmp[score_col].rank(pct=True)
                pct_glb = tmp[score_col].rank(pct=True)
                df_vfq["VFQ_pct_sector"] = np.where(grp_sz >= 6, pct_sec, pct_glb)
                df_vfq["VFQ_pct_sector"] = pd.to_numeric(df_vfq["VFQ_pct_sector"], errors="coerce").clip(0.0, 1.0).fillna(1.0)

                # ===== filtros UI =====
                min_cov_val = int(st.session_state.get("min_cov", 0))
                min_pct_val = float(st.session_state.get("min_pct", 0.0))

                if "coverage_count" in df_vfq.columns:
                    mask_cov = pd.to_numeric(df_vfq["coverage_count"], errors="coerce").fillna(0) >= min_cov_val
                else:
                    mask_cov = pd.Series(True, index=df_vfq.index)

                # ===== filtro sanitario (siempre) =====
                mask_sane = _build_mask_sane(df_vfq)

                mask_pct = pd.to_numeric(df_vfq["VFQ_pct_sector"], errors="coerce").fillna(1.0) >= min_pct_val
                df_vfq_sel = df_vfq.loc[mask_cov & mask_pct & mask_sane].copy()

                # guarda en sesión TODO lo necesario
                st.session_state["vfq"] = df_vfq
                st.session_state["vfq_sel"] = df_vfq_sel
                st.session_state["mask_sane"] = mask_sane

                status.update(label="VFQ calculado", state="complete")

        # ===================== RUTA "WARM" =====================
        elif "vfq" in st.session_state and "vfq_sel" in st.session_state:
            df_vfq     = st.session_state["vfq"].copy()
            df_vfq_sel = st.session_state["vfq_sel"].copy()

            # asegura que exista mask_sane en warm (y sea consistente en longitud)
            mask_sane = st.session_state.get("mask_sane", None)
            if (mask_sane is None) or (len(mask_sane) != len(df_vfq)):
                mask_sane = _build_mask_sane(df_vfq)
                st.session_state["mask_sane"] = mask_sane

            # score_col para orden
            if "VFQ" in df_vfq.columns:
                score_col = "VFQ"
            elif "VFQ_score" in df_vfq.columns:
                score_col = "VFQ_score"
            else:
                score_col = None
        else:
            st.info("Primero corre **Guardrails** (botón Ejecutar).")
            st.stop()

        # ===== orden por score (preferencia VFQ) =====
        if "VFQ" in df_vfq_sel.columns:
            sort_col = "VFQ"
        elif "VFQ_score" in df_vfq_sel.columns:
            sort_col = "VFQ_score"
        else:
            sort_col = None

        view_df = df_vfq_sel.sort_values(sort_col, ascending=False) if sort_col else df_vfq_sel.copy()

        # ===== KPIs + gráfico por sector =====
        left, right = st.columns([0.25, 0.75])
        with left:
            st.markdown("### VFQ")
            st.metric("Con VFQ calculado", f"{len(df_vfq):,}")
            st.metric("Seleccionados (filtros)", f"{len(df_vfq_sel):,}")

        with right:
            sec = _ensure_sector_strings(df_vfq_sel.copy())
            sector_counts = (
                sec.groupby("sector", dropna=False)
                   .size().reset_index(name="count")
                   .sort_values("count", ascending=False).head(20)
            )
            chart = (
                alt.Chart(sector_counts).mark_bar().encode(
                    x=alt.X("count:Q", title="Cantidad"),
                    y=alt.Y("sector:N", sort="-x", title=None),
                    tooltip=["sector", "count"]
                ).properties(height=320, width="container")
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # ===== Búsqueda + orden =====
        f1, f2, f3 = st.columns([0.36, 0.36, 0.28])
        with f1:
            search = st.text_input("🔎 Buscar (symbol/sector/industry)", "")
        with f2:
            order_opts = [
                c for c in [
                    "VFQ", "VFQ_pct_sector", "final_alpha", "prob_up", "qvm_score",
                    "BreakoutScore", "momentum_score", "marketCap_unified", "marketCap", "price"
                ] if c in df_vfq_sel.columns
            ]
            sort_by = st.selectbox("Ordenar por", order_opts, index=0) if order_opts else None
        with f3:
            ascending = st.toggle("Ascendente", value=False)

        # ===== Vista =====
        # ===== Vista (construir una sola vez) =====
        # 1) base: seleccionados o fallback
        if df_vfq_sel.empty:
            st.warning("Ningún símbolo pasó los filtros. Te muestro el Top por VFQ (sanitizado) para depurar.")
            _ms = st.session_state.get("mask_sane")
            if (_ms is None) or (len(_ms) != len(df_vfq)):
                _ms = _build_mask_sane(df_vfq)
                st.session_state["mask_sane"] = _ms
            tmp = df_vfq.loc[_ms].copy()
            base_sort_col = "VFQ" if "VFQ" in tmp.columns else ("VFQ_score" if "VFQ_score" in tmp.columns else None)
            view = tmp.sort_values(base_sort_col, ascending=False) if base_sort_col else tmp.copy()
        else:
            base_sort_col = "VFQ" if "VFQ" in df_vfq_sel.columns else ("VFQ_score" if "VFQ_score" in df_vfq_sel.columns else None)
            view = df_vfq_sel.sort_values(base_sort_col, ascending=False).copy()

        view = _ensure_sector_strings(view)

        # 2) filtro por texto
        if search.strip():
            s = search.strip().lower()
            mask_any = None
            for col in ("symbol", "sector", "industry"):
                if col in view.columns:
                    m = view[col].astype(str).str.lower().str.contains(s, na=False)
                    mask_any = m if mask_any is None else (mask_any | m)
            if mask_any is not None:
                view = view[mask_any]

        # 3) columnas derivadas SIEMPRE después de fijar 'view'
        # --- market_cap amigable
        if "marketCap_unified" in view.columns:
            view["market_cap"] = pd.to_numeric(view["marketCap_unified"], errors="coerce").apply(_fmt_mcap)
        elif "marketCap" in view.columns:
            view["market_cap"] = pd.to_numeric(view["marketCap"], errors="coerce").apply(_fmt_mcap)

        # --- percentil visible
        if "VFQ_pct_sector" in view.columns:
            pct = pd.to_numeric(view["VFQ_pct_sector"], errors="coerce")
            pct = np.where(pct > 1.5, pct / 100.0, pct)  # si llegó 0..100
            pct = pd.Series(pct, index=view.index).clip(0.0, 1.0)
            view["VFQ pct (sector)"] = (pct * 100).round(2).clip(0, 100)
        else:
            _sc = "VFQ" if "VFQ" in view.columns else ("VFQ_score" if "VFQ_score" in view.columns else None)
            if _sc is not None:
                tmp = _ensure_sector_strings(view.copy())
                if "sector" in tmp.columns and tmp["sector"].nunique(dropna=False) > 1:
                    pct = tmp.groupby("sector")[_sc].rank(pct=True, method="average")
                else:
                    pct = tmp[_sc].rank(pct=True, method="average")
                pct = pd.to_numeric(pct, errors="coerce").clip(0.0, 1.0)
                view["VFQ pct (sector)"] = (pct * 100).round(2)

        # 4) redondeos
        for col in ["VFQ", "VFQ_score", "value_adj_neut", "quality_adj_neut",
                    "BreakoutScore", "momentum_score", "beta", "price"]:
            if col in view.columns:
                view[col] = pd.to_numeric(view[col], errors="coerce").round(3)

        # 5) ordenar por lo elegido en la UI (sin destruir derivadas)
        if sort_by and sort_by in view.columns:
            view = view.sort_values(sort_by, ascending=ascending, na_position="last")

        # 6) columnas visibles (solo las que existan)
        pretty_cols_base = [
            "symbol", "sector", "industry", "market_cap", "price", "beta",
            "VFQ", "VFQ_score", "VFQ pct (sector)", "value_adj_neut",
            "quality_adj_neut", "BreakoutScore", "momentum_score"
        ]
        pretty_cols = [c for c in pretty_cols_base if c in view.columns]
        if not pretty_cols:
            pretty_cols = [c for c in ["symbol", "sector", "industry"] if c in view.columns] or view.columns.tolist()

        # ===== Render tabla =====
        st.dataframe(
            view[pretty_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                **({"market_cap": st.column_config.TextColumn("Market Cap", help="Unificado/estimado")} if "market_cap" in pretty_cols else {}),
                **({"price":      st.column_config.NumberColumn("Price", format="%.2f")} if "price" in pretty_cols else {}),
                **({"beta":       st.column_config.NumberColumn("Beta", format="%.3f")} if "beta" in pretty_cols else {}),
                **({"VFQ":        st.column_config.NumberColumn("VFQ", help="Score agregado", format="%.3f")} if "VFQ" in pretty_cols else {}),
                **({"VFQ_score":  st.column_config.NumberColumn("VFQ_score", help="Score agregado", format="%.3f")} if "VFQ_score" in pretty_cols else {}),
                **({"VFQ pct (sector)": st.column_config.ProgressColumn("VFQ pct (sector)", min_value=0, max_value=100, help="Percentil intra-sector (0–100%)")} if "VFQ pct (sector)" in pretty_cols else {}),
                **({"value_adj_neut":   st.column_config.NumberColumn("Value (neut.)", format="%.3f")} if "value_adj_neut" in pretty_cols else {}),
                **({"quality_adj_neut": st.column_config.NumberColumn("Quality (neut.)", format="%.3f")} if "quality_adj_neut" in pretty_cols else {}),
                **({"BreakoutScore":    st.column_config.NumberColumn("Breakout", format="%.3f")} if "BreakoutScore" in pretty_cols else {}),
                **({"momentum_score":   st.column_config.NumberColumn("Momentum", format="%.3f")} if "momentum_score" in pretty_cols else {}),
            }
        )

        # ======= Vistas extra: ranking por hits / BreakoutScore =======


        # --------- Descargas ----------
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Seleccionados (CSV)",
                view[pretty_cols].to_csv(index=False).encode(),
                "vfq_seleccionados.csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "⬇️ VFQ completo (CSV)",
                df_vfq.to_csv(index=False).encode(),
                "vfq_completo.csv",
                use_container_width=True
            )
        with c3:
            st.download_button(
                "⬇️ Universo crudo (CSV)",
                st.session_state.get("uni", pd.DataFrame()).to_csv(index=False).encode(),
                "universo.csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error en VFQ: {e}")

# ====== Paso 4: SEÑALES (placeholder si tu lógica está en otro módulo) ======
with tab4:
    st.subheader("Señales (Técnico)")
    try:
        if run_btn and "vfq_sel" in st.session_state:
            syms = st.session_state["vfq_sel"]["symbol"].dropna().astype(str).unique().tolist()

            with st.status("Cargando precios y calculando señales…", expanded=False) as status:
                # 1) precios como panel {symbol: df}
                # Símbolos base para señales: preferimos selección VFQ; si está vacía, fallback al Top VFQ sanitario
                if "vfq_sel" in st.session_state and isinstance(st.session_state["vfq_sel"], pd.DataFrame) and not st.session_state["vfq_sel"].empty:
                    syms = st.session_state["vfq_sel"]["symbol"].dropna().astype(str).unique().tolist()
                else:
                    df_vfq = st.session_state.get("vfq", pd.DataFrame())
                    if not df_vfq.empty:
                        _ms = st.session_state.get("mask_sane")
                        if _ms is None or len(_ms) != len(df_vfq):
                            _ms = _build_mask_sane(df_vfq)
                        tmp = df_vfq.loc[_ms].copy()
                        scol = "VFQ" if "VFQ" in tmp.columns else ("VFQ_score" if "VFQ_score" in tmp.columns else None)
                        if scol:
                            syms = tmp.sort_values(scol, ascending=False)["symbol"].dropna().astype(str).head(50).tolist()
                        else:
                            syms = tmp["symbol"].dropna().astype(str).head(50).tolist()
                    else:
                        syms = []

                if not syms:
                    st.warning("No hay símbolos para señales (ni selección VFQ ni fallback). Revisa filtros de VFQ.")
                    st.stop()

                panel = _cached_load_prices_panel(syms, start=str(start), end=str(end), cache_key=cache_tag)

                # 2) tendencia (AND/OR correcto)
                trend_df = apply_trend_filter(panel, use_and_condition=use_and)

                # 3) breakout (usa panel; NO admite atr_pct_min ni require_breakout)
                bo_df = enrich_with_breakout(
                    panel,
                    rvol_lookback=20,                 # por si quieres exponerlo luego
                    rvol_th=float(rvol_th),
                    closepos_th=float(closepos_th),
                    p52_th=float(p52_th),
                    updown_vol_th=float(updown_vol_th),
                    bench_series=None,               # opcional: serie de benchmark normalizada
                    min_hits=int(min_hits),
                    use_rs_slope=bool(use_rs_slope),
                    rs_min_slope=0.0
                )

                # 4) mezcla: una fila por símbolo
                sig_df = (
                    trend_df.merge(bo_df, on="symbol", how="outer")
                            .sort_values("symbol")
                            .reset_index(drop=True)
                )

                # 5) régimen de mercado (bool). Si risk_on=True en UI, aplicamos el freno.
                bench_df = _cached_load_benchmark(bench, start=str(start), end=str(end))
                ok_market = market_regime_on(bench_df, panel)

                if risk_on and not ok_market:
                    # No filtramos filas; apagamos las banderas de entrada
                    if "signal_trend" in sig_df.columns:
                        sig_df["signal_trend"] = False
                    if "signal_breakout" in sig_df.columns:
                        sig_df["signal_breakout"] = False
                    sig_df["risk_on"] = False
                else:
                    sig_df["risk_on"] = True

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
    sig = st.session_state.get("signals", pd.DataFrame()).copy()

    # Blindajes suaves
    need_cols = ["symbol", "hits", "BreakoutScore", "signal_breakout", "risk_on"]
    for c in need_cols:
        if c not in sig.columns:
            # crea si falta, para no romper la UI
            sig[c] = 0 if c in ("hits", "BreakoutScore") else False

    # Controles
    st.markdown("---")
    c1, c2, c3, c4 = st.columns([0.2, 0.25, 0.25, 0.30])
    with c1:
        top_n = st.number_input("Top N", min_value=5, max_value=300, value=50, step=5)
    with c2:
        only_risk_on = st.checkbox("Solo risk_on ✅", value=False)
    with c3:
        only_breakouts = st.checkbox("Solo con signal_breakout 🚀", value=False)
    with c4:
        cols_show = [c for c in [
            "symbol","hits","BreakoutScore","risk_on","signal_breakout",
            "ClosePos","P52","RVOL20","UDVol20","ATR_pct","mom_12_1","rs_ma20_slope"
        ] if c in sig.columns]

    # Filtros rápidos
    mask = pd.Series(True, index=sig.index)
    if only_risk_on and "risk_on" in sig.columns:
        mask &= sig["risk_on"].fillna(False)
    if only_breakouts and "signal_breakout" in sig.columns:
        mask &= sig["signal_breakout"].fillna(False)
    base = sig.loc[mask].copy()

    # —— Vista 1: Top por hits ——
    if len(base):
        st.subheader("Top por **hits**")
        view_hits = (
            base.sort_values(["hits","BreakoutScore"], ascending=[False, False])
                .loc[:, cols_show]
                .head(top_n)
        )
        st.dataframe(view_hits, use_container_width=True, hide_index=True)

    # —— Vista 2: Top por BreakoutScore ——
    if len(base):
        st.subheader("Top por **BreakoutScore**")
        view_bo = (
            base.sort_values(["BreakoutScore","hits"], ascending=[False, False])
                .loc[:, cols_show]
                .head(top_n)
        )
        st.dataframe(view_bo, use_container_width=True, hide_index=True)

    # —— Vista 3 (opcional): Ranking combinado ——
    if len(base):
        st.subheader("Ranking **combinado** (hits → BreakoutScore)")
        # si quieres un score compuesto, descomenta estas 2 líneas:
        # z_hits = (base["hits"] - base["hits"].mean()) / (base["hits"].std() or 1)
        # z_bo   = (base["BreakoutScore"] - base["BreakoutScore"].mean()) / (base["BreakoutScore"].std() or 1)
        # base["combo_rank"] = 0.6*z_hits + 0.4*z_bo
        # view_combo = base.sort_values("combo_rank", ascending=False)

        # por defecto, multi-orden estable:
        view_combo = (
            base.sort_values(["hits","BreakoutScore"], ascending=[False, False])
                .loc[:, cols_show]
                .head(top_n)
        )
        st.dataframe(view_combo, use_container_width=True, hide_index=True)

# ====== Paso 5: QVM (growth-aware) ======
# ============ QVM CORE: FundamentalStandardizer ============

from dataclasses import dataclass, field

@dataclass
class FundamentalStandardizer:
    """
    Normaliza/alia columnas fundamentales y técnicas para que QVM funcione
    sin 'Unknown' de sector ni Series 2D. Incluye coalesce *_vfq → limpio,
    coerción numérica, y derivaciones básicas (market_cap, net_debt, etc.)
    """
    alias: dict = field(default_factory=lambda: {
        # básicos
        "market_cap": ["market_cap","marketCap","marketCap_unified","mkt_cap","Market Cap"],
        "sector":     ["sector","Sector","industry","Industry","gicsSector","GICS_Sector"],
        # value
        "ev":                ["ev","enterpriseValue","EnterpriseValue","EV"],
        "ebitda_ttm":        ["ebitda_ttm","EBITDA_TTM","ebitdaTrailingTwelveMonths","ebitdaTTM"],
        "ebitda_ntm":        ["ebitda_ntm","EBITDA_NTM","ebitdaForward","ebitdaNextTwelveMonths"],
        "gross_profit_ttm":  ["gross_profit_ttm","grossProfitTTM","GrossProfitTTM"],
        "sales_ntm":         ["sales_ntm","revenueNTM","RevenueNTM","salesForward","revenueForward","revenue_ntm"],
        "capex_ttm":         ["capex_ttm","capexTTM","CapExTTM","capitalExpenditureTTM"],
        "sbc_ttm":           ["sbc_ttm","stockBasedCompTTM","shareBasedCompTTM","SBC_TTM"],
        "fcf_ttm":           ["fcf_ttm","freeCashFlowTTM","FCF_TTM"],
        "fcf_5y_median":     ["fcf_5y_median","FCF_5Y_MEDIAN","fcfMedian5Y"],
        # quality / intangible
        "rd_expense_ttm":        ["rd_expense_ttm","researchDevelopmentTTM","R&D_TTM","researchAndDevTTM"],
        "operating_income_ttm":  ["operating_income_ttm","operatingIncomeTTM","OperatingIncomeTTM","ebitTTM"],
        "total_assets_ttm":      ["total_assets_ttm","totalAssetsTTM","TotalAssetsTTM","totalAssets"],
        "net_debt_ttm":          ["net_debt_ttm","netDebtTTM","NetDebtTTM","netDebt"],
        "total_debt_ttm":        ["total_debt_ttm","totalDebtTTM","TotalDebtTTM","totalDebt"],
        "cash_ttm":              ["cash_ttm","cashAndEquivalentsTTM","cashAndCashEquivalentsTTM","cashAndEquivalents"],
        "noa_ttm":               ["noa_ttm","netOperatingAssetsTTM","NOA_TTM"],
        "invested_capital_ttm":  ["invested_capital_ttm","investedCapitalTTM","InvestedCapitalTTM"],
        "current_liabilities_ttm":["current_liabilities_ttm","currentLiabilitiesTTM","CurrentLiabilitiesTTM"],
        "tax_rate":              ["tax_rate","effectiveTaxRate","effectiveTaxRateTTM"],
        "op_margin_hist":        ["op_margin_hist","operatingMarginHistory","opMarginHistory"],
        # técnico
        "BreakoutScore": ["BreakoutScore"],
        "RVOL20":        ["RVOL20"],
        "UDVol20":       ["UDVol20"],
        "hits":          ["hits"],
        "P52":           ["P52"],
        "ClosePos":      ["ClosePos"],
        "momentum_score":["momentum_score","mom","momo","mom_sig","mom_px","momentum_score_prices"],
    })

    def _first_present(self, df: pd.DataFrame, std: str):
        """Toma la primera columna candidata presente; si no está, intenta *_vfq."""
        # ya existe la estándar
        if std in df.columns:
            return df[std]
        # busca alias directos
        for c in self.alias.get(std, []):
            if c in df.columns:
                return df[c]
        # busca versión *_vfq
        cand_vfq = f"{std}_vfq"
        if cand_vfq in df.columns:
            return df[cand_vfq]
        # busca alias con _vfq
        for c in self.alias.get(std, []):
            c_v = f"{c}_vfq"
            if c_v in df.columns:
                return df[c_v]
        # default vacío
        return pd.Series(np.nan, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Elimina columnas duplicadas por nombre (causa Series 2D)
        out = out.loc[:, ~out.columns.duplicated(keep="last")]

        # Asegura 'symbol'
        if "symbol" not in out.columns:
            raise ValueError("Falta columna 'symbol'")

        # Construye y coalesce columnas estándar
        std_cols = [
            # básicos
            "market_cap","sector",
            # value
            "ev","ebitda_ttm","ebitda_ntm","gross_profit_ttm","sales_ntm",
            "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median",
            # quality/intangible
            "rd_expense_ttm","operating_income_ttm","total_assets_ttm",
            "net_debt_ttm","total_debt_ttm","cash_ttm","noa_ttm",
            "invested_capital_ttm","current_liabilities_ttm","tax_rate","op_margin_hist",
            # técnico
            "BreakoutScore","RVOL20","UDVol20","hits","P52","ClosePos","momentum_score"
        ]
        for std in std_cols:
            s = self._first_present(out, std)
            # promedia fila a fila si llegó DataFrame (2D)
            if isinstance(s, pd.DataFrame):
                s = s.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            out[std] = s

        # Sector: si queda Unknown y hay industry, úsala
        if out["sector"].isna().all() and "industry" in out.columns:
            out["sector"] = out["industry"].astype(str)

        out["sector"] = out["sector"].astype(str).fillna("Unknown")
        # Market cap: coalesce si faltó
        if out["market_cap"].isna().all():
            for alt in ["marketCap_unified","marketCap","mkt_cap"]:
                if alt in df.columns:
                    out["market_cap"] = pd.to_numeric(df[alt], errors="coerce")
                    break

        # Derivación net_debt si falta: total_debt - cash
        if out["net_debt_ttm"].isna().all():
            td = pd.to_numeric(out.get("total_debt_ttm"), errors="coerce")
            cs = pd.to_numeric(out.get("cash_ttm"), errors="coerce")
            out["net_debt_ttm"] = (td - cs)

        # Coerción numérica masiva
        num_cols = [
            "market_cap","ev","ebitda_ttm","ebitda_ntm","gross_profit_ttm","sales_ntm",
            "capex_ttm","sbc_ttm","fcf_ttm","fcf_5y_median","rd_expense_ttm",
            "operating_income_ttm","total_assets_ttm","net_debt_ttm","noa_ttm",
            "invested_capital_ttm","current_liabilities_ttm","tax_rate",
            "BreakoutScore","RVOL20","UDVol20","hits","P52","ClosePos","momentum_score"
        ]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # Momentum seguro
        if "momentum_score" not in out.columns or out["momentum_score"].isna().all():
            out["momentum_score"] = 0.0
        else:
            # por si vinieron múltiples fuentes mezcladas
            if isinstance(out["momentum_score"], pd.DataFrame):
                out["momentum_score"] = out["momentum_score"].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            else:
                out["momentum_score"] = pd.to_numeric(out["momentum_score"], errors="coerce").fillna(0.0)

        # Limpieza final de duplicados de nombre
        out = out.loc[:, ~out.columns.duplicated(keep="last")]
        return out

# =================== TAB 5: QVM (growth-aware) ===================
with tab5:
    st.subheader("QVM (growth-aware) con Guardrails → Técnico → Ranking")

    # --- Parámetros UI ---
    colA, colB, colC = st.columns(3)
    with colA:
        th_q_neut   = st.slider("Min Quality neut.", 0.0, 1.0, 0.0, 0.05)   # puedes subir a 0.2–0.3
        th_v_neut   = st.slider("Min Value neut.",   0.0, 1.0, 0.0, 0.05)
        th_ndebt    = st.slider("Max NetDebt/EBITDA", 0.0, 8.0, 3.0, 0.5)
    with colB:
        th_acc_p    = st.slider("Accruals (NOA) percentil mínimo", 0, 100, 30, 5)
        beta_prob   = st.slider("β prob_up (logit)", 2.0, 12.0, 6.0, 0.5)
        top_n_show  = st.slider("Top N por prob_up", 5, 100, 30, 5)
    with colC:
        th_bo       = st.slider("Min BreakoutScore", 0, 100, 60, 1)
        th_hits     = st.slider("Min hits", 0, 5, 2, 1)
        th_rvol     = st.slider("Min RVOL20", 0.0, 5.0, 1.5, 0.1)

    try:
        # ------------------- Orígenes -------------------
        sig_df        = st.session_state.get("signals", pd.DataFrame())
        vfq_df        = st.session_state.get("vfq", pd.DataFrame())
        uni_df        = st.session_state.get("uni", pd.DataFrame())
        kept_df       = st.session_state.get("kept", pd.DataFrame())
        panel_prices  = st.session_state.get("panel_prices")  # opcional

        if sig_df.empty:
            st.info("Primero corre **Señales**.")
            st.stop()

        # ------------------- BASE desde señales -------------------
        base_keep = ["symbol","sector","industry","marketCap","marketCap_unified",
                     "BreakoutScore","ClosePos","P52","RVOL20","UDVol20","hits","rs_ma20_slope"]
        base_cols = ["symbol"] + [c for c in base_keep if c in sig_df.columns and c != "symbol"]
        base = sig_df[base_cols].drop_duplicates("symbol").copy()

        # Merge VFQ (fundamentales) y UNI (sector/mcap) si existen
        if isinstance(vfq_df, pd.DataFrame) and not vfq_df.empty:
            base = base.merge(vfq_df, on="symbol", how="left", suffixes=("", "_vfq"))
        if isinstance(uni_df, pd.DataFrame) and {"symbol"}.issubset(uni_df.columns):
            extra = [c for c in ["symbol","sector","industry","marketCap","market_cap"] if c in uni_df.columns]
            if len(extra) > 1:
                base = base.merge(uni_df[extra].drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_uni"))
        if isinstance(kept_df, pd.DataFrame) and "symbol" in kept_df.columns:
            base = base.merge(kept_df.drop_duplicates("symbol")[["symbol"]], on="symbol", how="right")

        # ------------------- Momentum proxy (coalesce) -------------------
        # 1) desde señales (idempotente)
        mom_sig = build_momentum_proxy(sig_df)
        if isinstance(mom_sig, pd.Series) and not mom_sig.empty:
            base = base.merge(mom_sig.to_frame("mom_sig"), left_on="symbol", right_index=True, how="left")

        # 2) desde precios (si está panel_prices)
        mom_px = None
        try:
            if isinstance(panel_prices, pd.DataFrame):
                df_long = panel_prices.copy()
                if "symbol" not in df_long.columns and "ticker" in df_long.columns:
                    df_long = df_long.rename(columns={"ticker": "symbol"})
                mom_px = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
            elif isinstance(panel_prices, dict):
                frames = []
                for sym, dfp in panel_prices.items():
                    if isinstance(dfp, pd.DataFrame) and "close" in dfp.columns:
                        tmp = dfp.reset_index().rename(columns={"index": "date"} if "date" not in dfp.columns else {})
                        tmp["symbol"] = sym
                        frames.append(tmp[["symbol","date","close"]])
                if frames:
                    df_long = pd.concat(frames, ignore_index=True)
                    mom_px = build_momentum_proxy(df_long, price_col="close", id_col="symbol", date_col="date")
        except Exception:
            mom_px = None

        if isinstance(mom_px, pd.Series) and not mom_px.empty:
            base = base.merge(mom_px.to_frame("mom_px"), left_on="symbol", right_index=True, how="left")

        # Coalesce momentum → 'momentum_score' 1D
        cand_moms = [c for c in ["momentum_score","mom_sig","mom_px","momentum_score_prices"] if c in base.columns]
        if cand_moms:
            base["momentum_score"] = base[cand_moms].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            base.drop(columns=[c for c in ["mom_sig","mom_px","momentum_score_prices"] if c in base.columns],
                      inplace=True, errors="ignore")
        else:
            base["momentum_score"] = 0.0

        # ------------------- Estandarizar fundamentales (alias/num/sector) -------------------
        if 'FundamentalStandardizer' not in globals():
            st.error("No se encontró FundamentalStandardizer. Asegúrate de haber pegado el QVM CORE arriba.")
            st.stop()

        base = FundamentalStandardizer().fit_transform(base)

        # Si no hay 'sector' pero sí 'industry', úsala como sector
        if base["sector"].eq("Unknown").all() and "industry" in base.columns:
            sec = base["industry"].astype(str).fillna("Unknown")
            base["sector"] = sec

        # ------------------- Cálculo QVM -------------------
        qvm_df = compute_qvm_scores(
            base,
            w_quality=0.40, w_value=0.25, w_momentum=0.35,
            momentum_col="momentum_score",
            sector_col="sector",
            mcap_col="market_cap"
        )

        # Guardrails especiales megacaps
        qvm_df = apply_megacap_rules(
            qvm_df,
            momentum_col="momentum_score",
            quality_col="quality_adj_neut",
            value_col="value_adj_neut"
        )

        # ------------------- Métricas auxiliares (deuda & accruals) -------------------
        def _pct(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            return s.rank(pct=True, method="average")

        # percentiles intra-sector para depurar si quieres ver dónde falla
        qvm_df["q_pct"]  = qvm_df.groupby("sector")["quality_adj_neut"].transform(_pct).fillna(0.5)
        qvm_df["v_pct"]  = qvm_df.groupby("sector")["value_adj_neut"].transform(_pct).fillna(0.5)
        qvm_df["vfq_pct"] = 0.6*qvm_df["q_pct"] + 0.4*qvm_df["v_pct"]

        # NetDebt/EBITDA (robusto a numpy.float64)
        net_debt = pd.to_numeric(qvm_df.get("net_debt_ttm"), errors="coerce")
        ebitda_ttm = pd.to_numeric(qvm_df.get("ebitda_ttm", qvm_df.get("ebitda_ntm")), errors="coerce").abs()
        qvm_df["ndebt_ebitda"] = (net_debt / (ebitda_ttm + 1e-9)).replace([np.inf, -np.inf], np.nan)

        # Accruals proxy (percentil de NOA; si falta NOA, no bloquea)
        noa = pd.to_numeric(qvm_df.get("noa_ttm"), errors="coerce")
        if noa.isna().all():
            qvm_df["acc_pct"] = 1.0
        else:
            qvm_df["acc_pct"] = noa.rank(pct=True, method="average")

        # ------------------- 1) Guardrails fundamentales -------------------
        gr = pd.Series(True, index=qvm_df.index, name="pass_guardrails")
        gr &= (pd.to_numeric(qvm_df["quality_adj_neut"], errors="coerce") > th_q_neut)
        gr &= (pd.to_numeric(qvm_df["value_adj_neut"],   errors="coerce") > th_v_neut)
        gr &= (qvm_df["ndebt_ebitda"].isna() | (qvm_df["ndebt_ebitda"] <= th_ndebt))
        gr &= (pd.to_numeric(qvm_df["acc_pct"], errors="coerce") >= (th_acc_p/100.0))
        qvm_df["pass_guardrails"] = gr

        # Tabla de rechazados (para depurar)
        rej_cols = [c for c in ["symbol","sector","market_cap","quality_adj_neut","value_adj_neut",
                                "ndebt_ebitda","acc_pct","BreakoutScore"] if c in qvm_df.columns]
        st.caption("🚧 Rechazados por guardrails")
        st.dataframe(qvm_df.loc[~gr, rej_cols].sort_values(["quality_adj_neut","value_adj_neut"], ascending=True),
                     use_container_width=True, hide_index=True)

        # ------------------- 2) Filtro técnico (entre los que pasan guardrails) -------------------
        tech_ok = pd.Series(True, index=qvm_df.index, name="pass_tech")
        tech_ok &= (pd.to_numeric(qvm_df.get("BreakoutScore"), errors="coerce") >= th_bo)
        tech_ok &= (pd.to_numeric(qvm_df.get("hits"), errors="coerce").fillna(0) >= th_hits)
        tech_ok &= (pd.to_numeric(qvm_df.get("RVOL20"), errors="coerce").fillna(0) >= th_rvol)
        qvm_df["pass_tech"] = tech_ok

        eligibles_pre_tech = qvm_df.loc[gr, ["symbol","sector","market_cap","value_adj_neut",
                                             "quality_adj_neut","qvm_score","ndebt_ebitda","acc_pct","vfq_pct"]]
        st.caption("✅ Elegibles por guardrails (antes del técnico)")
        st.dataframe(eligibles_pre_tech.sort_values(["quality_adj_neut","value_adj_neut","qvm_score"], ascending=False),
                     use_container_width=True, hide_index=True)

        # ------------------- 3) Ranking final -------------------
        def _z(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            mu, sd = s.mean(skipna=True), s.std(skipna=True)
            return (s - mu) / (sd if (sd and sd > 0) else 1.0)

        if "BreakoutScore" in qvm_df.columns:
            qvm_z = _z(qvm_df["qvm_score"])
            bo_z  = _z(qvm_df["BreakoutScore"])
            final_alpha = 0.70*qvm_z + 0.30*bo_z
        else:
            final_alpha = qvm_df["qvm_score"].rank(pct=True, method="average")

        qvm_df["final_alpha"]     = final_alpha
        qvm_df["final_alpha_pct"] = pd.to_numeric(final_alpha, errors="coerce").rank(pct=True, method="average")

        pct = qvm_df["final_alpha_pct"].clip(0, 1).fillna(0.5)
        qvm_df["prob_up"] = 1.0 / (1.0 + np.exp(-beta_prob * (pct - 0.5)))

        final_mask = qvm_df["pass_guardrails"] & qvm_df["pass_tech"]
        final_cols = [c for c in [
            "symbol","sector","market_cap","qvm_score","final_alpha","final_alpha_pct","prob_up",
            "value_adj_neut","quality_adj_neut","vfq_pct","momentum_score",
            "BreakoutScore","hits","RVOL20","ClosePos","P52",
            "ndebt_ebitda","acc_pct","mega_exception_ok","quality_too_low"
        ] if c in qvm_df.columns]

        st.metric("Seleccionadas (guardrails + técnico)", int(final_mask.sum()))
        st.dataframe(
            qvm_df.loc[final_mask, final_cols].sort_values(
                ["prob_up","final_alpha","qvm_score"], ascending=False
            ),
            use_container_width=True, hide_index=True
        )

        # ----------- Tabla técnica: ordenar por hits y BreakoutScore -----------
        st.subheader("Orden técnico (hits y BreakoutScore)")
        tech_cols = [c for c in ["symbol","hits","BreakoutScore","RVOL20","ClosePos","P52"] if c in qvm_df.columns]
        tech_table = qvm_df.loc[gr, tech_cols].copy()
        st.dataframe(
            tech_table.sort_values(["hits","BreakoutScore","RVOL20"], ascending=False),
            use_container_width=True, hide_index=True
        )

        # Guarda en sesión
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

    # ---------- Helpers locales (solo para esta pestaña) ----------
    def _to_panel_dict(panel_prices):
        """Acepta dict {sym: df} o DF largo y retorna dict {sym: df con index datetime y col 'close'}."""
        if isinstance(panel_prices, dict):
            out = {}
            for s, df in panel_prices.items():
                if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
                    continue
                dfi = df.copy()
                # asegura índice datetime
                if not isinstance(dfi.index, pd.DatetimeIndex):
                    if "date" in dfi.columns:
                        dfi = dfi.set_index(pd.to_datetime(dfi["date"])).drop(columns=[c for c in ["date"] if c in dfi.columns])
                    else:
                        dfi.index = pd.to_datetime(dfi.index)
                dfi = dfi.sort_index()
                out[s] = dfi[["close"]].dropna()
            return out
        elif isinstance(panel_prices, pd.DataFrame) and not panel_prices.empty:
            df = panel_prices.copy()
            if "symbol" not in df.columns and "ticker" in df.columns:
                df = df.rename(columns={"ticker": "symbol"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["symbol", "date"])
            out = {}
            for s, grp in df.groupby("symbol"):
                gg = grp.copy()
                if "date" in gg.columns:
                    gg = gg.set_index(gg["date"])
                out[s] = gg[["close"]].dropna()
            return out
        return {}

    def _month_ends_index(df: pd.DataFrame) -> pd.DatetimeIndex:
        return df.resample("M").last().index

    def _daily_returns_from_prices(df: pd.DataFrame) -> pd.Series:
        # espera index datetime y columna 'close'
        px = df["close"].astype(float)
        rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return rets

    def _portfolio_backtest(panel_dict: dict,
                            rank_df: pd.DataFrame,
                            rank_col: str = "final_alpha",
                            top_n: int = 10,
                            cost_bps: int = 10,
                            lag_days: int = 0,
                            target_vol: float | None = None,
                            lev_cap: float = 2.0) -> tuple[pd.Series, pd.DataFrame]:
        """
        Backtest simple de portafolio:
        - Cada fin de mes selecciona Top-N por 'rank_col'
        - Equal weight
        - Aplica lag_days a la ejecución
        - Aplica costos por cambio de pesos (bps)
        - (Opcional) Target de volatilidad diario sobre la serie de portafolio (20d rolling), con tope de leverage
        Devuelve: equity (Serie) y tabla de rebalances (DataFrame con pesos en cada mes).
        """
        # Asegura que rank_df tiene symbol y la columna de ranking
        if "symbol" not in rank_df.columns or rank_col not in rank_df.columns:
            return pd.Series(dtype=float), pd.DataFrame()

        # Construimos calendario de rebalanceo a partir de la intersección de meses disponibles
        # Usamos el universo de símbolos con historial
        if not panel_dict:
            return pd.Series(dtype=float), pd.DataFrame()

        # Calendario mensual común (intersección soft)
        any_sym = next(iter(panel_dict))
        cal = _month_ends_index(panel_dict[any_sym])
        # preferimos el calendario del benchmark si lo tienes; aquí usamos el primero

        lag = pd.Timedelta(days=int(lag_days)) if lag_days else pd.Timedelta(0)

        # Serie de retorno diario por símbolo
        daily_map = {s: _daily_returns_from_prices(df) for s, df in panel_dict.items()}

        # Construir DataFrame de retornos diario alineado
        all_rets = pd.DataFrame({s: sr for s, sr in daily_map.items()}).sort_index().fillna(0.0)
        if all_rets.empty:
            return pd.Series(dtype=float), pd.DataFrame()

        # Rebalances mensuales
        month_ends = all_rets.resample("M").last().index
        weights_hist = []
        port_rets = []

        prev_weights = pd.Series(0.0, index=all_rets.columns)

        for i in range(1, len(month_ends)):
            t0, t1 = month_ends[i-1], month_ends[i]

            # Selección Top-N por ranking (estático por ahora; si tu ranking es dinámico, adaptar a fecha)
            top = (rank_df[["symbol", rank_col]]
                   .dropna()
                   .sort_values(rank_col, ascending=False)
                   .head(int(top_n))["symbol"]
                   .tolist())

            # Pesos equal-weight en seleccionados
            current_weights = pd.Series(0.0, index=all_rets.columns, dtype=float)
            if len(top) > 0:
                w = 1.0 / len(top)
                current_weights.loc[[s for s in top if s in current_weights.index]] = w

            # Turnover y costos (costo proporcional al cambio de peso)
            tw = (current_weights - prev_weights).abs().sum() * 0.5  # convención: media del cambio
            cost_rate = (cost_bps / 1e4) * tw

            # Ventana diaria de [t0+lag, t1+lag]
            sl = all_rets.loc[(all_rets.index > t0 + lag) & (all_rets.index <= t1 + lag)]
            if sl.empty:
                continue

            # Retorno diario de portafolio (pesos estáticos dentro del mes)
            pr = (sl * current_weights).sum(axis=1)

            # Aplica costo SOLO el primer día del bloque
            if len(pr) > 0:
                pr.iloc[0] = pr.iloc[0] - cost_rate

            port_rets.append(pr)
            weights_hist.append(current_weights.rename(t1))
            prev_weights = current_weights

        if not port_rets:
            return pd.Series(dtype=float), pd.DataFrame()

        port_rets = pd.concat(port_rets).sort_index()
        equity = (1.0 + port_rets).cumprod()

        # Volatility targeting (opcional) — 20 días rolling
        if target_vol is not None and target_vol > 0:
            roll = port_rets.rolling(20).std().replace(0.0, np.nan)
            ann_vol = roll * np.sqrt(252)
            lev = (target_vol / ann_vol).clip(lower=0.0, upper=float(lev_cap)).fillna(0.0)
            adj_rets = port_rets * lev
            equity = (1.0 + adj_rets).cumprod()

        weights_table = pd.DataFrame(weights_hist) if weights_hist else pd.DataFrame()
        return equity.rename("Portfolio"), weights_table

    try:
        panel_prices = st.session_state.get("panel_prices")
        qvm_df = st.session_state.get("qvm")
        if panel_prices is None or qvm_df is None or qvm_df.empty:
            st.info("Corre **Señales** y **QVM** antes de backtestear.")
            st.stop()

        # ---------- Controles ----------
        left, right = st.columns([0.55, 0.45])
        with left:
            rank_by = st.selectbox("Criterio de ranking para Top-N", ["final_alpha", "prob_up", "qvm_score"], index=0)
            top_n = st.slider("Top-N (portafolio)", 5, 50, 15, 5)
            use_and_bt = st.toggle("Señal MA200 AND Mom 12-1 (para métricas por símbolo)", value=False)
            rebalance_freq = st.selectbox("Frecuencia rebalance (por símbolo)", ["M", "W"], index=0,
                                          help="Solo para backtest por símbolo (función backtest_many). El portafolio usa mensual fijo en este bloque.")
        with right:
            cost_bps = st.slider("Costos (bps por cambio de peso)", 0, 50, 10, 1)
            lag_days = st.slider("Lag de ejecución (días)", 0, 5, 1, 1)
            enable_target_vol = st.toggle("Target de volatilidad (portafolio)", value=False)
            target_vol = st.number_input("Volatilidad anual objetivo (p.ej. 0.15)", value=0.15, step=0.01, format="%.2f") if enable_target_vol else None
            lev_cap = st.number_input("Límite de apalancamiento", value=2.0, step=0.1, format="%.1f") if enable_target_vol else 2.0

        # ---------- Panel en dict ----------
        panel_dict = _to_panel_dict(panel_prices)
        if not panel_dict:
            st.warning("No hay datos de precios adecuados para backtesting.")
            st.stop()

        # ---------- Backtest por símbolo (usa tu backtests.py) ----------
        try:
            # Selección: los símbolos disponibles intersectados con panel
            avail_syms = [s for s in qvm_df["symbol"].dropna().astype(str).unique().tolist() if s in panel_dict]
            metrics_df, curves = backtest_many(
                panel=panel_dict,
                symbols=avail_syms,
                cost_bps=int(cost_bps),
                lag_days=int(lag_days),
                use_and_condition=bool(use_and_bt),
                rebalance_freq=str(rebalance_freq)
            )

            st.markdown("**Métricas por símbolo (backtest_many)**")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error en backtest por símbolo: {e}")
            metrics_df, curves = pd.DataFrame(), {}

        st.markdown("---")

        # ---------- Backtest de Portafolio Top-N ----------
        try:
            # rank_df: usa el último QVM disponible (columns: symbol, final_alpha, prob_up, qvm_score ...)
            rank_cols_needed = {"symbol", rank_by}
            if not rank_cols_needed.issubset(qvm_df.columns):
                st.warning(f"No encuentro columnas {rank_cols_needed} en QVM.")
                st.stop()

            # filtra a símbolos con precios
            rank_df = qvm_df.loc[qvm_df["symbol"].isin(panel_dict.keys()), ["symbol", rank_by]].dropna()
            equity, wtable = _portfolio_backtest(
                panel_dict=panel_dict,
                rank_df=rank_df,
                rank_col=rank_by,
                top_n=int(top_n),
                cost_bps=int(cost_bps),
                lag_days=int(lag_days),
                target_vol=(float(target_vol) if enable_target_vol else None),
                lev_cap=float(lev_cap)
            )

            if equity.empty:
                st.warning("No se pudo construir la curva del portafolio (revisa datos).")
            else:
                st.markdown("**Portafolio Top-N (equal-weight)**")
                st.line_chart(equity, use_container_width=True)
                # Resumen rápido
                pr = equity.pct_change().dropna()
                cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (252/len(pr))) - 1 if len(pr) > 0 else 0.0
                vol  = pr.std() * np.sqrt(252) if len(pr) > 0 else 0.0
                shar = (pr.mean()/pr.std()) * np.sqrt(252) if pr.std() > 0 else 0.0
                dd   = (equity / equity.cummax() - 1).min()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CAGR", f"{cagr:.2%}")
                c2.metric("Vol",  f"{vol:.2%}")
                c3.metric("Sharpe", f"{shar:.2f}")
                c4.metric("MaxDD", f"{dd:.2%}")

                with st.expander("Pesos en cada rebalance (Top-N)", expanded=False):
                    st.dataframe(wtable.fillna(0.0), use_container_width=True)

        except Exception as e:
            st.error(f"Error en backtest de portafolio: {e}")

        st.caption("Tip: el backtest por símbolo usa tu `backtest_many`. El del portafolio aplica Top-N por ranking QVM con equal-weight, costos, lag y target de volatilidad opcional.")

    except Exception as e:
        st.error(f"Error en Backtesting: {e}")