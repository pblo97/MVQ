# app_streamlit.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # /mount/src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os, streamlit as st, pandas as pd
from datetime import date
# --- IMPORTS ROBUSTOS (funcionan app en raíz o dentro de mvq) ---

    # Caso 1: app en la raíz del proyecto (streamlit run app_streamlit.py)
from qvm_trend.data_io import run_fmp_screener, filter_universe, load_prices_panel, load_benchmark, get_prices_fmp
from qvm_trend.fundamentals import download_fundamentals, build_vfq_scores, download_guardrails_batch, apply_quality_guardrails
from qvm_trend.pipeline import apply_trend_filter, enrich_with_breakout
from qvm_trend.scoring import DEFAULT_TH
from qvm_trend.cache_io import save_df, load_df, save_panel, load_panel


st.set_page_config(page_title="QVM — Screener → Fundamentals → Trend → Breakout", layout="wide")

# --- API key check ---
fmp_key = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
if not fmp_key:
    st.error("Configura FMP_API_KEY en .streamlit/secrets.toml o variable de entorno.")
    st.stop()
os.environ["FMP_API_KEY"] = fmp_key

# --- Sidebar: Universo base ---
st.sidebar.header("Universo (Screener)")
limit = st.sidebar.number_input("Tamaño screener", 50, 1000, 300, step=50)
min_mcap = st.sidebar.number_input("Min MarketCap", 1e7, 1e12, 5e8, step=1e8, format="%.0f")
ipo_days = st.sidebar.number_input("IPO ≥ días", 0, 2000, 365, step=30)
start = st.sidebar.date_input("Start", value=date(2020,1,1))
end = st.sidebar.date_input("End", value=date.today())
bench_ticker = st.sidebar.selectbox("Benchmark", ["SPY","QQQ","^IPSA"], index=0)

# botones de refresco
colb1, colb2, colb3 = st.sidebar.columns(3)
refresh_scr = colb1.button("Refrescar\nScreener")
refresh_funda = colb2.button("Refrescar\nFundamentales")
refresh_px = colb3.button("Refrescar\nPrecios")

cache_tag = f"{start.isoformat()}_{end.isoformat()}_{limit}_{int(min_mcap)}_{ipo_days}"

# 1) Screener (descarga y cache)
st.subheader("1) Screener")
if refresh_scr:
    uni_raw = run_fmp_screener(limit=limit)
    save_df(uni_raw, f"uni_raw_{cache_tag}")
else:
    uni_raw = load_df(f"uni_raw_{cache_tag}")
    if uni_raw is None:
        uni_raw = run_fmp_screener(limit=limit)
        save_df(uni_raw, f"uni_raw_{cache_tag}")

uni = filter_universe(uni_raw, min_mcap=min_mcap, ipo_min_days=ipo_days)
st.write(f"Universo tras filtros de exchange/mcap/IPO: **{len(uni)}** símbolos")
st.dataframe(uni[["symbol","sector","marketCap","exchange"]].head(500), use_container_width=True)

# 2) Fundamentales (set mínimo) + VFQ (cache)
# 2) Fundamentales (set mínimo) + GUARDRAILS + VFQ (cache)
st.subheader("2) Fundamentales (set mínimo) → Guardrails → VFQ")

symbols = uni["symbol"].tolist()
mc_map = {r["symbol"]: float(r["marketCap"]) for _, r in uni[["symbol","marketCap"]].dropna().iterrows()}

# Descarga/carga cache — set mínimo
if refresh_funda:
    df_fund = download_fundamentals(symbols, mc_map, cache_key=cache_tag, force=True)
else:
    df_fund = download_fundamentals(symbols, mc_map, cache_key=cache_tag, force=False)

# Descarga/carga cache — guardrails (anuales + ttm)
from qvm_trend.fundamentals import download_guardrails_batch, apply_quality_guardrails
if refresh_funda:
    df_guard = download_guardrails_batch(symbols, cache_key=cache_tag, force=True)
else:
    df_guard = download_guardrails_batch(symbols, cache_key=cache_tag, force=False)

# Controles de guardrails (UI)
with st.expander("Opciones de Guardrails (calidad)"):
    c1, c2, c3 = st.columns(3)
    use_guard = c1.checkbox("Aplicar guardrails de calidad", True)
    profit_hits_min = c1.slider("Pisos de rentabilidad (hits de 3)", 0, 3, 2, 1)  # EBIT>0, CFO>0, FCF>0
    max_issu = c2.slider("Dilución máx. 12–24m (Δshares)", 0.0, 0.20, 0.03, 0.01)
    max_ag   = c2.slider("Asset growth máx. (y/y)", 0.0, 0.60, 0.20, 0.01)
    max_acc  = c3.slider("Accruals/TA máx.", 0.0, 0.50, 0.10, 0.01)
    max_ndeb = c3.slider("NetDebt/EBITDA máx.", 0.0, 6.0, 3.0, 0.5)

# Merge y aplica guardrails
df_merge = uni[["symbol","sector","marketCap"]].merge(df_fund, on="symbol", how="left").merge(df_guard, on="symbol", how="left")

if use_guard:
    kept, diag = apply_quality_guardrails(
        df_merge,
        require_profit_floor=True,
        profit_floor_min_hits=profit_hits_min,
        max_net_issuance=max_issu,
        max_asset_growth=max_ag,
        max_accruals_ta=max_acc,
        max_netdebt_ebitda=max_ndeb
    )
    st.write(f"Tras guardrails: **{len(kept)}** / {len(df_merge)}")
    with st.expander("Diagnóstico guardrails"):
        st.dataframe(diag[["symbol","profit_hits","guard_profit","net_issuance","guard_issuance",
                           "asset_growth","guard_assets","accruals_ta","guard_accruals",
                           "netdebt_ebitda","guard_leverage","guard_all"]].sort_values("guard_all", ascending=True),
                     use_container_width=True)
    base_for_vfq = kept
else:
    st.info("Guardrails desactivados (se usará todo el universo del screener).")
    base_for_vfq = df_merge

# Ahora calcula VFQ sobre el subset resultante (y cacheado)
df_vfq = build_vfq_scores(base_for_vfq, base_for_vfq)  # pasa universo+fundas ya fusionado

st.write(f"Con cobertura ≥1 métrica VFQ: **{(df_vfq['ValueScore'].notna()).sum()}**")
c1, c2 = st.columns(2)
min_cov = c1.slider("Cobertura fundamentales (≥ métricas)", 1, 6, 2, step=1)
min_vfq_pct = c2.slider("VFQ pct (intra-sector) mínimo", 0.00, 1.00, 0.65, 0.05)

df_vfq_filt = df_vfq[
    (df_vfq["coverage_count"] >= min_cov) &
    (df_vfq["VFQ_pct_sector"] >= min_vfq_pct)
].copy()

st.write(f"Elegibles por VFQ & cobertura: **{len(df_vfq_filt)}**")
st.dataframe(
    df_vfq_filt[["symbol","sector_unified","marketCap_unified","coverage_count",
                 "ValueScore","QualityScore","VFQ","VFQ_pct_sector"]]
    .sort_values(["VFQ","coverage_count"], ascending=[False,False]),
    use_container_width=True
)

# 3) Precios + tendencia (cache de panel)
st.subheader("3) Tendencia (MA200 OR Mom 12–1)")
syms_vfq = df_vfq_filt["symbol"].tolist()

if refresh_px:
    panel = load_prices_panel(syms_vfq, start.isoformat(), end.isoformat(), cache_key=cache_tag, force=True)
else:
    panel = load_prices_panel(syms_vfq, start.isoformat(), end.isoformat(), cache_key=cache_tag, force=False)

eligibles = apply_trend_filter(panel, use_and=False)  # OR al inicio
cartera = pd.DataFrame({"symbol": eligibles})
st.write(f"Finalistas por tendencia: **{len(cartera)}**")

# 4) Breakout (RVOL/ClosePos/P52/…)
st.subheader("4) Breakout — Score y Señal")
# parámetros de breakout
colt1, colt2, colt3 = st.columns(3)
rvol_min = colt1.slider("RVOL mínimo", 1.0, 5.0, DEFAULT_TH["rvol_min"], 0.1)
closepos_min = colt1.slider("ClosePos mínimo", 0.0, 1.0, DEFAULT_TH["closepos_min"], 0.05)
p52_min = colt1.slider("P52 mínimo", 0.80, 1.05, 0.95, 0.01)
ud_min = colt2.slider("Up/Down Vol Ratio 20d", 0.5, 3.0, 1.2, 0.05)
atr_pct_min = colt2.slider("ATR pct rank (12m)", 0.0, 1.0, 0.6, 0.05)
rs_slope_min = colt2.slider("Pendiente RS(MA20)", -1.0, 1.0, 0.0, 0.05)
float_vel_min = colt3.slider("FloatVelocity (%/día)", 0.0, 0.05, 0.01, 0.001)
min_score = colt3.slider("Score de entrada mínimo", 0.0, 1.0, 0.6, 0.05)

TH = dict(DEFAULT_TH)
TH.update(dict(rvol_min=rvol_min, closepos_min=closepos_min, p52_min=p52_min,
               ud_vol_min=ud_min, atr_pct_min=atr_pct_min, rs_slope_min=rs_slope_min,
               float_vel_min=float_vel_min))

# benchmark y enriquecimiento
bench = load_benchmark(bench_ticker, start.isoformat(), end.isoformat(), cache_key=cache_tag, force=False)
float_map = {}  # puedes agregar shares_float si lo deseas
enriched = enrich_with_breakout(cartera, panel, benchmark_series=bench, float_map=float_map, th=TH, min_score=min_score)

if enriched is None or enriched.empty:
    st.info("No hay señales con los umbrales actuales.")
else:
    merged = df_vfq[["symbol","ValueScore","QualityScore","VFQ","VFQ_pct_sector"]].merge(enriched, on="symbol", how="right")

    st.dataframe(merged.sort_values(["EntrySignal","BreakoutScore","VFQ"], ascending=[False,False,False]),
                 use_container_width=True)

    # gráfico rápido
    sym = st.selectbox("Ver símbolo", merged["symbol"].tolist())
    if sym:
        dfp = panel.get(sym)
        if dfp is not None and not dfp.empty:
            st.line_chart(dfp[["close"]])
