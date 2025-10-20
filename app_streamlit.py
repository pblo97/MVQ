import os
import streamlit as st
import pandas as pd
from datetime import date
from qvm_trend.data_io import run_fmp_screener, load_prices_panel, load_benchmark, load_float_map, get_prices_fmp
from qvm_trend.pipeline import run_pipeline_vfq, enrich_with_breakout
from qvm_trend.backtests import backtest_vfq_trend_v2
from qvm_trend.scoring import DEFAULT_TH
from qvm_trend.mc import gbm_paths

st.set_page_config(page_title="QVM + Liquidez + Breakout", layout="wide")

st.sidebar.header("⚙️ Parámetros base")
limit = st.sidebar.number_input("Tamaño screener", 50, 1000, 200, step=50)
min_mcap = st.sidebar.number_input("Min MarketCap", 1e7, 1e11, 5e8, step=1e8, format="%.0f")
ipo_days = st.sidebar.number_input("IPO ≥ días", 0, 2000, 365, step=30)
start = st.sidebar.date_input("Start", value=date(2020,1,1))
end = st.sidebar.date_input("End", value=date.today())
bench_ticker = st.sidebar.selectbox("Benchmark", ["SPY","QQQ","^IPSA"], index=0)

@st.cache_data(show_spinner=False)
def _universe(limit, min_mcap, ipo_days):
    df = run_fmp_screener(limit=limit, min_mcap=min_mcap, ipo_min_days=ipo_days)
    if df.empty:
        st.warning("Screener vacío.")
    return df


def _thresholds_from_ui():
    col1, col2, col3 = st.columns(3)
    with col1:
        rvol_min = st.slider("RVOL mínimo", 1.0, 5.0, DEFAULT_TH["rvol_min"], 0.1)
        closepos_min = st.slider("ClosePos mínimo", 0.0, 1.0, DEFAULT_TH["closepos_min"], 0.05)
        p52_min = st.slider("Cercanía 52W (P52)", 0.80, 1.05, DEFAULT_TH["p52_min"], 0.01)
    with col2:
        ud_min = st.slider("Up/Down Vol Ratio", 0.5, 3.0, DEFAULT_TH["ud_vol_min"], 0.05)
        atr_pct_min = st.slider("ATR pct rank (12m)", 0.0, 1.0, DEFAULT_TH["atr_pct_min"], 0.05)
        rs_slope_min = st.slider("Pendiente RS(MA20) mín.", -1.0, 1.0, DEFAULT_TH["rs_slope_min"], 0.05)
    with col3:
        float_vel_min = st.slider("FloatVelocity (%/día)", 0.0, 0.05, DEFAULT_TH["float_vel_min"], 0.001)
        min_score = st.slider("Score de entrada mínimo", 0.0, 1.0, 0.6, 0.05)
    th = dict(DEFAULT_TH)
    th.update(dict(rvol_min=rvol_min, closepos_min=closepos_min, p52_min=p52_min,
                   ud_vol_min=ud_min, atr_pct_min=atr_pct_min, rs_slope_min=rs_slope_min,
                   float_vel_min=float_vel_min))
    return th, min_score


tab1, tab2, tab3, tab4 = st.tabs(["Universo", "Filtros", "Señales & Gráficos", "Backtest & Estadística"]) 

with tab1:
    st.subheader("1) Universo por Screener (FMP)")
    uni = _universe(limit, min_mcap, ipo_days)
    st.dataframe(uni[["symbol","sector","marketCap","exchange"]].head(500), use_container_width=True)

with tab2:
    st.subheader("2) Ajusta los umbrales del breakout")
    th, min_score = _thresholds_from_ui()
    st.info("Los umbrales se aplican en la pestaña 3 para calcular BreakoutScore y EntrySignal.")

with tab3:
    st.subheader("3) Señales de entrada y calidad del breakout")
    if uni.empty:
        st.stop()
    df_vfq, df_vfq_sel, cartera = run_pipeline_vfq(uni, start=start.isoformat(), end=end.isoformat())
    if cartera is None or cartera.empty:
        st.info("Sin finalistas tras filtro de tendencia. Ajusta parámetros o periodo.")
        st.stop()
    symbols = cartera["symbol"].unique().tolist()
    panel = load_prices_panel(symbols, start.isoformat(), end.isoformat())
    bench = load_benchmark(bench_ticker, start.isoformat(), end.isoformat())
    float_map = load_float_map(symbols)

    enriched = enrich_with_breakout(cartera, panel, benchmark_series=bench, float_map=float_map, th=th, min_score=min_score)
    st.dataframe(enriched.sort_values(["EntrySignal","BreakoutScore"], ascending=[False, False]), use_container_width=True)

    sym = st.selectbox("Ver símbolo", symbols)
    if sym:
        dfp = panel.get(sym)
        st.line_chart(dfp[["close"]])
        # Monte Carlo (GBM) opcional
        st.markdown("### Monte Carlo (20 días, GBM)")
        paths = gbm_paths(dfp["close"], horizon_days=20, n_sims=1000)
        st.line_chart(paths)

with tab4:
    st.subheader("4) Backtest (tendencia) & Métricas")
    if uni.empty:
        st.stop()
    df_vfq, df_vfq_sel, cartera = run_pipeline_vfq(uni, start=start.isoformat(), end=end.isoformat())
    if cartera is None or cartera.empty:
        st.info("No hay símbolos elegibles.")
        st.stop()
    universo = cartera[["symbol"]].drop_duplicates()
    eq, rets, summary = backtest_vfq_trend_v2(
        df_symbols=universo,
        price_loader=get_prices_fmp,
        start=start.isoformat(),
        end=end.isoformat(),
        hold_top_k=None,
        rebalance_freq="M",
        cost_bps=15,
        use_and_condition=False,
        lag_days=60,
        plot=False
    )
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CAGR", f"{summary['CAGR']:.2%}")
    c2.metric("Sharpe (anual)", f"{summary['Sharpe_anual']:.2f}")
    c3.metric("MaxDD", f"{summary['MaxDD']:.2%}")
    c4.metric("N° medio posiciones", f"{summary['N_posiciones_medio']:.1f}")
    st.line_chart(eq.rename("Equity"))

st.caption("Tip: exporta las tablas desde los menús de Streamlit (⋮)")
