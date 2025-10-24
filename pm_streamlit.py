import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

# ==== imports del paquete ====
from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.pm.orchestrator import build_portfolio
from qvm_trend.pm.exits import build_exit_table
from qvm_trend.macro.macro_score import z_to_regime, macro_z_from_series

st.set_page_config(page_title="Gestión de Cartera", page_icon="🧭", layout="wide")

st.title("🧭 Gestión de Cartera — Kelly + Macro")
st.caption("Pesos por Kelly robusto, tilt por calidad, caps por régimen macro, y reglas de salida.")

# ------------------ TABS ------------------
tab_in, tab_macro, tab_port, tab_exits, tab_diag = st.tabs(
    ["Entradas", "Macro", "Cartera", "Salidas", "Diagnóstico"]
)

# ------------------ ENTRADAS ------------------
with tab_in:
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        symbols_txt = st.text_area("Símbolos (coma-separados)", "CHDN,CNR,GOOGL,MMM,PYPL,PZZA,UBER", height=80)
    with c2:
        bench = st.text_input("Benchmark", value="SPY").strip().upper()
        start = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end   = st.date_input("Fin", value=pd.to_datetime(DEFAULT_END).date())
    with c3:
        base_kelly = st.slider("Fracción Kelly base", 0.1, 1.0, 0.5, 0.05)
        pos_cap = st.number_input("Cap por posición", 0.01, 0.10, 0.05, 0.01, format="%.2f")
        beta_cap = st.number_input("Cap ∑(β·w)", 0.25, 2.00, 1.00, 0.05)

    st.markdown("**(Opcional) CSV de calidad (VFQ o QualityScore)**")
    up_vfq = st.file_uploader("vfq.csv (de tu screener)", type=["csv"])

    quality_df = None
    if up_vfq is not None:
        try:
            qdf = pd.read_csv(up_vfq)
            if "symbol" in qdf.columns:
                keep_cols = [c for c in ["symbol","VFQ","QualityScore"] if c in qdf.columns]
                quality_df = qdf[keep_cols].copy()
                st.success(f"Cargado VFQ/Quality: {quality_df.shape}")
            else:
                st.warning("El CSV no tiene columna 'symbol'.")
        except Exception as e:
            st.error(f"No pude leer VFQ: {e}")

# ------------------ MACRO ------------------
with tab_macro:
    st.subheader("Macro Monitor: bundle o slider")
    c1, c2 = st.columns(2)

    with c1:
        up_macro = st.file_uploader("macro_monitor_bundle.csv", type=["csv"])
        macro_z_val = None
        beta_cap_sug = None
        pos_cap_sug  = None
        overlay_gate_series = None
        if up_macro is not None:
            try:
                mb = pd.read_csv(up_macro, index_col=0, parse_dates=True)
                macro_z_val = float(mb.get("macro_z", pd.Series([0])).iloc[-1])
                beta_cap_sug = float(mb.get("beta_cap_sug", pd.Series([np.nan])).iloc[-1])
                pos_cap_sug  = float(mb.get("pos_cap_sug",  pd.Series([np.nan])).iloc[-1])
                overlay_gate_series = mb.get("Overlay_Signal")
                st.success(f"Macro bundle OK (z={macro_z_val:.2f})")
            except Exception as e:
                st.error(f"Error leyendo macro bundle: {e}")

    with c2:
        st.caption("Si no subes bundle, usa el slider:")
        macro_z_slider = st.slider("macro_z (manual)", -2.5, 2.5, 0.0, 0.1)
        st.caption("Tip: puedes obtener macro_z = macro_z_from_series(COMPOSITE_Z).")

    macro_z_eff = macro_z_val if macro_z_val is not None else macro_z_slider
    reg = z_to_regime(float(macro_z_eff))
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("macro_z", f"{reg.z:.2f}")
    k2.metric("Régimen", reg.label)
    k3.metric("M_macro", f"{reg.m_multiplier:.2f}")
    k4.metric("Sugerencia β cap / pos cap", f"{reg.beta_cap:.2f} / {reg.vol_cap:.2f}")

    # Guardar en sesión para pestañas siguientes
    st.session_state["macro_z_eff"] = macro_z_eff
    st.session_state["beta_cap_sug"] = beta_cap_sug
    st.session_state["pos_cap_sug"]  = pos_cap_sug
    st.session_state["overlay_gate_series"] = overlay_gate_series

# ------------------ CARTERA ------------------
with tab_port:
    st.subheader("Pesos y métricas (Kelly + Macro + Quality)")
    symbols = [s.strip().upper() for s in symbols_txt.split(",") if s.strip()]
    if not symbols:
        st.warning("Ingrese símbolos en la pestaña Entradas.")
        st.stop()

    # Caps efectivos = min(usuario, sugerido por régimen si viene del bundle)
    beta_cap_eff = beta_cap
    pos_cap_eff  = pos_cap
    if st.session_state.get("beta_cap_sug") is not None:
        beta_cap_eff = min(beta_cap_eff, float(st.session_state["beta_cap_sug"]))
    if st.session_state.get("pos_cap_sug") is not None:
        pos_cap_eff = min(pos_cap_eff, float(st.session_state["pos_cap_sug"]))

    # Gate táctico con overlay (si el último valor es 0 → bloquear nuevas)
    allow_new_when_z_below = -0.5
    ov = st.session_state.get("overlay_gate_series")
    if ov is not None:
        try:
            if int(pd.Series(ov).astype(int).iloc[-1]) == 0:
                allow_new_when_z_below = 10.0  # bloquea nuevas totalmente
        except Exception:
            pass

    dfw = build_portfolio(
        symbols=symbols,
        bench=bench,
        start=start.isoformat(),
        end=end.isoformat(),
        base_kelly=base_kelly,
        macro_z=float(st.session_state.get("macro_z_eff", 0.0)),
        quality_df=quality_df,
        pos_cap=pos_cap_eff,
        beta_cap_user=beta_cap_eff,
        allow_new_when_z_below=allow_new_when_z_below,
        current_holdings=None  # o lista de tus posiciones ya abiertas
    )

    if dfw.empty:
        st.warning("No se pudieron calcular pesos (verifica precios/fechas).")
        st.stop()

    st.dataframe(dfw, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.bar_chart(dfw.set_index("symbol")["weight"])
        st.caption("Pesos finales (w)")
    with c2:
        st.bar_chart(dfw.set_index("symbol")["beta_w"])
        st.caption("Contribución β·w")

# ------------------ SALIDAS ------------------
with tab_exits:
    st.subheader("Reglas de salida por símbolo")
    try:
        panel = load_prices_panel(symbols + [bench], start.isoformat(), end.isoformat(), cache_key="pm_panel")
        bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())
        table = build_exit_table(
            panel=panel,
            bench_close=None if bench_px is None else bench_px["close"],
            ma_window=200,
            mom_lookback=252,
            review_freq="Q"   # revisión trimestral
        )
        st.dataframe(table, use_container_width=True)
        st.caption("Salida si: rompe MA200 y/o Mom 12-1 < 0 en revisión trimestral; motivos y fecha estimada.")
    except Exception as e:
        st.error(f"Error generando salidas: {e}")

# ------------------ DIAGNÓSTICO ------------------
with tab_diag:
    st.subheader("Equity (pesos estáticos) y diagnóstico")
    try:
        pnl = load_prices_panel(symbols + [bench], start.isoformat(), end.isoformat(), cache_key="pm_panel")
        # equity estática con pesos w
        merged = None
        for s in symbols:
            if s in pnl and "close" in pnl[s].columns:
                r = pnl[s]["close"].pct_change().rename(s)
                merged = r if merged is None else merged.join(r, how="outer")
        merged = merged.dropna(how="all").fillna(0.0)
        weights = dfw.set_index("symbol")["weight"].reindex(merged.columns).fillna(0.0).values
        port_ret = (merged * weights).sum(axis=1)
        bench_ret = None
        bdf = load_benchmark(bench, start.isoformat(), end.isoformat())
        if bdf is not None and "close" in bdf.columns:
            bench_ret = bdf["close"].pct_change().reindex(port_ret.index).fillna(0.0)
        eq = (1+port_ret).cumprod().rename("Portfolio")
        if bench_ret is not None:
            eq_b = (1+bench_ret).cumprod().rename(bench)
            st.line_chart(pd.concat([eq, eq_b], axis=1))
        else:
            st.line_chart(eq)
    except Exception as e:
        st.error(f"Diag error: {e}")
